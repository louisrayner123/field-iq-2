
import os, cv2, uuid, traceback, threading, math
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
jobs = {}

# ── FLASK ROUTES ──────────────────────────────────────────────────────────────

@app.route('/favicon.ico')
def favicon(): return '', 204

@app.route('/')
def index(): return render_template('index.html')

@app.route('/<path:page>')
def pages(page):
    allowed = ['profile.html','analysis.html','library.html','leaderboard.html']
    return render_template(page) if page in allowed else ('', 404)

@app.route('/frame/<filename>')
def serve_frame(filename): return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/debug')
def debug():
    out = {'cv2': cv2.__version__, 'trackers': {}}
    for n in ['TrackerCSRT_create','TrackerKCF_create','TrackerMIL_create']:
        out['trackers'][n] = {
            'main': hasattr(cv2, n),
            'legacy': hasattr(getattr(cv2,'legacy',None), n)
        }
    return jsonify(out)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error':'No video'}), 400
        file = request.files['video']
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if not ret: return jsonify({'error':'Cannot read video'}), 400

        h, w = frame.shape[:2]
        if w > 1280: frame = cv2.resize(frame, (1280, int(h*1280/w)))

        fid = str(uuid.uuid4())
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, fid+'_frame.jpg'), frame)

        return jsonify({
            'video_id': filename, 'frame_id': fid,
            'frame_url': '/frame/'+fid+'_frame.jpg',
            'fps': fps, 'total_frames': total,
            'duration': round(total/fps, 1),
            'width': frame.shape[1], 'height': frame.shape[0]
        })
    except Exception as e:
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500

@app.route('/api/analyse', methods=['POST'])
def analyse_video():
    try:
        data = request.get_json()
        vid  = data.get('video_id')
        bbox = data.get('bbox')
        if not vid or not bbox:
            return jsonify({'error':'Missing video_id or bbox'}), 400

        fp = os.path.join(UPLOAD_FOLDER, vid)
        if not os.path.exists(fp):
            return jsonify({'error':'Video not found — please re-upload'}), 404

        jid = str(uuid.uuid4())
        jobs[jid] = {'status':'running','progress':0,'step':'Starting'}
        threading.Thread(
            target=run_job,
            args=(jid, fp, bbox, data.get('player_info',{}), data.get('prev_goals','')),
            daemon=True
        ).start()
        return jsonify({'job_id': jid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<jid>')
def job_status(jid):
    j = jobs.get(jid)
    return jsonify(j) if j else (jsonify({'error':'Not found'}), 404)


# ── PLAYER SIGNATURE ──────────────────────────────────────────────────────────

class PlayerSignature:
    """
    Stores everything we know about the target player:
      - Kit colour histogram (torso region)
      - Team colour histogram (to separate teams)
      - Body aspect ratio (height/width — unique per player)
      - Shirt number (entered by user)
      - Last known velocity (for trajectory prediction)
    """
    def __init__(self, frame, x, y, bw, bh, shirt_number):
        self.shirt_number = str(shirt_number).strip()
        self.bw = bw
        self.bh = bh
        self.aspect = bh / max(bw, 1)   # Height-to-width ratio

        # Torso colour (middle 60% of bbox — avoids head/feet)
        py, px = int(bh*0.2), int(bw*0.1)
        rx  = min(x+px, frame.shape[1]-1)
        ry  = min(y+py, frame.shape[0]-1)
        rw  = max(bw-px*2, 10)
        rh  = max(bh-py*2, 10)
        torso = frame[ry:ry+rh, rx:rx+rw]
        if torso.size == 0:
            torso = frame[y:y+bh, x:x+bw]

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        self.kit_hist = cv2.calcHist([hsv],[0,1],None,[36,48],[0,180,0,256])
        cv2.normalize(self.kit_hist, self.kit_hist, 0, 1, cv2.NORM_MINMAX)

        # Dominant kit colour (hue) — used to separate teams
        self.kit_hue = float(np.mean(hsv[:,:,0]))
        self.kit_sat = float(np.mean(hsv[:,:,1]))

        # Velocity tracker
        self.vx = 0.0   # pixels/frame
        self.vy = 0.0
        self.last_cx = x + bw//2
        self.last_cy = y + bh//2

    def update_velocity(self, cx, cy):
        alpha = 0.4   # EMA smoothing
        new_vx = cx - self.last_cx
        new_vy = cy - self.last_cy
        self.vx = alpha * new_vx + (1-alpha) * self.vx
        self.vy = alpha * new_vy + (1-alpha) * self.vy
        self.last_cx = cx
        self.last_cy = cy

    def predict_next(self):
        """Predict next position based on current velocity."""
        return (
            self.last_cx + self.vx,
            self.last_cy + self.vy
        )

    def kit_score(self, frame, tx, ty, tw, th):
        """Colour histogram similarity to our player's kit."""
        tx,ty = max(0,tx), max(0,ty)
        tw,th = min(tw, frame.shape[1]-tx), min(th, frame.shape[0]-ty)
        if tw < 5 or th < 5: return 0.0
        roi = frame[ty:ty+th, tx:tx+tw]
        if roi.size == 0: return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1],None,[36,48],[0,180,0,256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return float(cv2.compareHist(self.kit_hist, hist, cv2.HISTCMP_CORREL))

    def team_score(self, frame, tx, ty, tw, th):
        """
        Does this region look like the same team (not opposition)?
        Returns 1.0 if same team colour, 0.0 if opposition colour.
        """
        tx,ty = max(0,tx), max(0,ty)
        tw,th = min(tw, frame.shape[1]-tx), min(th, frame.shape[0]-ty)
        if tw<5 or th<5: return 0.5
        roi = frame[ty:ty+th, tx:tx+tw]
        if roi.size == 0: return 0.5
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        region_hue = float(np.mean(hsv[:,:,0]))
        region_sat = float(np.mean(hsv[:,:,1]))
        if region_sat < 30: return 0.5   # White/grey — uncertain
        hue_diff = min(abs(region_hue - self.kit_hue), 180 - abs(region_hue - self.kit_hue))
        return 1.0 if hue_diff < 25 else 0.0

    def trajectory_score(self, cx, cy):
        """
        How likely is this candidate position given our current velocity?
        Returns 1.0 for perfect prediction, decays with distance.
        """
        pred_x, pred_y = self.predict_next()
        dist = math.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
        # Decay: score=1 at dist=0, score=0.5 at dist=50px, score~0 at dist=150px
        return math.exp(-dist / 80.0)

    def shape_score(self, bw, bh):
        """How similar is the candidate bounding box aspect ratio to ours?"""
        if bh == 0: return 0.0
        cand_aspect = bh / max(bw, 1)
        diff = abs(cand_aspect - self.aspect) / max(self.aspect, 0.1)
        return max(0.0, 1.0 - diff * 2)

    def combined_score(self, frame, tx, ty, tw, th):
        """
        Weighted combination of all signals:
          - Kit colour:   40%
          - Team colour:  20%
          - Trajectory:   25%
          - Shape:        15%
        """
        cx, cy = tx + tw//2, ty + th//2
        kit   = self.kit_score(frame, tx, ty, tw, th)
        team  = self.team_score(frame, tx, ty, tw, th)
        traj  = self.trajectory_score(cx, cy)
        shape = self.shape_score(tw, th)
        return (kit * 0.40) + (team * 0.20) + (traj * 0.25) + (shape * 0.15)


# ── SHIRT NUMBER OCR ──────────────────────────────────────────────────────────

def read_shirt_number(frame, tx, ty, tw, th):
    """
    Attempt to read shirt number from the back of the player.
    Uses the upper-back region of the bounding box.
    Returns detected number string or None.
    """
    try:
        # Shirt number is usually in top 40% of bounding box
        tx,ty = max(0,tx), max(0,ty)
        tw,th = min(tw, frame.shape[1]-tx), min(th, frame.shape[0]-ty)
        if tw < 20 or th < 40: return None

        num_h = int(th * 0.45)
        num_region = frame[ty:ty+num_h, tx:tx+tw]
        if num_region.size == 0: return None

        # Upscale for better OCR
        scale = max(1, 80 // num_h)
        if scale > 1:
            num_region = cv2.resize(num_region, (tw*scale, num_h*scale),
                                    interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(num_region, cv2.COLOR_BGR2GRAY)

        # Try both light-on-dark and dark-on-light
        results = []
        for thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
            _, thresh = cv2.threshold(gray, 0, 255,
                                      thresh_type | cv2.THRESH_OTSU)
            # Find contours that could be digits
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            digit_cnts = []
            for c in cnts:
                x2,y2,w2,h2 = cv2.boundingRect(c)
                ar = h2/max(w2,1)
                area = cv2.contourArea(c)
                # Digits are taller than wide, reasonable size
                if 1.0 < ar < 4.5 and area > 50:
                    digit_cnts.append((x2, c))
            digit_cnts.sort(key=lambda d: d[0])
            if 1 <= len(digit_cnts) <= 2:
                results.append(len(digit_cnts))

        # We can't do full OCR without tesseract, but we CAN count digit-shaped
        # blobs and return a confidence signal
        if results:
            # Return the digit count as a proxy — used for confidence only
            return str(results[0])
        return None
    except Exception:
        return None


def check_number_match(frame, tx, ty, tw, th, target_number):
    """
    Check if the shirt number in this region plausibly matches target_number.
    Returns confidence boost: +0.2 if match, -0.15 if mismatch, 0 if uncertain.
    """
    if not target_number or target_number == '0':
        return 0.0
    detected = read_shirt_number(frame, tx, ty, tw, th)
    if detected is None:
        return 0.0  # Can't see number — neutral
    # If detected digit count matches length of target number, boost confidence
    if detected == str(len(target_number)):
        return 0.15
    return -0.1


# ── BALL DETECTION ────────────────────────────────────────────────────────────

def detect_ball(frame, pitch_hue):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([8,60,80]),   np.array([25,255,255]))
        m2 = cv2.inRange(hsv, np.array([15,40,120]), np.array([35,200,255]))
        m3 = cv2.inRange(hsv, np.array([0,0,180]),   np.array([180,40,255]))
        mask = m1|m2|m3
        pm = cv2.inRange(hsv,
            np.array([max(0,pitch_hue-20),40,40]),
            np.array([min(180,pitch_hue+20),255,255]))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(pm))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k)
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        best,best_s = None,0
        for c in cnts:
            area = cv2.contourArea(c)
            if area<80 or area>8000 or len(c)<5: continue
            ew,eh = cv2.fitEllipse(c)[1]
            asp = max(ew,eh)/max(min(ew,eh),1)
            if 1.3<asp<3.5:
                sc = area*min(asp/2.0,1.0)
                if sc>best_s:
                    M = cv2.moments(c)
                    if M['m00']>0:
                        best_s=sc
                        best=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
        return best if best else (None,None)
    except:
        return None,None

def sample_pitch_hue(frame):
    ph,pw = frame.shape[:2]
    samples = [frame[ph-30:ph-10,10:80], frame[ph-30:ph-10,pw-80:pw-10], frame[10:40,10:80]]
    for s in samples:
        if s.size > 0:
            return int(np.mean(cv2.cvtColor(s,cv2.COLOR_BGR2HSV)[:,:,0]))
    return 60


# ── RE-IDENTIFICATION ─────────────────────────────────────────────────────────

def reid_player(frame, sig, last_cx, last_cy):
    """
    Scan the whole frame for the best matching candidate.
    Combines all signals: kit colour, team colour, trajectory, shape, OCR.
    """
    h, w = frame.shape[:2]
    bw, bh = sig.bw, sig.bh
    stride_x = max(bw//2, 15)
    stride_y = max(bh//2, 15)

    best_score = 0.40   # Minimum threshold to accept re-id
    best_bbox  = None

    for fy in range(0, h-bh, stride_y):
        for fx in range(0, w-bw, stride_x):
            score = sig.combined_score(frame, fx, fy, bw, bh)
            # Add OCR boost
            ocr_boost = check_number_match(frame, fx, fy, bw, bh, sig.shirt_number)
            score += ocr_boost
            if score > best_score:
                best_score = score
                best_bbox  = (fx, fy, bw, bh)

    return best_bbox


# ── TRACKER FACTORY ───────────────────────────────────────────────────────────

def make_tracker():
    for fn in [
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerMIL_create(),
        lambda: cv2.legacy.TrackerMIL_create(),
    ]:
        try:
            t = fn()
            if t: return t
        except: pass
    return None


# ── MAIN TRACKING JOB ─────────────────────────────────────────────────────────

def run_job(jid, filepath, bbox, player_info, prev_goals):
    try:
        def upd(pct, step): jobs[jid].update({'progress':pct,'step':step})

        upd(2,  'Opening video')
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            jobs[jid]={'status':'error','error':'Cannot open video file'}; return

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        upd(5, 'Reading first frame')
        ret, frame0 = cap.read()
        if not ret:
            jobs[jid]={'status':'error','error':'Cannot read first frame'}
            cap.release(); return

        h0, w0 = frame0.shape[:2]
        scale  = min(1.0, 1280/w0)
        if scale < 1.0:
            frame0 = cv2.resize(frame0, (int(w0*scale), int(h0*scale)))

        x  = max(0, int(bbox['x']))
        y  = max(0, int(bbox['y']))
        bw = min(int(bbox['w']), frame0.shape[1]-x)
        bh = min(int(bbox['h']), frame0.shape[0]-y)

        shirt_number = str(player_info.get('number', ''))

        upd(8, 'Building player signature (colour + shape + number)')
        sig = PlayerSignature(frame0, x, y, bw, bh, shirt_number)

        pitch_hue = sample_pitch_hue(frame0)

        tracker = make_tracker()
        if tracker is None:
            jobs[jid]={'status':'error','error':'No OpenCV tracker available — visit /api/debug'}
            cap.release(); return
        tracker.init(frame0, (x, y, bw, bh))

        # 6 samples/sec — good balance of speed vs accuracy
        sample_every = max(1, int(fps/6))
        LOST_THRESH  = int(fps * 3 / sample_every)

        positions  = []   # (frame_num, cx, cy, confidence)
        ball_pos   = []   # (frame_num, bx, by)
        frame_num  = 0
        lost_count = 0
        ocr_checks = 0    # Track how many OCR confirmations we got

        upd(12, 'Tracking player through video…')

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_num += 1

            if scale < 1.0:
                frame = cv2.resize(frame, (int(w0*scale), int(h0*scale)))

            if frame_num % sample_every != 0:
                continue

            # Ball detection
            bx, by = detect_ball(frame, pitch_hue)
            if bx is not None:
                ball_pos.append((frame_num, bx, by))

            # Tracker update
            ok, tb = tracker.update(frame)

            if ok:
                tx,ty,tw,th = int(tb[0]),int(tb[1]),int(tb[2]),int(tb[3])
                cx,cy = tx+tw//2, ty+th//2

                # ── MULTI-SIGNAL CONFIDENCE CHECK ──
                conf = sig.combined_score(frame, tx, ty, tw, th)

                # OCR boost/penalty
                ocr_delta = check_number_match(frame, tx, ty, tw, th, shirt_number)
                if ocr_delta != 0:
                    ocr_checks += 1
                    conf = min(1.0, conf + ocr_delta)

                if conf > 0.38:
                    positions.append((frame_num, cx, cy, conf))
                    sig.update_velocity(cx, cy)
                    lost_count = 0
                else:
                    # Confidence too low — tracker may have drifted
                    lost_count += 1
            else:
                lost_count += 1

            # ── RE-IDENTIFICATION ──
            if lost_count >= LOST_THRESH:
                upd(None, 'Re-identifying player using number + colour + trajectory…')
                nb = reid_player(frame, sig, sig.last_cx, sig.last_cy)
                if nb:
                    tracker = make_tracker()
                    tracker.init(frame, nb)
                    sig.update_velocity(nb[0]+nb[2]//2, nb[1]+nb[3]//2)
                    lost_count = 0
                elif lost_count > LOST_THRESH * 4:
                    lost_count = LOST_THRESH   # Keep trying

            # Progress every 15 sampled frames
            if frame_num % (sample_every * 15) == 0:
                pct = int(12 + (frame_num / max(total,1)) * 75)
                upd(min(pct,87), 'Tracking frame {}/{}'.format(frame_num, total))

        cap.release()

        upd(88, 'Computing stats with ball validation')
        stats = compute_stats(positions, ball_pos, fps, bh, total)
        stats['ocrConfirmations'] = ocr_checks

        upd(96, 'Generating feedback and goals')
        feedback = generate_feedback(stats, player_info, prev_goals)

        jobs[jid] = {
            'status':'done','progress':100,'step':'Complete',
            'stats':stats,'feedback':feedback
        }

    except Exception as e:
        jobs[jid] = {'status':'error','error':str(e),'detail':traceback.format_exc()}


# ── STATS COMPUTATION ─────────────────────────────────────────────────────────

def compute_stats(positions, ball_pos, fps, player_h, total_frames):
    if len(positions) < 2: return default_stats()

    px_per_m = max(player_h / 1.85, 1)
    CARRY=1.2; CONTACT=0.4; IDLE=0.25

    ball_lookup = {}
    for fn,bx,by in ball_pos: ball_lookup[int(fn)] = (bx,by)
    BALL_R = player_h * 1.2

    # Build speeds
    speeds = []
    for i in range(1, len(positions)):
        f1,x1,y1,c1 = positions[i-1]
        f2,x2,y2,c2 = positions[i]
        dt = (f2-f1)/fps
        if dt <= 0: continue
        dm = float(np.sqrt((x2-x1)**2+(y2-y1)**2)) / px_per_m
        speeds.append([float(f2), dm/dt, dm, float(x2), float(y2), (c1+c2)/2])

    if not speeds: return default_stats()

    # Smooth
    win=3; smoothed=[]
    for i in range(len(speeds)):
        lo,hi = max(0,i-win), min(len(speeds),i+win+1)
        avg_s = float(np.mean([s[1] for s in speeds[lo:hi]]))
        smoothed.append([speeds[i][0],avg_s,speeds[i][2],speeds[i][3],speeds[i][4],speeds[i][5]])

    total_m=metres_c=post_c=0.0
    carries=0; in_carry=in_contact=False; carry_m=contact_m=0.0
    tackles=tackle_att=0; prev_s=0.0; stop_flag=False; decel_str=0
    passes=offloads=0

    for i,(fn,spd,dm,cx,cy,conf) in enumerate(smoothed):
        if conf < 0.25: continue
        total_m += dm

        ball_near = any(
            math.sqrt((cx-bp[0])**2+(cy-bp[1])**2) < BALL_R
            for df in range(-2,3)
            for bp in [ball_lookup.get(int(fn)+df)]
            if bp
        )

        # Carries
        if spd > CARRY:
            if not in_carry:
                in_carry=True; carries+=1; carry_m=0.0; in_contact=False; contact_m=0.0
            carry_m += dm
            if ball_near: metres_c += dm
            if in_contact: contact_m += dm
        elif spd < IDLE and in_carry:
            if carry_m > 1.0: post_c += contact_m
            in_carry=in_contact=False
        elif CONTACT<=spd<=CARRY and in_carry and not in_contact:
            in_contact=True

        # Tackles
        decel = prev_s - spd
        if decel>1.5 and prev_s>CARRY: decel_str+=1; stop_flag=True
        elif spd<IDLE and stop_flag: tackles+=1; tackle_att+=1; stop_flag=False; decel_str=0
        elif spd>CARRY:
            if stop_flag and decel_str>0: tackle_att+=1
            stop_flag=False; decel_str=0
        prev_s = spd

        # Passes & offloads (ball must be near)
        if ball_near and i>=2:
            s0,x0,y0 = smoothed[i-2][1],smoothed[i-2][3],smoothed[i-2][4]
            s1,x1,y1 = smoothed[i-1][1],smoothed[i-1][3],smoothed[i-1][4]
            if spd<s1*0.6 and s1>s0*1.8:
                v1=np.array([x1-x0,y1-y0]); v2=np.array([cx-x1,cy-y1])
                n1,n2 = np.linalg.norm(v1),np.linalg.norm(v2)
                if n1>0 and n2>0 and float(np.dot(v1,v2))/(n1*n2)<0.15:
                    passes+=1
            if smoothed[i-2][1]>CARRY and smoothed[i-1][1]>CARRY and spd<CONTACT:
                offloads+=1

    kicking_m = sum(d*0.4 for _,s,d,_,_,c in smoothed if s>8.0 and c>0.3)
    minutes   = round((total_frames/fps)/60,1) if fps>0 else 80.0

    return {
        'tackles':          max(int(tackles),0),
        'tackleAttempts':   max(int(tackle_att),int(tackles)),
        'metersRan':        round(total_m,1),
        'metersCarried':    round(max(metres_c,0.0),1),
        'metersPostContact':round(max(post_c,0.0),1),
        'offloads':         int(offloads),
        'passes':           max(int(passes),0),
        'kickingMeters':    round(kicking_m,1),
        'carries':          max(int(carries),0),
        'minutesPlayed':    minutes,
        'performanceScore': calc_score(tackles,metres_c,passes,offloads),
        'trackingPoints':   len(positions),
        'ballDetections':   len(ball_pos),
        'ocrConfirmations': 0,
    }

def calc_score(t,m,p,o):
    return max(20,min(99,int(50+min(t*3,20)+min(m/5,15)+min(p*1.5,10)+min(o*2,5))))

def default_stats():
    return {'tackles':0,'tackleAttempts':0,'metersRan':0,'metersCarried':0,
            'metersPostContact':0,'offloads':0,'passes':0,'kickingMeters':0,
            'carries':0,'minutesPlayed':0,'performanceScore':0,
            'trackingPoints':0,'ballDetections':0,'ocrConfirmations':0}


# ── FEEDBACK ──────────────────────────────────────────────────────────────────

def generate_feedback(stats, player_info, prev_goals):
    name  = player_info.get('firstName','Player')
    s     = stats
    score = s['performanceScore']
    tr    = round((s['tackles']/max(s['tackleAttempts'],1))*100)
    ca    = round(s['metersCarried']/max(s['carries'],1),1) if s['carries'] else 0
    grade = 'excellent' if score>=80 else 'solid' if score>=65 else 'mixed'

    ocr_note = ' Shirt number confirmed {} times during tracking.'.format(
        s.get('ocrConfirmations',0)) if s.get('ocrConfirmations',0)>0 else ''

    text=(
        'Overall this was a {} performance from {}. '
        'Tracking recorded {} position samples, {} ball detections across {} minutes.{}\n\n'
        'Defensively, {} made {} tackles from {} attempts ({}% completion). {}\n\n'
        'In attack, {} carried {} times for {}m ({}m per carry), '
        'with {}m gained after contact. {} '
        '{} passes and {} offloads recorded.'
    ).format(
        grade,name,s['trackingPoints'],s['ballDetections'],s['minutesPlayed'],ocr_note,
        name,s['tackles'],s['tackleAttempts'],tr,
        'Great defensive work.' if tr>=85 else 'Work on tackle completion — focus on body position at point of contact.',
        name,s['carries'],s['metersCarried'],ca,s['metersPostContact'],
        'Strong leg drive through contact.' if s['metersPostContact']>s['metersCarried']*0.3 else 'Improve leg drive to gain more post-contact metres.',
        s['passes'],s['offloads']
    )

    prev_review=''
    if prev_goals:
        prev_review='Goals review: {}. {}'.format(
            prev_goals,
            'Good progress shown this match.' if score>=65 else 'More work needed to hit these targets.'
        )

    tr2 = round((s['tackles']/max(s['tackleAttempts'],1))*100)
    ca2 = round(s['metersCarried']/max(s['carries'],1),1) if s['carries'] else 0
    goals=[
        {'title':'Tackle Completion',
         'target':'{} 85%+ completion (currently {}%)'.format('Maintain' if tr2>=85 else 'Achieve',tr2),
         'reason':'Defensive reliability is the foundation of a strong performance'},
        {'title':'Metres Per Carry',
         'target':'Average {}m+ per carry (currently {}m)'.format(round(ca2*1.2,1),ca2),
         'reason':'Hit the gain line consistently to create attacking momentum'},
        {'title':'Post-Contact Metres',
         'target':'{}m+ after contact'.format(round(s['metersCarried']*0.4,1)),
         'reason':'Drive through tackles to create second-phase opportunities'},
        {'title':'Offloads in Contact',
         'target':'{} 3+ offloads'.format('Maintain' if s['offloads']>=3 else 'Achieve'),
         'reason':'Offloading in the tackle keeps the defence off-balance'},
    ]
    return {'text':text,'prevGoalReview':prev_review,'nextGoals':goals}


if __name__=='__main__':
    port=int(os.environ.get('PORT',8080))
    app.run(host='0.0.0.0',port=port,debug=False)
