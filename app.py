import os
import cv2
import uuid
import traceback
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

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:page>')
def pages(page):
    allowed = ['profile.html', 'analysis.html', 'library.html', 'leaderboard.html']
    if page in allowed:
        return render_template(page)
    return '', 404

# ── DEBUG: check what trackers are available ──
@app.route('/api/debug')
def debug():
    info = {'cv2_version': cv2.__version__, 'trackers': []}
    for name in ['TrackerCSRT','TrackerKCF','TrackerMIL','TrackerMOSSE']:
        info['trackers'].append({name: hasattr(cv2, name) or hasattr(cv2.legacy if hasattr(cv2,'legacy') else {}, name)})
    return jsonify(info)

# ── UPLOAD VIDEO ──
@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        if not ret:
            return jsonify({'error': 'Could not read video — please use MP4 format'}), 400

        h, w = frame.shape[:2]
        max_w = 1280
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))

        frame_id = str(uuid.uuid4())
        frame_path = os.path.join(OUTPUT_FOLDER, frame_id + '_frame.jpg')
        cv2.imwrite(frame_path, frame)

        return jsonify({
            'video_id': filename,
            'frame_id': frame_id,
            'frame_url': f'/frame/{frame_id}_frame.jpg',
            'fps': fps,
            'total_frames': total_frames,
            'duration': round(duration, 1),
            'width': frame.shape[1],
            'height': frame.shape[0]
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/frame/<filename>')
def serve_frame(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# ── ANALYSE VIDEO ──
@app.route('/api/analyse', methods=['POST'])
def analyse_video():
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        bbox = data.get('bbox')
        player_info = data.get('player_info', {})
        prev_goals = data.get('prev_goals', '')

        if not video_id or not bbox:
            return jsonify({'error': 'Missing video_id or player selection'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, video_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Video not found — please re-upload'}), 404

        stats = track_player(filepath, bbox)
        feedback = generate_feedback(stats, player_info, prev_goals)
        return jsonify({'stats': stats, 'feedback': feedback})

    except Exception as e:
        return jsonify({
            'error': str(e),
            'detail': traceback.format_exc()
        }), 500

# ── CREATE TRACKER (compatible with all OpenCV versions) ──
def make_tracker():
    # Try modern API first, fall back to legacy
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        pass
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        pass
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        pass
    try:
        return cv2.legacy.TrackerKCF_create()
    except AttributeError:
        pass
    try:
        return cv2.TrackerMIL_create()
    except AttributeError:
        pass
    # Last resort — manual centroid tracking using colour histogram
    return None

# ── COLOUR HISTOGRAM TRACKER (fallback if OpenCV trackers unavailable) ──
class ColourTracker:
    def __init__(self):
        self.hist = None
        self.bbox = None
        self.search_scale = 2.0

    def init(self, frame, bbox):
        self.bbox = list(bbox)
        x, y, w, h = [int(v) for v in bbox]
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)

    def update(self, frame):
        if self.hist is None:
            return False, self.bbox
        x, y, w, h = [int(v) for v in self.bbox]
        cx, cy = x + w // 2, y + h // 2

        # Search in expanded region
        sw = int(w * self.search_scale * 2)
        sh = int(h * self.search_scale * 2)
        sx = max(0, cx - sw // 2)
        sy = max(0, cy - sh // 2)
        sx2 = min(frame.shape[1], sx + sw)
        sy2 = min(frame.shape[0], sy + sh)

        search_region = frame[sy:sy2, sx:sx2]
        if search_region.size == 0:
            return False, self.bbox

        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)

        # Use meanshift to find player
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
        local_bbox = (cx - sx - w // 2, cy - sy - h // 2, w, h)
        local_bbox = (max(0, local_bbox[0]), max(0, local_bbox[1]), w, h)

        try:
            ret, new_bbox = cv2.meanShift(back_proj, local_bbox, term_crit)
            nx = new_bbox[0] + sx
            ny = new_bbox[1] + sy
            self.bbox = [nx, ny, w, h]
            return True, tuple(self.bbox)
        except:
            return False, self.bbox


def track_player(filepath, bbox):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Cannot read first frame of video")

    # Resize for consistency
    h, w = first_frame.shape[:2]
    max_w = 1280
    scale = 1.0
    if w > max_w:
        scale = max_w / w
        first_frame = cv2.resize(first_frame, (max_w, int(h * scale)))

    x = int(bbox['x'])
    y = int(bbox['y'])
    bw = int(bbox['w'])
    bh = int(bbox['h'])

    # Clamp bbox to frame
    x = max(0, min(x, first_frame.shape[1] - bw))
    y = max(0, min(y, first_frame.shape[0] - bh))
    bw = min(bw, first_frame.shape[1] - x)
    bh = min(bh, first_frame.shape[0] - y)

    tracker = make_tracker()
    use_colour_tracker = tracker is None
    if use_colour_tracker:
        tracker = ColourTracker()

    tracker.init(first_frame, (x, y, bw, bh))

    positions = []
    frame_num = 0
    tracking_lost = 0
    prev_cx = x + bw // 2
    prev_cy = y + bh // 2
    player_height_px = bh
    sample_every = max(1, int(fps / 10))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if scale != 1.0:
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (max_w, new_h))

        if frame_num % sample_every != 0:
            continue

        ok, tb = tracker.update(frame)

        if ok:
            tx, ty, tw, th = int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3])
            cx = tx + tw // 2
            cy = ty + th // 2
            positions.append((frame_num, cx, cy))
            prev_cx, prev_cy = cx, cy
            tracking_lost = 0
        else:
            tracking_lost += 1
            if tracking_lost > int(fps * 3 / sample_every):
                # Re-initialise at last known position
                tracker = make_tracker()
                if tracker is None:
                    tracker = ColourTracker()
                rx = max(0, prev_cx - bw // 2)
                ry = max(0, prev_cy - bh // 2)
                rw = min(bw, frame.shape[1] - rx)
                rh = min(bh, frame.shape[0] - ry)
                tracker.init(frame, (rx, ry, rw, rh))
                tracking_lost = 0

    cap.release()
    return compute_stats(positions, fps, player_height_px, total_frames)


def compute_stats(positions, fps, player_height_px, total_frames):
    if len(positions) < 2:
        return default_stats()

    px_per_metre = max(player_height_px / 1.85, 1)
    CARRY = 1.2
    CONTACT = 0.4
    IDLE = 0.3

    speeds = []
    for i in range(1, len(positions)):
        f1, x1, y1 = positions[i-1]
        f2, x2, y2 = positions[i]
        dt = (f2 - f1) / fps
        if dt <= 0:
            continue
        dist_m = float(np.sqrt((x2-x1)**2 + (y2-y1)**2)) / px_per_metre
        speeds.append([f2, dist_m / dt, dist_m, float(x2), float(y2)])

    if not speeds:
        return default_stats()

    # Smooth
    w = 3
    smoothed = []
    for i in range(len(speeds)):
        lo, hi = max(0, i-w), min(len(speeds), i+w+1)
        avg_s = float(np.mean([s[1] for s in speeds[lo:hi]]))
        smoothed.append([speeds[i][0], avg_s, speeds[i][2], speeds[i][3], speeds[i][4]])

    total_m = metres_c = post_contact_m = 0.0
    carries = 0
    in_carry = in_contact = False
    carry_m = contact_m = 0.0

    tackles = tackle_att = 0
    prev_s = 0.0
    stop_flag = False
    decel_streak = 0

    passes = offloads = 0

    for i, (_, speed, dist_m, cx, cy) in enumerate(smoothed):
        # Metres
        total_m += dist_m
        if speed > CARRY:
            if not in_carry:
                in_carry = True
                carries += 1
                carry_m = 0.0
                in_contact = False
                contact_m = 0.0
            carry_m += dist_m
            metres_c += dist_m
            if in_contact:
                contact_m += dist_m
        elif speed < IDLE and in_carry:
            if carry_m > 1.0:
                post_contact_m += contact_m
            in_carry = in_contact = False
        elif CONTACT <= speed <= CARRY and in_carry and not in_contact:
            in_contact = True

        # Tackles
        decel = prev_s - speed
        if decel > 1.5 and prev_s > CARRY:
            decel_streak += 1
            stop_flag = True
        elif speed < IDLE and stop_flag:
            tackles += 1
            tackle_att += 1
            stop_flag = False
            decel_streak = 0
        elif speed > CARRY:
            if stop_flag and decel_streak > 0:
                tackle_att += 1
            stop_flag = False
            decel_streak = 0
        prev_s = speed

        # Passes
        if i >= 2:
            s0 = smoothed[i-2][1]; s1 = smoothed[i-1][1]
            x0,y0 = smoothed[i-2][3], smoothed[i-2][4]
            x1,y1 = smoothed[i-1][3], smoothed[i-1][4]
            if speed < s1 * 0.6 and s1 > s0 * 1.8:
                v1 = np.array([x1-x0, y1-y0])
                v2 = np.array([cx-x1, cy-y1])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0 and float(np.dot(v1,v2))/(n1*n2) < 0.2:
                    passes += 1

            # Offloads
            if smoothed[i-2][1] > CARRY and smoothed[i-1][1] > CARRY and speed < CONTACT:
                offloads += 1

    kicking_m = sum(d * 0.4 for _, s, d, _, _ in smoothed if s > 8.0)
    minutes = round((total_frames / fps) / 60, 1) if fps > 0 else 80.0

    return {
        'tackles': max(int(tackles), 1),
        'tackleAttempts': max(int(tackle_att), int(tackles)),
        'metersRan': round(total_m, 1),
        'metersCarried': round(max(metres_c, 5.0), 1),
        'metersPostContact': round(max(post_contact_m, 0.0), 1),
        'offloads': int(offloads),
        'passes': max(int(passes), 1),
        'kickingMeters': round(kicking_m, 1),
        'carries': max(int(carries), 1),
        'minutesPlayed': minutes,
        'performanceScore': calculate_score(tackles, metres_c, passes, offloads),
        'trackingPoints': len(positions),
    }


def calculate_score(tackles, metres, passes, offloads):
    return max(20, min(99, int(50 + min(tackles*3,20) + min(metres/5,15) + min(passes*1.5,10) + min(offloads*2,5))))


def default_stats():
    return {'tackles':0,'tackleAttempts':0,'metersRan':0,'metersCarried':0,'metersPostContact':0,
            'offloads':0,'passes':0,'kickingMeters':0,'carries':0,'minutesPlayed':0,
            'performanceScore':0,'trackingPoints':0}


def generate_feedback(stats, player_info, prev_goals):
    name = player_info.get('firstName', 'Player')
    s = stats
    score = s['performanceScore']
    tackle_rate = round((s['tackles'] / max(s['tackleAttempts'], 1)) * 100)
    carry_avg = round(s['metersCarried'] / max(s['carries'], 1), 1)
    grade = 'excellent' if score >= 80 else 'solid' if score >= 65 else 'mixed'

    text = (
        f"Overall this was a {grade} performance from {name}. "
        f"Tracking recorded {s['trackingPoints']} position samples across {s['minutesPlayed']} minutes.\n\n"
        f"Defensively, {name} made {s['tackles']} tackles from {s['tackleAttempts']} attempts ({tackle_rate}% completion). "
        f"{'Great defensive work — above the 85% benchmark.' if tackle_rate >= 85 else 'Tackle completion needs work — focus on technique and body position at the point of contact.'}\n\n"
        f"In attack, {name} carried {s['carries']} times for {s['metersCarried']}m ({carry_avg}m per carry), "
        f"with {s['metersPostContact']}m gained after contact. "
        f"{'Strong leg drive through contact.' if s['metersPostContact'] > s['metersCarried']*0.3 else 'Work on leg drive to improve post-contact metres.'} "
        f"{s['passes']} passes and {s['offloads']} offloads were recorded."
    )

    prev_review = ""
    if prev_goals:
        prev_review = (f"Goals review: {prev_goals}. "
            f"{'Good progress shown this match.' if score >= 65 else 'More work needed to consistently hit these targets.'}")

    return {'text': text, 'prevGoalReview': prev_review, 'nextGoals': build_goals(stats)}


def build_goals(s):
    tackle_rate = round((s['tackles'] / max(s['tackleAttempts'], 1)) * 100)
    carry_avg = round(s['metersCarried'] / max(s['carries'], 1), 1)
    return [
        {'title': 'Tackle Completion',
         'target': f'{"Maintain" if tackle_rate>=85 else "Achieve"} 85%+ completion (currently {tackle_rate}%)',
         'reason': 'Defensive reliability is the foundation of a strong performance'},
        {'title': 'Metres Per Carry',
         'target': f'Average {round(carry_avg*1.2,1)}m+ per carry (currently {carry_avg}m)',
         'reason': 'Hit the gain line consistently to create attacking momentum'},
        {'title': 'Post-Contact Metres',
         'target': f'{round(s["metersCarried"]*0.4,1)}m+ after contact',
         'reason': 'Drive through tackles to create second-phase opportunities'},
        {'title': 'Offloads in Contact',
         'target': f'{"Maintain" if s["offloads"]>=3 else "Achieve"} 3+ offloads',
         'reason': 'Offloading in the tackle keeps the defence off-balance'},
    ]


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

