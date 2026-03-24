import os
import cv2
import uuid
import traceback
import threading
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

# In-memory job store
jobs = {}

# ── ROUTES ──
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

@app.route('/frame/<filename>')
def serve_frame(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/debug')
def debug():
    trackers = []
    for name in ['TrackerCSRT_create','TrackerKCF_create','TrackerMIL_create']:
        has_main = hasattr(cv2, name)
        has_legacy = hasattr(getattr(cv2, 'legacy', None), name) if hasattr(cv2, 'legacy') else False
        trackers.append({name: {'main': has_main, 'legacy': has_legacy}})
    return jsonify({'cv2_version': cv2.__version__, 'trackers': trackers})

# ── UPLOAD VIDEO — returns first frame quickly ──
@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        file = request.files['video']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if not ret:
            return jsonify({'error': 'Could not read video — please use MP4 format'}), 400

        h, w = frame.shape[:2]
        max_w = 1280
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))

        frame_id = str(uuid.uuid4())
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, frame_id + '_frame.jpg'), frame)

        return jsonify({
            'video_id': filename,
            'frame_id': frame_id,
            'frame_url': '/frame/' + frame_id + '_frame.jpg',
            'fps': fps,
            'total_frames': total_frames,
            'duration': round(total_frames / fps, 1) if fps > 0 else 0,
            'width': frame.shape[1],
            'height': frame.shape[0]
        })
    except Exception as e:
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500


# ── START ANALYSIS — fires background thread, returns job_id immediately ──
@app.route('/api/analyse', methods=['POST'])
def analyse_video():
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        bbox = data.get('bbox')
        player_info = data.get('player_info', {})
        prev_goals = data.get('prev_goals', '')

        if not video_id or not bbox:
            return jsonify({'error': 'Missing video_id or player selection box'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, video_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Video not found — please re-upload'}), 404

        job_id = str(uuid.uuid4())
        jobs[job_id] = {'status': 'running', 'progress': 0, 'step': 'Starting…'}

        # Run tracking in background thread — avoids Railway 502 timeout
        t = threading.Thread(
            target=run_tracking_job,
            args=(job_id, filepath, bbox, player_info, prev_goals),
            daemon=True
        )
        t.start()

        return jsonify({'job_id': job_id})

    except Exception as e:
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500


# ── POLL JOB STATUS ──
@app.route('/api/job/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


# ── BACKGROUND TRACKING JOB ──
def run_tracking_job(job_id, filepath, bbox, player_info, prev_goals):
    try:
        jobs[job_id]['step'] = 'Opening video file'
        jobs[job_id]['progress'] = 5

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            jobs[job_id] = {'status': 'error', 'error': 'Cannot open video file'}
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        jobs[job_id]['step'] = 'Reading first frame'
        jobs[job_id]['progress'] = 8

        ret, first_frame = cap.read()
        if not ret:
            jobs[job_id] = {'status': 'error', 'error': 'Cannot read first frame'}
            cap.release()
            return

        # Resize
        h, w = first_frame.shape[:2]
        max_w = 1280
        scale = 1.0
        if w > max_w:
            scale = max_w / w
            first_frame = cv2.resize(first_frame, (max_w, int(h * scale)))

        x = max(0, int(bbox['x']))
        y = max(0, int(bbox['y']))
        bw = int(bbox['w'])
        bh = int(bbox['h'])
        bw = min(bw, first_frame.shape[1] - x)
        bh = min(bh, first_frame.shape[0] - y)
        player_height_px = bh

        jobs[job_id]['step'] = 'Initialising player tracker'
        jobs[job_id]['progress'] = 12

        tracker = make_tracker()
        if tracker is None:
            jobs[job_id] = {'status': 'error', 'error': 'No OpenCV tracker available on this server. Visit /api/debug to check.'}
            cap.release()
            return

        tracker.init(first_frame, (x, y, bw, bh))

        positions = []
        frame_num = 0
        tracking_lost = 0
        prev_cx = x + bw // 2
        prev_cy = y + bh // 2
        sample_every = max(1, int(fps / 8))  # 8 samples/sec — good balance of speed and accuracy

        jobs[job_id]['step'] = 'Tracking player through video'
        jobs[job_id]['progress'] = 15

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if scale != 1.0:
                frame = cv2.resize(frame, (max_w, int(frame.shape[0] * scale)))

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
                if tracking_lost > int(fps * 4 / sample_every):
                    tracker = make_tracker()
                    if tracker:
                        rx = max(0, prev_cx - bw // 2)
                        ry = max(0, prev_cy - bh // 2)
                        rw = min(bw, frame.shape[1] - rx)
                        rh = min(bh, frame.shape[0] - ry)
                        tracker.init(frame, (rx, ry, rw, rh))
                    tracking_lost = 0

            # Update progress
            pct = int(15 + (frame_num / max(total_frames, 1)) * 70)
            jobs[job_id]['progress'] = min(pct, 85)
            jobs[job_id]['step'] = 'Tracking frame {}/{}'.format(frame_num, total_frames)

        cap.release()

        jobs[job_id]['step'] = 'Computing stats from tracking data'
        jobs[job_id]['progress'] = 88

        stats = compute_stats(positions, fps, player_height_px, total_frames)

        jobs[job_id]['step'] = 'Generating AI feedback'
        jobs[job_id]['progress'] = 95

        feedback = generate_feedback(stats, player_info, prev_goals)

        jobs[job_id] = {
            'status': 'done',
            'progress': 100,
            'step': 'Complete',
            'stats': stats,
            'feedback': feedback
        }

    except Exception as e:
        jobs[job_id] = {
            'status': 'error',
            'error': str(e),
            'detail': traceback.format_exc()
        }


# ── TRACKER FACTORY ──
def make_tracker():
    attempts = [
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerMIL_create(),
        lambda: cv2.legacy.TrackerMIL_create(),
    ]
    for fn in attempts:
        try:
            t = fn()
            if t is not None:
                return t
        except Exception:
            continue
    return None


# ── COMPUTE STATS ──
def compute_stats(positions, fps, player_height_px, total_frames):
    if len(positions) < 2:
        return default_stats()

    px_per_m = max(player_height_px / 1.85, 1)
    CARRY = 1.2
    CONTACT = 0.4
    IDLE = 0.3

    # Build speed array
    speeds = []
    for i in range(1, len(positions)):
        f1, x1, y1 = positions[i-1]
        f2, x2, y2 = positions[i]
        dt = (f2 - f1) / fps
        if dt <= 0:
            continue
        dist_m = float(np.sqrt((x2-x1)**2 + (y2-y1)**2)) / px_per_m
        speeds.append([float(f2), dist_m/dt, dist_m, float(x2), float(y2)])

    if not speeds:
        return default_stats()

    # Smooth
    win = 3
    smoothed = []
    for i in range(len(speeds)):
        lo, hi = max(0, i-win), min(len(speeds), i+win+1)
        avg_s = float(np.mean([s[1] for s in speeds[lo:hi]]))
        smoothed.append([speeds[i][0], avg_s, speeds[i][2], speeds[i][3], speeds[i][4]])

    total_m = metres_c = post_c = 0.0
    carries = 0
    in_carry = in_contact = False
    carry_m = contact_m = 0.0

    tackles = tackle_att = 0
    prev_spd = 0.0
    stop_flag = False
    decel_str = 0

    passes = offloads = 0

    for i, (_, spd, dist_m, cx, cy) in enumerate(smoothed):
        total_m += dist_m

        # Carries
        if spd > CARRY:
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
        elif spd < IDLE and in_carry:
            if carry_m > 1.0:
                post_c += contact_m
            in_carry = in_contact = False
        elif CONTACT <= spd <= CARRY and in_carry and not in_contact:
            in_contact = True

        # Tackles
        decel = prev_spd - spd
        if decel > 1.5 and prev_spd > CARRY:
            decel_str += 1
            stop_flag = True
        elif spd < IDLE and stop_flag:
            tackles += 1
            tackle_att += 1
            stop_flag = False
            decel_str = 0
        elif spd > CARRY:
            if stop_flag and decel_str > 0:
                tackle_att += 1
            stop_flag = False
            decel_str = 0
        prev_spd = spd

        # Passes
        if i >= 2:
            s0, x0, y0 = smoothed[i-2][1], smoothed[i-2][3], smoothed[i-2][4]
            s1, x1, y1 = smoothed[i-1][1], smoothed[i-1][3], smoothed[i-1][4]
            if spd < s1 * 0.6 and s1 > s0 * 1.8:
                v1 = np.array([x1-x0, y1-y0])
                v2 = np.array([cx-x1, cy-y1])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0 and float(np.dot(v1, v2)) / (n1*n2) < 0.2:
                    passes += 1

            # Offloads
            if smoothed[i-2][1] > CARRY and smoothed[i-1][1] > CARRY and spd < CONTACT:
                offloads += 1

    kicking_m = sum(d * 0.4 for s in smoothed for d in [s[2]] if s[1] > 8.0)
    minutes = round((total_frames / fps) / 60, 1) if fps > 0 else 80.0

    return {
        'tackles': max(int(tackles), 1),
        'tackleAttempts': max(int(tackle_att), int(tackles)),
        'metersRan': round(total_m, 1),
        'metersCarried': round(max(metres_c, 5.0), 1),
        'metersPostContact': round(max(post_c, 0.0), 1),
        'offloads': int(offloads),
        'passes': max(int(passes), 1),
        'kickingMeters': round(kicking_m, 1),
        'carries': max(int(carries), 1),
        'minutesPlayed': minutes,
        'performanceScore': calc_score(tackles, metres_c, passes, offloads),
        'trackingPoints': len(positions),
    }


def calc_score(tackles, metres, passes, offloads):
    return max(20, min(99, int(50 + min(tackles*3,20) + min(metres/5,15) + min(passes*1.5,10) + min(offloads*2,5))))


def default_stats():
    return {'tackles':0,'tackleAttempts':0,'metersRan':0,'metersCarried':0,
            'metersPostContact':0,'offloads':0,'passes':0,'kickingMeters':0,
            'carries':0,'minutesPlayed':0,'performanceScore':0,'trackingPoints':0}


def generate_feedback(stats, player_info, prev_goals):
    name = player_info.get('firstName', 'Player')
    s = stats
    score = s['performanceScore']
    tackle_rate = round((s['tackles'] / max(s['tackleAttempts'], 1)) * 100)
    carry_avg = round(s['metersCarried'] / max(s['carries'], 1), 1)
    grade = 'excellent' if score >= 80 else 'solid' if score >= 65 else 'mixed'

    text = (
        'Overall this was a {} performance from {}. '
        'Tracking recorded {} position samples across {} minutes.\n\n'
        'Defensively, {} made {} tackles from {} attempts ({}% completion). '
        '{}\n\n'
        'In attack, {} carried {} times for {}m ({}m per carry), '
        'with {}m gained after contact. '
        '{} '
        '{} passes and {} offloads recorded.'
    ).format(
        grade, name,
        s['trackingPoints'], s['minutesPlayed'],
        name, s['tackles'], s['tackleAttempts'], tackle_rate,
        'Great defensive work.' if tackle_rate >= 85 else 'Work on tackle completion — focus on body position at the point of contact.',
        name, s['carries'], s['metersCarried'], carry_avg,
        s['metersPostContact'],
        'Strong leg drive through contact.' if s['metersPostContact'] > s['metersCarried']*0.3 else 'Improve leg drive to gain more post-contact metres.',
        s['passes'], s['offloads']
    )

    prev_review = ''
    if prev_goals:
        prev_review = ('Goals review: {}. {}').format(
            prev_goals,
            'Good progress shown this match.' if score >= 65 else 'More work needed to consistently hit these targets.'
        )

    tackle_rate2 = round((s['tackles'] / max(s['tackleAttempts'], 1)) * 100)
    carry_avg2 = round(s['metersCarried'] / max(s['carries'], 1), 1)

    goals = [
        {'title': 'Tackle Completion',
         'target': '{} 85%+ completion (currently {}%)'.format('Maintain' if tackle_rate2>=85 else 'Achieve', tackle_rate2),
         'reason': 'Defensive reliability is the foundation of a strong performance'},
        {'title': 'Metres Per Carry',
         'target': 'Average {}m+ per carry (currently {}m)'.format(round(carry_avg2*1.2,1), carry_avg2),
         'reason': 'Hit the gain line consistently to create attacking momentum'},
        {'title': 'Post-Contact Metres',
         'target': '{}m+ after contact'.format(round(s['metersCarried']*0.4,1)),
         'reason': 'Drive through tackles to create second-phase opportunities'},
        {'title': 'Offloads in Contact',
         'target': '{} 3+ offloads'.format('Maintain' if s['offloads']>=3 else 'Achieve'),
         'reason': 'Offloading in the tackle keeps the defence off-balance'},
    ]

    return {'text': text, 'prevGoalReview': prev_review, 'nextGoals': goals}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)


