
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time
from pathlib import Path
from datetime import datetime
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

CAMERA_W = 640
CAMERA_H = 480

CURSOR_MARGIN = 0.05
CURSOR_SMOOTHING = 0.30

# Independent cooldowns
CLICK_COOLDOWN = 0.8
SWIPE_COOLDOWN = 1.3

EDGE_DWELL_SECONDS = 3.0
ZOOM_IN_COOLDOWN = 1.0
ZOOM_OUT_COOLDOWN = 1.0
SCREENSHOT_COOLDOWN = 4.0

# Finger detection
FINGER_EXTENSION_THRESHOLD = 0.12

# Pinch thresholds, relative to palm width
CLICK_NEAR = 0.28
CLICK_RELEASE = 0.38

# Zoom thresholds
ZOOM_IN_NEAR = 0.50
ZOOM_OUT_FAR = 0.65

# Swipe thresholds
SWIPE_DISTANCE = 0.18
SWIPE_MIN_TIME = 0.12
SWIPE_MAX_TIME = 0.60

# Edge detection
EDGE_MARGIN = 0.12
EDGE_MOVEMENT_LIMIT = 0.035
EDGE_MIN_ACTIVE_FINGERS = 3

SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

# ============================================================
# COLOURS (BGR)
# ============================================================

COLORS = {
    "THUMB":  (0, 255, 255),
    "INDEX":  (0, 0, 255),
    "MIDDLE": (255, 0, 0),
    "RING":   (203, 192, 255),
    "PINKY":  (255, 0, 255)
}

TIP_IDS = {
    "THUMB": 4,
    "INDEX": 8,
    "MIDDLE": 12,
    "RING": 16,
    "PINKY": 20
}

# DIP landmarks: the third point from the finger base.
DIP_IDS = {
    "INDEX": 7,
    "MIDDLE": 11,
    "RING": 15,
    "PINKY": 19
}

# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.60
)

# ============================================================
# STATE
# ============================================================

paused = False

cursor_previous = None

last_click_time = 0.0
last_swipe_time = 0.0
last_zoom_in_time = 0.0
last_zoom_out_time = 0.0
last_screenshot_time = 0.0

last_click_gesture = None
last_fist_state = False

swipe_history = deque(maxlen=12)
palm_path = deque(maxlen=35)

previous_time = time.time()
fps = 0.0

# Edge hold
edge_direction = None
edge_started = None
edge_start_position = None
edge_key_held = None

# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_width(lm):
    return distance(lm[5], lm[17])


def normalized_distance(a, b, width):
    if width < 0.001:
        return 99.0
    return distance(a, b) / width


def get_pinch_distances(lm):
    width = palm_width(lm)

    d_ti = normalized_distance(lm[4], lm[8], width)
    d_tm = normalized_distance(lm[4], lm[12], width)
    d_im = normalized_distance(lm[8], lm[12], width)

    return d_ti, d_tm, d_im


# ============================================================
# PALM DIRECTION
# ============================================================

def get_palm_direction(lm):

    wrist = lm[0]
    bases = [lm[5], lm[9], lm[13], lm[17]]

    center_x = sum(p.x for p in bases) / 4
    center_y = sum(p.y for p in bases) / 4

    dx = center_x - wrist.x
    dy = center_y - wrist.y

    length = math.hypot(dx, dy)

    if length < 0.001:
        return 0.0, -1.0

    return dx / length, dy / length


# ============================================================
# THUMB INSIDE / OUTSIDE PALM
# ============================================================

def get_thumb_outside(lm):

    # Approximate palm boundary.
    palm_ids = [0, 1, 5, 9, 13, 17]

    polygon = np.array(
        [
            (int(lm[i].x * CAMERA_W), int(lm[i].y * CAMERA_H))
            for i in palm_ids
        ],
        dtype=np.int32
    )

    thumb_point = (
        int(lm[4].x * CAMERA_W),
        int(lm[4].y * CAMERA_H)
    )

    inside = cv2.pointPolygonTest(
        polygon,
        thumb_point,
        False
    ) >= 0

    return not inside


# ============================================================
# ACTIVE FINGER DETECTION
# ============================================================

def get_active_fingers(lm):

    width = palm_width(lm)

    if width < 0.001:
        return {}, 0

    direction_x, direction_y = get_palm_direction(lm)

    joints = {
        "INDEX": (6, 8),
        "MIDDLE": (10, 12),
        "RING": (14, 16),
        "PINKY": (18, 20)
    }

    active = {}

    for name, (pip_id, tip_id) in joints.items():

        pip = lm[pip_id]
        tip = lm[tip_id]

        dx = tip.x - pip.x
        dy = tip.y - pip.y

        extension = (
            dx * direction_x + dy * direction_y
        ) / width

        active[name] = (
            extension > FINGER_EXTENSION_THRESHOLD
        )

    # Thumb is active only when outside the palm boundary
    # and extended away from its base.
    thumb_outside = get_thumb_outside(lm)

    thumb_extension = normalized_distance(
        lm[4], lm[2], width
    )

    active["THUMB"] = (
        thumb_outside and thumb_extension > 0.12
    )

    return active, sum(active.values())


# ============================================================
# DRAW FINGERTIPS, PALM LINE AND JOINT CONNECTIONS
# ============================================================

def draw_hand_guides(frame, lm, active):

    h, w = frame.shape[:2]

    # Adjust dot size according to apparent palm width.
    palm_pixels = palm_width(lm) * w
    radius = int(palm_pixels * 0.025)
    radius = clamp(radius, 3, 9)

    # Every fingertip gets a coloured dot.
    for name, tip_id in TIP_IDS.items():

        point = lm[tip_id]
        x = int(point.x * w)
        y = int(point.y * h)

        color = COLORS[name]

        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(
            frame,
            (x, y),
            radius + 2,
            (255, 255, 255),
            1
        )

        state = "ACTIVE" if active.get(name, False) else "FOLDED"

        cv2.putText(
            frame,
            state,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color,
            1
        )

    # Line from index fingertip to bottom of palm (wrist).
    index_tip = lm[8]
    wrist = lm[0]

    cv2.line(
        frame,
        (int(index_tip.x * w), int(index_tip.y * h)),
        (int(wrist.x * w), int(wrist.y * h)),
        (0, 255, 255),
        2
    )

    # Connect the DIP / third points of each finger.
    third_points = [19, 14, 10]

    for i in range(len(third_points) - 1):

        a = lm[third_points[i]]
        b = lm[third_points[i + 1]]

        cv2.line(
            frame,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            (180, 180, 180),
            2
        )

    # Also show the base connections.
    base_ids = [5, 9, 13, 17]

    for i in range(len(base_ids) - 1):

        a = lm[base_ids[i]]
        b = lm[base_ids[i + 1]]

        cv2.line(
            frame,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            (90, 90, 90),
            1
        )


# ============================================================
# ACTIVE-FINGER TRACKING CENTRE
# ============================================================

def get_tracking_center(lm, active):

    points = [lm[0]]

    base_ids = {
        "INDEX": 5,
        "MIDDLE": 9,
        "RING": 13,
        "PINKY": 17
    }

    for name, landmark_id in base_ids.items():

        if active.get(name, False):
            points.append(lm[landmark_id])

    x = sum(p.x for p in points) / len(points)
    y = sum(p.y for p in points) / len(points)

    return x, y


# ============================================================
# X/Y GRAPH
# ============================================================

def draw_xy_graph(frame, px, py):

    h, w = frame.shape[:2]

    center_x = w // 2
    center_y = h // 2

    cv2.line(
        frame,
        (0, center_y),
        (w, center_y),
        (90, 90, 90),
        1
    )

    cv2.line(
        frame,
        (center_x, 0),
        (center_x, h),
        (90, 90, 90),
        1
    )

    x = int(px * w)
    y = int(py * h)

    cv2.line(
        frame,
        (center_x, center_y),
        (x, y),
        (255, 255, 0),
        2
    )

    cv2.circle(frame, (x, y), 7, (255, 255, 0), -1)

    palm_path.append((x, y))

    for i in range(1, len(palm_path)):
        cv2.line(
            frame,
            palm_path[i - 1],
            palm_path[i],
            (255, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"X: {px - 0.5:+.3f}  Y: {0.5 - py:+.3f}",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1
    )


# ============================================================
# GESTURE CLASSIFICATION
# ============================================================

def classify_gesture(lm, active, count):

    d_ti, d_tm, d_im = get_pinch_distances(lm)

    index_up = active["INDEX"]
    middle_up = active["MIDDLE"]
    ring_up = active["RING"]
    pinky_up = active["PINKY"]
    thumb_up = active["THUMB"]

    # --------------------------------------------------------
    # LEFT CLICK: thumb + index pinch.
    # --------------------------------------------------------

    if (
        thumb_up
        and index_up
        and d_ti < CLICK_NEAR
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return "LEFT CLICK"

    # --------------------------------------------------------
    # RIGHT CLICK: thumb + middle pinch.
    # Index, ring and pinky should be folded.
    # --------------------------------------------------------

    if (
        thumb_up
        and middle_up
        and d_tm < CLICK_NEAR
        and not index_up
        and not ring_up
        and not pinky_up
    ):
        return "RIGHT CLICK"

  # --------------------------------------------------------
# ZOOM IN:
# Only index + middle active.
# Thumb, ring and pinky must be inactive.
# --------------------------------------------------------

    if (
    index_up
    and middle_up
    and not ring_up
    and not pinky_up
    and not thumb_up
):
        return "ZOOM IN"


# --------------------------------------------------------
# ZOOM OUT:
# Only index + middle + ring active.
# Thumb and pinky must be inactive.
# --------------------------------------------------------

    if (
    index_up
    and middle_up
    and ring_up
    and not pinky_up
    and not thumb_up
    ):
        return "ZOOM OUT"
    # --------------------------------------------------------
    # CURSOR: index active, thumb not active.
    # --------------------------------------------------------

    if (
        index_up
        and not thumb_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return "CURSOR"

    # Thumb and index extended but not pinched:
    # freeze the cursor in its current position.
    if (
        thumb_up
        and index_up
        and d_ti >= CLICK_NEAR
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return "CURSOR FROZEN"

    # --------------------------------------------------------
    # FIST: no active fingers.
    # --------------------------------------------------------

    if count == 0:
        return "FIST"

    # Multiple fingers for swipe / edge hold.
    # Swipe / scroll only when all five fingers are active.
    if (
    thumb_up
    and index_up
    and middle_up
    and ring_up
    and pinky_up
):
        return "SWIPE"

    return "NEUTRAL"


# ============================================================
# CURSOR
# ============================================================

def move_cursor(lm):

    global cursor_previous

    index = lm[8]

    nx = clamp(
        (index.x - CURSOR_MARGIN) /
        (1 - 2 * CURSOR_MARGIN),
        0, 1
    )

    ny = clamp(
        (index.y - CURSOR_MARGIN) /
        (1 - 2 * CURSOR_MARGIN),
        0, 1
    )

    target_x = int(nx * (SCREEN_W - 1))
    target_y = int(ny * (SCREEN_H - 1))

    if cursor_previous is None:
        smooth_x, smooth_y = target_x, target_y
    else:
        a = CURSOR_SMOOTHING

        smooth_x = int(
            a * cursor_previous[0] + (1 - a) * target_x
        )

        smooth_y = int(
            a * cursor_previous[1] + (1 - a) * target_y
        )

    cursor_previous = (smooth_x, smooth_y)

    pyautogui.moveTo(
        smooth_x,
        smooth_y,
        duration=0,
        _pause=False
    )


# ============================================================
# CLICK
# ============================================================

def handle_click(gesture, now):

    global last_click_time
    global last_click_gesture

    # Fire once when entering a pinch gesture.
    if gesture == last_click_gesture:
        return gesture + " HELD"

    last_click_gesture = gesture

    if now - last_click_time < CLICK_COOLDOWN:
        return "CLICK COOLDOWN"

    if gesture == "LEFT CLICK":
        pyautogui.click(button="left")
    else:
        pyautogui.click(button="right")

    last_click_time = now

    return gesture + " DONE"


# ============================================================
# ZOOM
# ============================================================

def handle_zoom(gesture, now):

    global last_zoom_in_time
    global last_zoom_out_time

    if gesture == "ZOOM IN":

        if now - last_zoom_in_time < ZOOM_IN_COOLDOWN:
            return "ZOOM IN COOLDOWN"

        pyautogui.hotkey("ctrl", "=")
        last_zoom_in_time = now

        return "ZOOM IN ACTION"

    if gesture == "ZOOM OUT":

        if now - last_zoom_out_time < ZOOM_OUT_COOLDOWN:
            return "ZOOM OUT COOLDOWN"

        pyautogui.hotkey("ctrl", "-")
        last_zoom_out_time = now

        return "ZOOM OUT ACTION"

    return "ZOOM"


# ============================================================
# NORMAL SWIPE / SCROLL
# ============================================================

def handle_swipe(px, py, now):

    global last_swipe_time

    swipe_history.append((now, px, py))

    if len(swipe_history) < 2:
        return "TRACKING MOVEMENT"

    first_time, first_x, first_y = swipe_history[0]

    elapsed = now - first_time

    if elapsed > SWIPE_MAX_TIME:

        swipe_history.clear()
        swipe_history.append((now, px, py))

        return "TRACKING MOVEMENT"

    if elapsed < SWIPE_MIN_TIME:
        return "TRACKING MOVEMENT"

    dx = px - first_x
    dy = py - first_y

    if (
        abs(dx) >= SWIPE_DISTANCE
        and abs(dx) > abs(dy) * 1.15
    ):

        if now - last_swipe_time < SWIPE_COOLDOWN:
            return "SWIPE COOLDOWN"

        if dx > 0:
            pyautogui.press("right")
            message = "SLIDE RIGHT"
        else:
            pyautogui.press("left")
            message = "SLIDE LEFT"

        last_swipe_time = now
        swipe_history.clear()

        return message

    if (
        abs(dy) >= SWIPE_DISTANCE
        and abs(dy) > abs(dx) * 1.15
    ):

        if now - last_swipe_time < SWIPE_COOLDOWN:
            return "SCROLL COOLDOWN"

        if dy > 0:
            pyautogui.press("pagedown")
            message = "SCROLL DOWN"
        else:
            pyautogui.press("pageup")
            message = "SCROLL UP"

        last_swipe_time = now
        swipe_history.clear()

        return message

    return "TRACKING MOVEMENT"


# ============================================================
# CONTINUOUS EDGE HOLD
# ============================================================

def get_edge_direction(px, py):

    if py <= EDGE_MARGIN:
        return "up"

    if py >= 1 - EDGE_MARGIN:
        return "down"

    if px <= EDGE_MARGIN:
        return "left"

    if px >= 1 - EDGE_MARGIN:
        return "right"

    return None


def stop_edge_hold():

    global edge_direction
    global edge_started
    global edge_start_position
    global edge_key_held

    if edge_key_held is not None:

        try:
            pyautogui.keyUp(edge_key_held)
        except Exception:
            pass

    edge_direction = None
    edge_started = None
    edge_start_position = None
    edge_key_held = None


def handle_edge_hold(px, py, active_count, now):

    global edge_direction
    global edge_started
    global edge_start_position
    global edge_key_held

    if active_count < EDGE_MIN_ACTIVE_FINGERS:

        stop_edge_hold()
        return None

    direction = get_edge_direction(px, py)

    if direction is None:

        stop_edge_hold()
        return None

    # Entering a new edge starts its own dwell timer.
    if direction != edge_direction:

        stop_edge_hold()

        edge_direction = direction
        edge_started = now
        edge_start_position = (px, py)

        return "EDGE HOLD - WAIT 2s"

    # Check movement from the starting position.
    # Small camera jitter is ignored.
    if edge_start_position is not None:

        movement = math.hypot(
            px - edge_start_position[0],
            py - edge_start_position[1]
        )

        if movement > EDGE_MOVEMENT_LIMIT:

            stop_edge_hold()
            return "HAND MOVED - EDGE STOPPED"

    held_for = now - edge_started

    if held_for < EDGE_DWELL_SECONDS:

        remaining = EDGE_DWELL_SECONDS - held_for

        return f"EDGE HOLD - {remaining:.1f}s"

    # Start a continuous key hold.
    # The operating system repeats the key while it remains held.
    if edge_key_held is None:

        pyautogui.keyDown(direction)
        edge_key_held = direction

    return "CONTINUOUS EDGE " + direction.upper()


# ============================================================
# SCREENSHOT
# ============================================================

def take_screenshot(now):

    global last_screenshot_time

    if now - last_screenshot_time < SCREENSHOT_COOLDOWN:
        return "SCREENSHOT COOLDOWN"

    filename = (
        "GestureHUD_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".png"
    )

    filepath = SCREENSHOT_DIR / filename

    pyautogui.screenshot().save(filepath)

    last_screenshot_time = now

    print("Screenshot saved:", filepath)

    return "SCREENSHOT SAVED"


# ============================================================
# HUD
# ============================================================

def draw_hud(frame, gesture, action, active, count, fps):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (5, 5),
        (w - 5, 155),
        (8, 24, 38),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0
    )

    cv2.putText(
        frame,
        "GESTURE HUD",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 255, 220),
        2
    )

    cv2.putText(
        frame,
        "GESTURE: " + gesture,
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "ACTION: " + action,
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )

    finger_text = " ".join(
        name + (":ON" if active.get(name, False) else ":OFF")
        for name in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    )

    cv2.putText(
        frame,
        f"ACTIVE FINGERS: {count}/5",
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        finger_text,
        (15, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)} | P: PAUSE | Q: QUIT",
        (15, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (0, 255, 255),
        1
    )

    return frame


# ============================================================
# MAIN LOOP
# ============================================================

print("======================================")
print("          GESTURE HUD")
print("======================================")
print("Index only                 = Move cursor")
print("Index + thumb open         = Freeze cursor")
print("Thumb + index pinch        = Left click")
print("Thumb + middle pinch       = Right click")
print("Index + middle             = Zoom in")
print("Index + middle + ring      = Zoom out")
print("Fist                       = Screenshot")
print("3+ active fingers at edge  = Continuous edge hold")
print("Swipe / scroll cooldown    = 2 seconds")
print("Click cooldown             = 0.8 seconds")
print("Zoom cooldown              = 1 second")
print("Screenshot cooldown        = 4 seconds")
print("P = Pause | Q = Quit")
print("======================================")

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame unavailable.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        now = time.time()

        dt = now - previous_time
        previous_time = now

        if dt > 0:
            current_fps = 1 / dt
            fps = (
                current_fps if fps == 0
                else fps * 0.9 + current_fps * 0.1
            )

        gesture = "NO HAND"
        action = "SHOW YOUR HAND"

        active = {
            "THUMB": False,
            "INDEX": False,
            "MIDDLE": False,
            "RING": False,
            "PINKY": False
        }

        count = 0

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            active, count = get_active_fingers(lm)

            draw_hand_guides(frame, lm, active)

            px, py = get_tracking_center(lm, active)

            draw_xy_graph(frame, px, py)

            if paused:

                gesture = "PAUSED"
                action = "PRESS P TO RESUME"

                stop_edge_hold()
                swipe_history.clear()
                cursor_previous = None

            else:

                gesture = classify_gesture(
                    lm,
                    active,
                    count
                )

                # --------------------------------------------
                # LEFT / RIGHT CLICK
                # --------------------------------------------

                if gesture in ("LEFT CLICK", "RIGHT CLICK"):

                    stop_edge_hold()
                    swipe_history.clear()
                    cursor_previous = None

                    action = handle_click(gesture, now)

                # --------------------------------------------
                # ZOOM
                # --------------------------------------------

                elif gesture in ("ZOOM IN", "ZOOM OUT"):

                    stop_edge_hold()
                    swipe_history.clear()
                    cursor_previous = None

                    last_click_gesture = None

                    action = handle_zoom(gesture, now)

                # --------------------------------------------
                # CURSOR
                # --------------------------------------------

                elif gesture == "CURSOR":

                    stop_edge_hold()
                    swipe_history.clear()

                    last_click_gesture = None

                    move_cursor(lm)

                    action = "CURSOR MOVING"

                # --------------------------------------------
                # CURSOR FROZEN
                # --------------------------------------------

                elif gesture == "CURSOR FROZEN":

                    stop_edge_hold()
                    swipe_history.clear()

                    last_click_gesture = None

                    # Deliberately do not move the mouse.
                    action = "CURSOR FROZEN"

                # --------------------------------------------
                # SWIPE / EDGE HOLD
                # --------------------------------------------

                elif gesture == "SWIPE":

                    cursor_previous = None
                    last_click_gesture = None

                    edge_action = handle_edge_hold(
                        px,
                        py,
                        count,
                        now
                    )

                    if edge_action is not None:

                        action = edge_action

                        # Do not run normal swipe detection
                        # while the edge-hold gesture is active.
                        swipe_history.clear()

                    else:

                        action = handle_swipe(
                            px,
                            py,
                            now
                        )

                # --------------------------------------------
                # FIST SCREENSHOT
                # --------------------------------------------

                elif gesture == "FIST":

                    stop_edge_hold()
                    swipe_history.clear()
                    cursor_previous = None

                    if not last_fist_state:
                        action = take_screenshot(now)
                    else:
                        action = "FIST HELD"

                    last_fist_state = True

                # --------------------------------------------
                # NEUTRAL
                # --------------------------------------------

                else:

                    stop_edge_hold()
                    swipe_history.clear()
                    cursor_previous = None

                    last_click_gesture = None
                    last_fist_state = False

                    action = "WAITING FOR GESTURE"

                if gesture != "FIST":
                    last_fist_state = False

        else:

            stop_edge_hold()

            palm_path.clear()
            swipe_history.clear()

            cursor_previous = None

            last_click_gesture = None
            last_fist_state = False

            gesture = "NO HAND"
            action = "SHOW YOUR HAND"

        # --------------------------------------------
        # DISPLAY
        # --------------------------------------------

        frame = draw_hud(
            frame,
            gesture,
            action,
            active,
            count,
            fps
        )

        cv2.imshow(
            "GESTURE HUD - CONTINUOUS EDGE CONTROL",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord("p"):

            paused = not paused

            stop_edge_hold()

            palm_path.clear()
            swipe_history.clear()

            cursor_previous = None

            last_click_gesture = None
            last_fist_state = False

finally:

    stop_edge_hold()

    camera.release()
    hands.close()
    cv2.destroyAllWindows()
