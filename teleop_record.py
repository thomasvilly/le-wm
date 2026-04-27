"""
teleop_record.py — Keyboard Teleoperation + Recording for Dobot Magician
=========================================================================
Replaces scripted data collection with human-in-the-loop teleoperation.
Uses the JOG API for real-time Cartesian control (non-queued / immediate),
reads pose concurrently via GetPose, and saves HDF5 in the same format
as record_simple.py for direct pipeline compatibility.

Controls (shown in an OpenCV window — it must be focused to capture keys):
    W / S   →  Y+ / Y-   (forward / backward)
    A / D   →  X- / X+   (left / right)
    R / F   →  Z+ / Z-   (up / down)
    SPACE   →  Toggle suction cup
    ENTER   →  Finish episode & prompt to save
    ESC     →  Quit without saving current episode
    P       →  Pause/resume recording (robot still controllable)
    H       →  Move to home position (PTP, non-recording)

Architecture:
    The main loop runs at ~20 Hz. Each tick:
      1. Read keyboard state from the OpenCV window
      2. Determine which JOG axis command to send (or Idle if no key)
      3. Send ONE SetJOGCmd  (immediate, non-queued)
      4. Call GetPose to read current XYZ + R
      5. Capture a camera frame
      6. Append (frame, state, action) to the episode buffer
    JOG commands are stateful on the controller: sending JogAPPressed
    starts the axis moving, and it keeps moving until you send JogIdle.
    We re-send the command every tick so the watchdog doesn't stop it,
    and send Idle the moment the key is released.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import h5py
import glob
import threading
import ctypes
import DobotDllType as dType

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  WINDOWS KEY-STATE READER                                          ║
# ║  Uses GetAsyncKeyState to detect if a key is currently HELD DOWN.  ║
# ║  This bypasses the cv2.waitKey single-event problem entirely.      ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Virtual-Key codes (Windows VK_*)
VK_W      = 0x57
VK_A      = 0x41
VK_S      = 0x53
VK_D      = 0x44
VK_R      = 0x52
VK_F      = 0x46
VK_H      = 0x48
VK_P      = 0x50
VK_SPACE  = 0x20
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B

user32 = ctypes.windll.user32

def is_key_held(vk_code):
    """Returns True if the key is CURRENTLY held down (polled, not event-based)."""
    return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0

def is_key_just_pressed(vk_code):
    """Returns True if key was pressed since last call (single-shot trigger)."""
    state = user32.GetAsyncKeyState(vk_code)
    return (state & 0x0001) != 0

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit these to match your setup                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

DATASET_DIR   = "dataset_hdf5/teleop_session"
CAM_INDEX     = 0
EXPOSURE_VAL  = 0            # 0 = auto exposure
RECORD_HZ     = 20           # target recording frequency

# Task prompts are loaded from tasks.json in the script directory.
# Format:  { "1": "pick up the blue block and put it on the plate",
#             "2": "move the block to the left side", ... }
# Keys are what you type to select; values are the instruction strings.
TASKS_FILE    = "tasks.json"
TASKS         = {}  # populated in initialize()

# JOG speeds — tune these for comfortable teleoperation
# Lower = more precise but slower.  Higher = faster but harder to control.
JOG_XY_VEL   = 30.0         # mm/s  for X and Y axes
JOG_XY_ACC   = 30.0         # mm/s² for X and Y axes
JOG_Z_VEL    = 30.0         # mm/s  for Z axis
JOG_Z_ACC    = 30.0         # mm/s² for Z axis
JOG_R_VEL    = 30.0         # °/s   for R (rotation) — not used in teleop but set anyway
JOG_R_ACC    = 30.0         # °/s²  for R
JOG_RATIO    = 100.0        # velocity ratio (percentage), applied on top of axis speeds

# Home position for reset (PTP move)
HOME_X, HOME_Y, HOME_Z = 150.0, 0.0, 50.0

# Safety Z limits (optional soft clamp)
Z_MIN = -80.0
Z_MAX = 120.0

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  JOG COMMAND CODES  (Cartesian mode, isJoint=0)                     ║
# ║  1=X+, 2=X-, 3=Y+, 4=Y-, 5=Z+, 6=Z-, 7=R+, 8=R-,  0=Idle        ║
# ╚══════════════════════════════════════════════════════════════════════╝
JOG_IDLE  = 0
JOG_X_POS = 1
JOG_X_NEG = 2
JOG_Y_POS = 3
JOG_Y_NEG = 4
JOG_Z_POS = 5
JOG_Z_NEG = 6

# Map: Windows VK code → JOG command  (checked every tick via GetAsyncKeyState)
# These are polled as "is this key held down RIGHT NOW?" — no event lag.
HELD_KEY_TO_JOG = {
    VK_D: JOG_Y_POS,    # D → Y+  (forward)
    VK_A: JOG_Y_NEG,    # A → Y-  (backward)
    VK_W: JOG_X_POS,    # W → X+
    VK_S: JOG_X_NEG,    # S → X-
    VK_R: JOG_Z_POS,    # R → Z+  (up)
    VK_F: JOG_Z_NEG,    # F → Z-  (down)
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  GLOBALS                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

api = None
cam = None
suction_on = False


def initialize():
    """Connect to Dobot, configure JOG parameters, open camera, load tasks."""
    global api, cam, TASKS

    # --- Load task prompts from JSON ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_path = os.path.join(script_dir, TASKS_FILE)
    if os.path.exists(tasks_path):
        with open(tasks_path, 'r') as f:
            TASKS = json.load(f)
        print(f"[OK] Loaded {len(TASKS)} tasks from {TASKS_FILE}")
    else:
        # Create a default tasks.json so the user has a template
        TASKS = {
            "1": "pick up the blue block and put it on the plate",
            "2": "pick up the block and place it to the left",
            "3": "pick up the block and place it to the right",
        }
        with open(tasks_path, 'w') as f:
            json.dump(TASKS, f, indent=2)
        print(f"[INFO] No {TASKS_FILE} found — created default with {len(TASKS)} tasks.")
        print(f"       Edit {tasks_path} to add your own.")

    # --- Load DLL & Connect ---
    api = dType.load()
    state = dType.ConnectDobot(api, "", 115200)[0]
    if state != dType.DobotConnect.DobotConnect_NoError:
        print("[FATAL] Could not connect to Dobot. Check USB and power.")
        sys.exit(1)
    print("[OK] Connected to Dobot Magician.")

    dType.ClearAllAlarmsState(api)

    # --- Stop any queued execution (clean slate) ---
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)

    # --- Configure JOG parameters ---
    # Per-axis velocity and acceleration (Cartesian)
    dType.SetJOGCoordinateParams(
        api,
        JOG_XY_VEL, JOG_XY_ACC,   # X
        JOG_XY_VEL, JOG_XY_ACC,   # Y
        JOG_Z_VEL,  JOG_Z_ACC,    # Z
        JOG_R_VEL,  JOG_R_ACC,    # R
        isQueued=0
    )
    # Global velocity/acceleration ratio (scales the above)
    dType.SetJOGCommonParams(api, JOG_RATIO, JOG_RATIO, isQueued=0)

    # Also set PTP params for the Home move
    dType.SetPTPCommonParams(api, 50, 50, isQueued=1)
    dType.SetPTPJumpParams(api, 20, 100, isQueued=1)

    # --- Open Camera ---
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("[FATAL] Cannot open camera index", CAM_INDEX)
        sys.exit(1)
    if EXPOSURE_VAL != 0:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    # Warm up the camera buffer
    for _ in range(10):
        cam.read()

    print("[OK] Camera ready.")
    print()
    print("=" * 60)
    print("  DOBOT TELEOP RECORDER")
    print("=" * 60)
    print("  W/S   = X left/right        A/D   = Y forward/back")
    print("  R/F   = Z up/down           SPACE = toggle suction")
    print("  ENTER = finish episode      ESC   = quit")
    print("  P     = pause recording     H     = go home")
    print("=" * 60)
    print()


def get_state_vector():
    """
    Read current pose and suction state.
    Returns 7D vector matching record_simple.py format:
        [x, y, z, r, 0.0, 0.0, grip]
    where grip = 1.0 if suction on, -1.0 if off.
    """
    pose = dType.GetPose(api)  # [x, y, z, rHead, j1, j2, j3, j4]
    grip_val = 1.0 if suction_on else -1.0
    return [pose[0], pose[1], pose[2], pose[3], 0.0, 0.0, grip_val]


def set_suction(on):
    """Toggle suction cup immediately (non-queued)."""
    global suction_on
    suction_on = on
    dType.SetEndEffectorSuctionCup(api, 1, 1 if on else 0, isQueued=0)


def send_jog(cmd):
    """
    Send a JOG command in Cartesian mode (immediate, non-queued).
    cmd: 0=idle, 1=X+, 2=X-, 3=Y+, 4=Y-, 5=Z+, 6=Z-, 7=R+, 8=R-
    """
    dType.SetJOGCmd(api, 0, cmd, isQueued=0)  # isJoint=0 → Cartesian


def move_home():
    """PTP move to home position (blocking)."""
    print("[INFO] Moving to home...")
    send_jog(JOG_IDLE)  # stop any jog first
    time.sleep(0.1)

    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)

    last_idx = dType.SetPTPCmd(
        api, dType.PTPMode.PTPMOVJXYZMode,
        HOME_X, HOME_Y, HOME_Z, 0, isQueued=1
    )[0]
    dType.SetQueuedCmdStartExec(api)

    while dType.GetQueuedCmdCurrentIndex(api)[0] < last_idx:
        time.sleep(0.1)

    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    print("[OK] At home position.")


def draw_hud(display_frame, state, recording, paused, suction, fps, frame_count):
    """Draw a heads-up display on the camera frame."""
    h, w = display_frame.shape[:2]

    # Semi-transparent black bar at top
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

    # Status text
    x, y, z, r = state[0], state[1], state[2], state[3]
    pos_text = f"X:{x:7.1f}  Y:{y:7.1f}  Z:{z:7.1f}  R:{r:5.1f}"
    cv2.putText(display_frame, pos_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    suc_text = "SUCTION: ON" if suction else "SUCTION: OFF"
    suc_color = (0, 255, 255) if suction else (128, 128, 128)
    cv2.putText(display_frame, suc_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, suc_color, 1)

    if recording and not paused:
        rec_text = f"REC  Frames: {frame_count}  ({fps:.0f} Hz)"
        cv2.putText(display_frame, rec_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    elif paused:
        cv2.putText(display_frame, "PAUSED", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    else:
        cv2.putText(display_frame, "IDLE — Press any movement key to start",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Key legend at bottom
    legend = "W/S:X  A/D:Y  R/F:Z  SPACE:suck  ENTER:save  ESC:quit  P:pause  H:home"
    cv2.putText(display_frame, legend, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return display_frame


def save_hdf5(buffer, instruction, episode_num=None):
    """Save episode buffer in the same format as record_simple.py."""
    if not buffer:
        print("[WARN] Empty buffer, nothing to save.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    if episode_num is None:
        existing = glob.glob(os.path.join(DATASET_DIR, "episode_*.h5"))
        episode_num = len(existing) + 1

    fname = os.path.join(DATASET_DIR, f"episode_{episode_num:03d}.h5")
    print(f"[SAVE] Writing {fname}  ({len(buffer)} frames) ...")
    print(f"       Task: \"{instruction}\"")

    images = np.array([b['top'] for b in buffer])
    states = np.array([b['state'] for b in buffer], dtype=np.float32)
    actions = np.array([b['action'] for b in buffer], dtype=np.float32)

    with h5py.File(fname, 'w') as f:
        f.create_dataset('observations/images/top', data=images, compression="gzip")
        f.create_dataset('observations/state', data=states)
        f.create_dataset('action', data=actions)
        f.attrs['instruction'] = instruction

    print(f"[SAVE] Done. {fname}")


def compute_action(prev_state, curr_state):
    """
    Compute the action vector.

    For VLA fine-tuning, the 'action' at timestep t is typically the
    state (or delta) the robot should move toward. Here we use the NEXT
    state as the action label — i.e., action[t] = state[t+1].

    This matches the convention in record_simple.py where action == state.
    We'll do the same: action[t] = state[t] (current state as action).
    The caller can shift this by one timestep if needed.

    Returns the same 7D format: [x, y, z, r, 0, 0, grip]
    """
    return curr_state.copy()


def run_episode():
    """
    Run one teleoperation episode.
    Returns the frame buffer (list of dicts) or None if cancelled.
    
    Key handling uses GetAsyncKeyState (Windows) to poll whether movement
    keys are CURRENTLY HELD DOWN each tick.  This means:
      - Hold W → robot moves Y+ continuously, no stuttering
      - Release W → next tick sees key is up → sends Idle → robot stops
      - No dependency on cv2.waitKey event timing
    """
    global suction_on

    buffer = []
    recording = False
    paused = False
    active_jog_cmd = JOG_IDLE
    fps = 0.0
    last_state = get_state_vector()

    # Debounce trackers for toggle keys (SPACE, P, H, ENTER, ESC)
    # Prevents repeated triggers when key is held across multiple ticks
    _space_was_down = False
    _p_was_down = False
    _h_was_down = False

    print("\n" + "=" * 60)
    print("  NEW EPISODE — move the robot to the starting position.")
    print("  Recording begins on the first movement key press.")
    print("=" * 60)

    set_suction(False)

    while True:
        loop_start = time.time()

        # ── 1. READ CAMERA ──────────────────────────────────────────
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.02)
            continue

        # ── 2. READ CURRENT STATE ───────────────────────────────────
        curr_state = get_state_vector()

        # ── 3. DISPLAY (cv2.waitKey still needed to pump the GUI) ──
        display = frame.copy()
        display = draw_hud(display, curr_state, recording, paused,
                           suction_on, fps, len(buffer))
        cv2.imshow("Dobot Teleop", display)
        cv2.waitKey(1)  # just pumps the OpenCV event loop for display

        # ── 4. POLL TOGGLE KEYS (single-shot via debounce) ─────────

        # ESC → abort
        if is_key_held(VK_ESCAPE):
            send_jog(JOG_IDLE)
            print("[INFO] Episode cancelled.")
            return None

        # ENTER → finish
        if is_key_held(VK_RETURN):
            send_jog(JOG_IDLE)
            print(f"[INFO] Episode finished. {len(buffer)} frames captured.")
            return buffer

        # SPACE → toggle suction (debounced)
        space_down = is_key_held(VK_SPACE)
        if space_down and not _space_was_down:
            set_suction(not suction_on)
            print(f"[INFO] Suction {'ON' if suction_on else 'OFF'}")
        _space_was_down = space_down

        # P → toggle pause (debounced)
        p_down = is_key_held(VK_P)
        if p_down and not _p_was_down:
            paused = not paused
            if paused:
                send_jog(JOG_IDLE)
                active_jog_cmd = JOG_IDLE
            print(f"[INFO] Recording {'PAUSED' if paused else 'RESUMED'}")
        _p_was_down = p_down

        # H → go home (debounced)
        h_down = is_key_held(VK_H)
        if h_down and not _h_was_down:
            send_jog(JOG_IDLE)
            active_jog_cmd = JOG_IDLE
            move_home()
            last_state = get_state_vector()
            _h_was_down = h_down
            continue
        _h_was_down = h_down

        # ── 5. POLL MOVEMENT KEYS (held = keep moving) ─────────────
        jog_cmd = JOG_IDLE
        for vk, cmd in HELD_KEY_TO_JOG.items():
            if is_key_held(vk):
                jog_cmd = cmd
                break  # first held movement key wins (no multi-axis)

        # Start recording on first movement
        if jog_cmd != JOG_IDLE and not recording:
            recording = True
            print("[REC] Recording started!")

        # ── 6. SAFETY: soft Z clamp ────────────────────────────────
        if curr_state[2] <= Z_MIN and jog_cmd == JOG_Z_NEG:
            jog_cmd = JOG_IDLE
        if curr_state[2] >= Z_MAX and jog_cmd == JOG_Z_POS:
            jog_cmd = JOG_IDLE

        # ── 7. SEND JOG COMMAND ─────────────────────────────────────
        # Send every tick to keep the watchdog alive while key is held.
        # Send Idle only once when transitioning from moving → stopped.
        if jog_cmd != JOG_IDLE:
            send_jog(jog_cmd)
            active_jog_cmd = jog_cmd
        elif active_jog_cmd != JOG_IDLE:
            send_jog(JOG_IDLE)
            active_jog_cmd = JOG_IDLE

        # ── 8. RECORD FRAME ────────────────────────────────────────
        if recording and not paused:
            action = compute_action(last_state, curr_state)
            buffer.append({
                'top': frame,
                'state': curr_state,
                'action': action,
            })

        last_state = curr_state

        # ── 9. TIMING ──────────────────────────────────────────────
        elapsed = time.time() - loop_start
        target_dt = 1.0 / RECORD_HZ
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

        actual_dt = time.time() - loop_start
        if actual_dt > 0:
            instant_fps = 1.0 / actual_dt
            fps = 0.9 * fps + 0.1 * instant_fps


def prompt_task_selection():
    """Show available tasks and let the user pick one. Returns the instruction string."""
    print("\n[TASK] Select the task performed in this episode:")
    keys_sorted = sorted(TASKS.keys(), key=lambda k: (k.isdigit(), k))
    for k in keys_sorted:
        print(f"    [{k}]  {TASKS[k]}")
    print(f"    [c]  Custom (type your own)")

    while True:
        choice = input("[TASK] Enter selection: ").strip()
        if choice.lower() == 'c':
            custom = input("[TASK] Type custom instruction: ").strip()
            if custom:
                return custom
            print("[WARN] Empty instruction, try again.")
        elif choice in TASKS:
            return TASKS[choice]
        else:
            print(f"[WARN] Invalid selection '{choice}'. Try again.")


def main():
    initialize()

    try:
        while True:
            buf = run_episode()

            if buf is None:
                # Episode was cancelled
                ans = input("\n[?] Start new episode? [Y/N]: ").strip().upper()
                if ans != 'Y':
                    break
                continue

            if len(buf) == 0:
                print("[WARN] No frames recorded.")
                continue

            # ── REVIEW: show first/last frame ──
            print(f"\n[REVIEW] Episode: {len(buf)} frames")
            print(f"  Start pos: X={buf[0]['state'][0]:.1f} Y={buf[0]['state'][1]:.1f} Z={buf[0]['state'][2]:.1f}")
            print(f"  End pos:   X={buf[-1]['state'][0]:.1f} Y={buf[-1]['state'][1]:.1f} Z={buf[-1]['state'][2]:.1f}")

            ans = input("[?] Save this episode? [Y]es / [N]o / [R]eplay / [Q]uit: ").strip().upper()

            if ans == 'R':
                # Quick replay of the episode frames
                print("[REPLAY] Press any key to advance, ESC to stop...")
                for i, b in enumerate(buf):
                    disp = b['top'].copy()
                    cv2.putText(disp, f"Frame {i+1}/{len(buf)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Replay", disp)
                    k = cv2.waitKey(0) & 0xFF
                    if k == 27:  # ESC
                        break
                cv2.destroyWindow("Replay")
                ans = input("[?] Save? [Y/N/Q]: ").strip().upper()

            if ans == 'Y':
                instruction = prompt_task_selection()
                save_hdf5(buf, instruction)
            elif ans == 'Q':
                break
            # else: discard and loop

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        # ── CLEANUP ──
        print("[INFO] Cleaning up...")
        send_jog(JOG_IDLE)
        set_suction(False)
        dType.DisconnectDobot(api)
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()
        print("[OK] Done.")


if __name__ == "__main__":
    main()
