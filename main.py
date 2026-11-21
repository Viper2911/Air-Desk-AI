import math
import os
import shutil
import subprocess
import time
from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- 1. One-Euro Filter for Jitter-Free Tracking ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.1, beta=0.08, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-5)
        dx = (x - self.x_prev) / dt
        edx = self._alpha(self.d_cutoff, dt) * dx + (1 - self._alpha(self.d_cutoff, dt)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(edx)
        x_hat = self._alpha(cutoff, dt) * x + (1 - self._alpha(cutoff, dt)) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, edx, t
        return x_hat


# --- 2. Fedora Wayland, PipeWire & Brightness Controller ---
class FedoraController:
    def __init__(self):
        self.has_ydotool = shutil.which("ydotool") is not None
        self.has_brightnessctl = shutil.which("brightnessctl") is not None
        self.has_playerctl = shutil.which("playerctl") is not None
        self.ydotool_socket = (
            "/run/user/1000/.ydotool_socket"
            if os.path.exists("/run/user/1000/.ydotool_socket")
            else "/tmp/.ydotool_socket"
        )
        self.alt_held = False

    @staticmethod
    def run_cmd(cmd: list[str], extra_env: dict[str, str] | None = None):
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def launch_app(self, candidates: list[str]):
        cmd_str = " || ".join(f"command -v {c} >/dev/null 2>&1 && exec {c}" for c in candidates)
        self.run_cmd(["bash", "-lc", cmd_str])

    def open_brave(self):
        self.launch_app(["brave-browser", "brave"])

    def open_spotify(self):
        cmd_str = (
            "if command -v spotify >/dev/null 2>&1; then exec spotify; "
            "elif flatpak info com.spotify.Client >/dev/null 2>&1; then exec flatpak run com.spotify.Client; "
            "elif command -v gtk-launch >/dev/null 2>&1 && gtk-launch spotify 2>/dev/null; then exit 0; "
            "else exit 1; fi"
        )
        self.run_cmd(["bash", "-lc", cmd_str])

    def play_pause(self):
        if self.has_playerctl:
            self.run_cmd(["playerctl", "--player=spotify,%any", "play-pause"])

    def next_track(self):
        if self.has_playerctl:
            self.run_cmd(["playerctl", "--player=spotify,%any", "next"])

    def prev_track(self):
        if self.has_playerctl:
            self.run_cmd(["playerctl", "--player=spotify,%any", "previous"])

    def change_volume(self, delta_percent: int):
        if delta_percent == 0:
            return
        sign = "+" if delta_percent > 0 else "-"
        self.run_cmd([
            "wpctl", "set-volume", "--limit", "1.0",
            "@DEFAULT_AUDIO_SINK@", f"{abs(delta_percent)}%{sign}"
        ])

    def change_brightness(self, delta_percent: int):
        if not self.has_brightnessctl or delta_percent == 0:
            return
        if delta_percent > 0:
            self.run_cmd(["brightnessctl", "-q", "set", f"{abs(delta_percent)}%+"])
        else:
            self.run_cmd(["brightnessctl", "-q", "set", f"{abs(delta_percent)}%-"])

    def get_brightness_info(self) -> int | None:
        if not self.has_brightnessctl:
            return None
        try:
            cur = int(subprocess.check_output(["brightnessctl", "g"], text=True).strip())
            mx = max(int(subprocess.check_output(["brightnessctl", "m"], text=True).strip()), 1)
            return int(round((cur / mx) * 100))
        except Exception:
            return None

    def begin_window_switcher(self) -> bool:
        if not self.has_ydotool:
            return False
        self.alt_held = True
        self.run_cmd(
            ["ydotool", "key", "--key-delay", "40", "56:1", "15:1", "15:0"],
            extra_env={"YDOTOOL_SOCKET": self.ydotool_socket},
        )
        return True

    def step_window_switcher(self) -> bool:
        if not self.has_ydotool:
            return False
        self.run_cmd(
            ["ydotool", "key", "--key-delay", "40", "15:1", "15:0"],
            extra_env={"YDOTOOL_SOCKET": self.ydotool_socket},
        )
        return True

    def commit_window_switcher(self):
        if self.has_ydotool and self.alt_held:
            self.alt_held = False
            self.run_cmd(
                ["ydotool", "key", "--key-delay", "30", "56:0"],
                extra_env={"YDOTOOL_SOCKET": self.ydotool_socket},
            )

    def get_volume_info(self) -> tuple[int, bool]:
        try:
            out = subprocess.check_output(
                ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], text=True
            ).strip()
            muted = "[MUTED]" in out
            val = float(out.split()[1])
            return int(round(val * 100)), muted
        except Exception:
            return 50, False


# --- 3. Gesture Engine (Volume, Brightness, Window Switcher & Thumb+Pinky 1s Quit) ---
class GestureEngine:
    ARMED_TIMEOUT = 15.0
    SWEEP_MIN_DISTANCE = 0.22
    SWEEP_MAX_WINDOW = 1.0
    WINDOW_STEP_INTERVAL = 0.55
    QUIT_HOLD_DURATION = 1.0

    def __init__(self):
        self.state = "IDLE"
        self.arm_start_time = 0.0
        self.fist_start = 0.0
        self.quit_start = 0.0
        self.quit_last_seen = 0.0
        self.win_pinch_active = False
        self.win_last_step_time = 0.0
        self.win_release_seen_time = 0.0
        self.last_action_time = 0.0
        self.last_active_time = 0.0
        self.vol_anchor_y = None
        self.bright_anchor_y = None
        self.palm_history = deque(maxlen=25)
        self.palm_x_filter = OneEuroFilter(min_cutoff=1.4, beta=0.12)
        self.os_ctrl = FedoraController()

    def _dist(self, lm, a: int, b: int) -> float:
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

    def _hand_scale(self, lm) -> float:
        return max(self._dist(lm, 0, 9), 0.08)

    def _is_open_palm(self, lm) -> bool:
        return all([
            lm[8].y < lm[6].y,
            lm[12].y < lm[10].y,
            lm[16].y < lm[14].y,
            lm[20].y < lm[18].y,
        ])

    def _is_fist(self, lm) -> bool:
        return all([
            lm[8].y > lm[6].y + 0.02,
            lm[12].y > lm[10].y + 0.02,
            lm[16].y > lm[14].y + 0.02,
            lm[20].y > lm[18].y + 0.02,
        ])

    def _check_palm_sweep(self, now: float) -> str | None:
        while len(self.palm_history) > 1 and (now - self.palm_history[0][1] > self.SWEEP_MAX_WINDOW):
            self.palm_history.popleft()

        if len(self.palm_history) < 4:
            return None

        dx = self.palm_history[-1][0] - self.palm_history[0][0]
        if dx <= -self.SWEEP_MIN_DISTANCE:
            return "NEXT"
        elif dx >= self.SWEEP_MIN_DISTANCE:
            return "PREV"
        return None

    def _reset_state_buffers(self, keep_sliders=False):
        self.fist_start = 0.0
        self.quit_start = 0.0
        self.quit_last_seen = 0.0
        self.palm_history.clear()
        if not keep_sliders:
            self.vol_anchor_y = None
            self.bright_anchor_y = None

    def _release_window_switcher_if_needed(self):
        if self.win_pinch_active or self.os_ctrl.alt_held:
            self.os_ctrl.commit_window_switcher()
            self.win_pinch_active = False
            self.win_last_step_time = 0.0
            self.win_release_seen_time = 0.0

    def _check_quit_pinch(self, pinky_pinched: bool, now: float) -> tuple[bool, str | None, str | None]:
        if pinky_pinched:
            self.quit_last_seen = now
            if self.quit_start == 0.0:
                self.quit_start = now
            elapsed = now - self.quit_start
            if elapsed >= self.QUIT_HOLD_DURATION:
                return True, "ACTION: Closing Air-Desk AI...", "QUIT"
            rem = max(0.0, self.QUIT_HOLD_DURATION - elapsed)
            return False, f"HOLD THUMB+PINKY ({rem:.1f}s) -> Quit", "QUIT"
        if self.quit_start != 0.0 and (now - self.quit_last_seen > 0.10):
            self.quit_start = 0.0
            self.quit_last_seen = 0.0
        return False, None, None

    def update(self, landmarks) -> tuple[str, str | None, bool]:
        """Returns (status_text, slider_mode, should_quit)."""
        now = time.time()
        scale = self._hand_scale(landmarks)
        pinch_thresh = 0.28 * scale
        pinky_pinch_thresh = 0.34 * scale

        palm_x_raw = (landmarks[0].x + landmarks[9].x) / 2.0
        palm_y_raw = (landmarks[0].y + landmarks[9].y) / 2.0
        palm_x = self.palm_x_filter(palm_x_raw, now)

        index_pinched = self._dist(landmarks, 4, 8) < pinch_thresh
        middle_pinched = self._dist(landmarks, 4, 12) < pinch_thresh * 0.95
        ring_pinched = self._dist(landmarks, 4, 16) < pinch_thresh
        pinky_pinched = self._dist(landmarks, 4, 20) < pinky_pinch_thresh

        if self.state == "COOLDOWN":
            if now - self.last_action_time > 0.55:
                self.state = "ARMED"
                self.last_active_time = now
                self._reset_state_buffers()
            return "COOLDOWN (Staying Armed...)", None, False

        if self.state == "IDLE":
            self._release_window_switcher_if_needed()
            self._reset_state_buffers()
            should_quit, quit_msg, mode = self._check_quit_pinch(pinky_pinched, now)
            if should_quit or quit_msg:
                return quit_msg or "CLOSING...", mode, should_quit

            if self._is_open_palm(landmarks):
                if self.arm_start_time == 0.0:
                    self.arm_start_time = now
                elif now - self.arm_start_time >= 0.30:
                    self.state = "ARMED"
                    self.last_active_time = now
            else:
                self.arm_start_time = 0.0
            return "IDLE (Open palm 0.3s=Arm | Thumb+Pinky 1s=Quit)", None, False

        if self.state == "ARMED":
            open_palm = self._is_open_palm(landmarks)
            fist = self._is_fist(landmarks)

            # Thumb + Pinky pinch held for 1.0s -> Quit Air-Desk AI
            if pinky_pinched and not index_pinched and not middle_pinched and not ring_pinched:
                self.last_active_time = now
                should_quit, quit_msg, mode = self._check_quit_pinch(True, now)
                if should_quit or quit_msg:
                    return quit_msg or "CLOSING...", mode, should_quit
            else:
                _, _, _ = self._check_quit_pinch(False, now)

            # 1. Thumb + Ring Pinch -> Multi-Window Switcher (Hold Alt, step Tab every 0.55s, release Alt on unpinch)
            if ring_pinched and not index_pinched and not middle_pinched and not pinky_pinched:
                self.last_active_time = now
                self.win_release_seen_time = 0.0
                if not self.win_pinch_active:
                    self.win_pinch_active = True
                    self.win_last_step_time = now
                    ok = self.os_ctrl.begin_window_switcher()
                    return (
                        "WINDOW SWITCHER: Hold pinch to cycle, release to select"
                        if ok else "ERR: ydotool missing"
                    ), "WIN", False
                elif now - self.win_last_step_time >= self.WINDOW_STEP_INTERVAL:
                    self.win_last_step_time = now
                    self.os_ctrl.step_window_switcher()
                    return "WINDOW SWITCHER: Next window... (release to focus)", "WIN", False
                return "WINDOW SWITCHER ACTIVE (release pinch to focus)", "WIN", False
            elif self.win_pinch_active:
                if self.win_release_seen_time == 0.0:
                    self.win_release_seen_time = now
                elif now - self.win_release_seen_time >= 0.14:
                    self._release_window_switcher_if_needed()
                    self.state = "COOLDOWN"
                    self.last_action_time = now
                    self.last_active_time = now
                    return "ACTION: Focused Selected Window", None, False
                return "WINDOW SWITCHER: Release to focus...", "WIN", False

            # 2. Thumb + Index Pinch & Vertical Drag -> Volume Slider
            if index_pinched and not pinky_pinched:
                self.last_active_time = now
                self.bright_anchor_y = None
                self._reset_state_buffers(keep_sliders=True)
                pinch_y = (landmarks[4].y + landmarks[8].y) / 2.0
                if self.vol_anchor_y is None:
                    self.vol_anchor_y = pinch_y
                    return "VOLUME SLIDER: Move Up / Down", "VOL", False

                dy = self.vol_anchor_y - pinch_y
                if abs(dy) >= 0.015:
                    step = max(-10, min(10, int(round(dy * 220))))
                    self.os_ctrl.change_volume(step)
                    self.vol_anchor_y = pinch_y
                    return f"VOLUME: {'+' if step > 0 else ''}{step}%", "VOL", False
                return "VOLUME SLIDER ACTIVE", "VOL", False
            else:
                self.vol_anchor_y = None

            # 3. Thumb + Middle Pinch & Vertical Drag -> Brightness Slider
            if middle_pinched and not index_pinched and not pinky_pinched:
                self.last_active_time = now
                self.vol_anchor_y = None
                self._reset_state_buffers(keep_sliders=True)
                b_y = (landmarks[4].y + landmarks[12].y) / 2.0
                if self.bright_anchor_y is None:
                    self.bright_anchor_y = b_y
                    return "BRIGHTNESS SLIDER: Move Up / Down", "BRIGHT", False

                dy = self.bright_anchor_y - b_y
                if abs(dy) >= 0.018:
                    step = max(-10, min(10, int(round(dy * 200))))
                    self.os_ctrl.change_brightness(step)
                    self.bright_anchor_y = b_y
                    return f"BRIGHTNESS: {'+' if step > 0 else ''}{step}%", "BRIGHT", False
                return "BRIGHTNESS SLIDER ACTIVE", "BRIGHT", False
            else:
                self.bright_anchor_y = None

            # 4. Open Palm Sweep (R->L = Next Track | L->R = Prev Track)
            if open_palm and not middle_pinched and not pinky_pinched:
                self.palm_history.append((palm_x, now))
                sweep_dir = self._check_palm_sweep(now)
                if sweep_dir == "NEXT":
                    self.os_ctrl.next_track()
                    self._trigger_cooldown()
                    return "ACTION: Next Track (Palm Sweep R->L)", None, False
                elif sweep_dir == "PREV":
                    self.os_ctrl.prev_track()
                    self._trigger_cooldown()
                    return "ACTION: Previous Track (Palm Sweep L->R)", None, False
                if len(self.palm_history) >= 2:
                    self.last_active_time = now
            else:
                self.palm_history.clear()

            # 5. Fist Zones: Top-Left = Brave | Top-Right = Spotify | Elsewhere = Play/Pause
            if fist:
                self.last_active_time = now
                in_top_left = palm_x_raw < 0.28 and palm_y_raw < 0.35
                in_top_right = palm_x_raw > 0.72 and palm_y_raw < 0.35

                if self.fist_start == 0.0:
                    self.fist_start = now
                elif now - self.fist_start >= 0.22:
                    if in_top_left:
                        self.os_ctrl.open_brave()
                        self._trigger_cooldown()
                        return "ACTION: Launch Brave", None, False
                    elif in_top_right:
                        self.os_ctrl.open_spotify()
                        self._trigger_cooldown()
                        return "ACTION: Launch Spotify", None, False
                    else:
                        self.os_ctrl.play_pause()
                        self._trigger_cooldown()
                        return "ACTION: Play / Pause", None, False

                if in_top_left:
                    return "HOLDING FIST TOP-LEFT (Brave)...", None, False
                elif in_top_right:
                    return "HOLDING FIST TOP-RIGHT (Spotify)...", None, False
                return "HOLDING FIST (Play/Pause)...", None, False
            else:
                self.fist_start = 0.0

            remaining = self.ARMED_TIMEOUT - (now - self.last_active_time)
            if remaining <= 0:
                self.state = "IDLE"
                self.arm_start_time = 0.0
                return "IDLE (Timed out)", None, False

            return f"ARMED ({remaining:.1f}s) | Thumb+Ring=Win | Mid=Bright | Idx=Vol", None, False

        return "IDLE", None, False

    def _trigger_cooldown(self):
        self.state = "COOLDOWN"
        self.last_action_time = time.time()
        self.last_active_time = time.time()
        self._reset_state_buffers()


# --- 4. Main Webcam Loop with Persistent Topmost HUD ---
def main():
    base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    engine = GestureEngine()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win_name = "Air-Desk Fedora HUD"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 480)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    vol_pct, is_muted = engine.os_ctrl.get_volume_info()
    bright_pct = engine.os_ctrl.get_brightness_info()
    last_sys_poll = 0.0
    should_quit = False

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            now = time.time()
            if now - last_sys_poll > 0.20:
                vol_pct, is_muted = engine.os_ctrl.get_volume_info()
                bright_pct = engine.os_ctrl.get_brightness_info()
                last_sys_poll = now

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = detector.detect_for_video(mp_image, int(now * 1000))
            status = "No Hand Detected"
            slider_mode = None

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                status, slider_mode, should_quit = engine.update(lm)
                for pt in lm:
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 180), -1)

                if slider_mode == "VOL":
                    t_pt = (int(lm[4].x * w), int(lm[4].y * h))
                    i_pt = (int(lm[8].x * w), int(lm[8].y * h))
                    cv2.line(frame, t_pt, i_pt, (0, 220, 255), 3)
                elif slider_mode == "BRIGHT":
                    t_pt = (int(lm[4].x * w), int(lm[4].y * h))
                    m_pt = (int(lm[12].x * w), int(lm[12].y * h))
                    cv2.line(frame, t_pt, m_pt, (255, 180, 50), 3)
                elif slider_mode == "WIN":
                    t_pt = (int(lm[4].x * w), int(lm[4].y * h))
                    rg_pt = (int(lm[16].x * w), int(lm[16].y * h))
                    cv2.line(frame, t_pt, rg_pt, (220, 120, 255), 3)
                elif slider_mode == "QUIT":
                    t_pt = (int(lm[4].x * w), int(lm[4].y * h))
                    pk_pt = (int(lm[20].x * w), int(lm[20].y * h))
                    cv2.line(frame, t_pt, pk_pt, (80, 80, 255), 3)
            else:
                engine._release_window_switcher_if_needed()
                engine.quit_start = 0.0
                engine.quit_last_seen = 0.0

            # Corner launch guides (Top-Left = Brave, Top-Right = Spotify)
            cv2.rectangle(frame, (8, 52), (int(0.28 * w), int(0.35 * h)), (90, 140, 255), 1)
            cv2.putText(frame, "BRAVE", (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 140, 255), 1)

            cv2.rectangle(frame, (int(0.72 * w), 52), (w - 68, int(0.35 * h)), (90, 220, 120), 1)
            cv2.putText(frame, "SPOTIFY", (int(0.73 * w) + 4, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 220, 120), 1)

            active_highlight = any(
                k in status for k in ("ARMED", "VOLUME", "BRIGHTNESS", "ACTION", "COOLDOWN", "HOLDING", "WINDOW", "HOLD THUMB+PINKY")
            )
            color = (0, 255, 0) if active_highlight else (200, 200, 200)
            cv2.putText(frame, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

            # Right-side Volume Bar
            bar_x, bar_top, bar_bottom, bar_w = w - 42, 70, h - 50, 18
            bar_h = bar_bottom - bar_top
            cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bottom), (80, 80, 80), 2)
            fill_h = int(bar_h * (0 if is_muted else min(vol_pct, 100) / 100.0))
            fill_color = (0, 100, 255) if is_muted else (0, 230, 180)
            cv2.rectangle(frame, (bar_x + 2, bar_bottom - fill_h), (bar_x + bar_w - 2, bar_bottom - 2), fill_color, -1)
            vol_label = "MUTE" if is_muted else f"{vol_pct}%"
            cv2.putText(frame, vol_label, (bar_x - 14, bar_bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fill_color, 2)

            # Brightness Bar (when brightnessctl is available on laptop)
            if bright_pct is not None:
                b_x = w - 68
                cv2.rectangle(frame, (b_x, bar_top), (b_x + bar_w, bar_bottom), (80, 80, 80), 2)
                b_fill_h = int(bar_h * (min(bright_pct, 100) / 100.0))
                cv2.rectangle(frame, (b_x + 2, bar_bottom - b_fill_h), (b_x + bar_w - 2, bar_bottom - 2), (255, 180, 50), -1)
                cv2.putText(frame, f"{bright_pct}%", (b_x - 10, bar_bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 50), 2)

            cv2.imshow(win_name, frame)
            if should_quit or (cv2.waitKey(1) & 0xFF == 27):
                break
    finally:
        engine._release_window_switcher_if_needed()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()