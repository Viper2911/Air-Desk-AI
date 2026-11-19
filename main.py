import math
import subprocess
import time
from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --- 1. One-Euro Filter for Jitter-Free Tracking ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
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


# --- 2. Fedora Wayland & PipeWire OS Controller ---
class FedoraController:
    @staticmethod
    def run_cmd(cmd: list[str]):
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def play_pause(self):
        self.run_cmd(["playerctl", "--player=spotify,%any", "play-pause"])

    def next_track(self):
        self.run_cmd(["playerctl", "--player=spotify,%any", "next"])

    def prev_track(self):
        self.run_cmd(["playerctl", "--player=spotify,%any", "previous"])

    def change_volume(self, delta_percent: int):
        if delta_percent == 0:
            return
        sign = "+" if delta_percent > 0 else "-"
        self.run_cmd([
            "wpctl", "set-volume", "--limit", "1.0",
            "@DEFAULT_AUDIO_SINK@", f"{abs(delta_percent)}%{sign}"
        ])

    def toggle_mute(self):
        self.run_cmd(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])

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


# --- 3. Gesture Classifier & Persistent Armed State Machine ---
class GestureEngine:
    ARMED_TIMEOUT = 12.0
    SWEEP_MIN_DISTANCE = 0.22  # ~22% of frame width for a deliberate palm sweep
    SWEEP_MAX_WINDOW = 1.0     # Max seconds to complete the sweep

    def __init__(self):
        self.state = "IDLE"
        self.arm_start_time = 0.0
        self.fist_start = 0.0
        self.last_action_time = 0.0
        self.last_active_time = 0.0
        self.pinch_anchor_y = None
        self.palm_history = deque(maxlen=25)
        self.palm_x_filter = OneEuroFilter(min_cutoff=1.4, beta=0.12)
        self.os_ctrl = FedoraController()

    def _dist(self, lm, a: int, b: int) -> float:
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

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
        """
        Returns 'NEXT' for right-to-left palm sweep (dx <= -SWEEP_MIN_DISTANCE),
        'PREV' for left-to-right palm sweep (dx >= +SWEEP_MIN_DISTANCE),
        or None if not swept far enough yet.
        """
        while len(self.palm_history) > 1 and (now - self.palm_history[0][1] > self.SWEEP_MAX_WINDOW):
            self.palm_history.popleft()

        if len(self.palm_history) < 4:
            return None

        start_x = self.palm_history[0][0]
        curr_x = self.palm_history[-1][0]
        dx = curr_x - start_x

        if dx <= -self.SWEEP_MIN_DISTANCE:
            return "NEXT"
        elif dx >= self.SWEEP_MIN_DISTANCE:
            return "PREV"
        return None

    def _reset_state_buffers(self, keep_pinch=False):
        self.fist_start = 0.0
        self.palm_history.clear()
        if not keep_pinch:
            self.pinch_anchor_y = None

    def update(self, landmarks) -> tuple[str, bool]:
        now = time.time()
        pinch_y = (landmarks[4].y + landmarks[8].y) / 2.0
        # Palm center approximation using wrist (0) and middle finger MCP (9)
        raw_palm_x = (landmarks[0].x + landmarks[9].x) / 2.0
        palm_x = self.palm_x_filter(raw_palm_x, now)

        if self.state == "COOLDOWN":
            if now - self.last_action_time > 0.75:
                self.state = "ARMED"
                self.last_active_time = now
                self._reset_state_buffers()
            return "COOLDOWN (Staying Armed...)", False

        if self.state == "IDLE":
            self._reset_state_buffers()
            if self._is_open_palm(landmarks):
                if self.arm_start_time == 0.0:
                    self.arm_start_time = now
                elif now - self.arm_start_time >= 0.30:
                    self.state = "ARMED"
                    self.last_active_time = now
            else:
                self.arm_start_time = 0.0
            return "IDLE (Hold open palm 0.3s to arm)", False

        if self.state == "ARMED":
            index_pinched = self._dist(landmarks, 4, 8) < 0.045
            middle_pinched = self._dist(landmarks, 4, 12) < 0.042
            open_palm = self._is_open_palm(landmarks)
            fist = self._is_fist(landmarks)

            # 1. Continuous Pinch + Drag Volume Slider
            if index_pinched:
                self.last_active_time = now
                self._reset_state_buffers(keep_pinch=True)
                if self.pinch_anchor_y is None:
                    self.pinch_anchor_y = pinch_y
                    return "VOLUME SLIDER: Move hand Up / Down", True

                dy = self.pinch_anchor_y - pinch_y
                if abs(dy) >= 0.015:
                    step = int(round(dy * 220))
                    step = max(-10, min(10, step))
                    self.os_ctrl.change_volume(step)
                    self.pinch_anchor_y = pinch_y
                    direction = f"+{step}%" if step > 0 else f"{step}%"
                    return f"VOLUME: {direction}", True
                return "VOLUME SLIDER ACTIVE", True
            else:
                self.pinch_anchor_y = None

            # 2. Open Palm Sweep (Right->Left = Next Track | Left->Right = Prev Track)
            if open_palm and not middle_pinched:
                self.palm_history.append((palm_x, now))
                sweep_dir = self._check_palm_sweep(now)
                if sweep_dir == "NEXT":
                    self.os_ctrl.next_track()
                    self._trigger_cooldown()
                    return "ACTION: Next Track (Palm Sweep R->L)", False
                elif sweep_dir == "PREV":
                    self.os_ctrl.prev_track()
                    self._trigger_cooldown()
                    return "ACTION: Previous Track (Palm Sweep L->R)", False
                if len(self.palm_history) >= 2:
                    self.last_active_time = now
            else:
                self.palm_history.clear()

            # 3. Middle-Finger Pinch -> Toggle Mute
            if middle_pinched:
                self.os_ctrl.toggle_mute()
                self._trigger_cooldown()
                return "ACTION: Toggle Mute", False

            # 4. Closed Fist (0.18s) -> Play / Pause
            if fist:
                self.last_active_time = now
                if self.fist_start == 0.0:
                    self.fist_start = now
                elif now - self.fist_start >= 0.18:
                    self.os_ctrl.play_pause()
                    self._trigger_cooldown()
                    return "ACTION: Play / Pause", False
                return "HOLDING FIST (Play/Pause)...", False
            else:
                self.fist_start = 0.0

            remaining = self.ARMED_TIMEOUT - (now - self.last_active_time)
            if remaining <= 0:
                self.state = "IDLE"
                self.arm_start_time = 0.0
                return "IDLE (Timed out)", False

            return f"ARMED ({remaining:.1f}s) | Sweep R->L=Next | L->R=Prev | Pinch=Vol", False

        return "IDLE", False

    def _trigger_cooldown(self):
        self.state = "COOLDOWN"
        self.last_action_time = time.time()
        self.last_active_time = time.time()
        self._reset_state_buffers()


# --- 4. Main Webcam Loop with Live Volume HUD ---
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

    vol_pct, is_muted = engine.os_ctrl.get_volume_info()
    last_vol_poll = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_vol_poll > 0.15:
            vol_pct, is_muted = engine.os_ctrl.get_volume_info()
            last_vol_poll = now

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, int(now * 1000))
        status = "No Hand Detected"
        sliding_vol = False

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            status, sliding_vol = engine.update(lm)
            for pt in lm:
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 180), -1)

            if sliding_vol:
                t_pt = (int(lm[4].x * w), int(lm[4].y * h))
                i_pt = (int(lm[8].x * w), int(lm[8].y * h))
                cv2.line(frame, t_pt, i_pt, (0, 220, 255), 3)

        active_highlight = any(
            k in status for k in ("ARMED", "VOLUME", "ACTION", "COOLDOWN", "HOLDING")
        )
        color = (0, 255, 0) if active_highlight else (200, 200, 200)
        cv2.putText(frame, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        bar_x, bar_top, bar_bottom, bar_w = w - 45, 70, h - 50, 20
        bar_h = bar_bottom - bar_top
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bottom), (80, 80, 80), 2)
        fill_h = int(bar_h * (0 if is_muted else min(vol_pct, 100) / 100.0))
        fill_color = (0, 100, 255) if is_muted else (0, 230, 180)
        cv2.rectangle(frame, (bar_x + 2, bar_bottom - fill_h), (bar_x + bar_w - 2, bar_bottom - 2), fill_color, -1)
        vol_label = "MUTE" if is_muted else f"{vol_pct}%"
        cv2.putText(frame, vol_label, (bar_x - 12, bar_bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fill_color, 2)

        cv2.imshow("Air-Desk Fedora HUD", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()