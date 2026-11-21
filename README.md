# Air-Desk AI

Touchless webcam hand-gesture desktop controller for **Fedora Linux (GNOME / Wayland)**. Built with **MediaPipe HandLandmarker**, **OpenCV**, **One-Euro Filtering**, **PipeWire (`wpctl`)**, **`brightnessctl`**, **`playerctl`**, and **`ydotool`**, Air-Desk AI lets you adjust system volume and screen brightness, cycle through open windows, control media playback, launch apps, and quit the HUD hands-free.

---

## Features & Gesture Reference

Air-Desk AI uses a state-machine engine (`IDLE` -> `ARMED` -> `COOLDOWN`) with a `15.0s` active timeout and a `OneEuroFilter` to keep tracking smooth and jitter-free.

| Gesture / Pose | State | Action |
| :--- | :--- | :--- |
| **Open Palm** (hold `0.30s`) | `IDLE` | Arms Air-Desk AI (`15s` active window) |
| **Thumb + Index Pinch & Drag Up/Down** | `ARMED` | Adjusts PipeWire output volume (`wpctl`) |
| **Thumb + Middle Pinch & Drag Up/Down** | `ARMED` | Adjusts screen brightness (`brightnessctl`) |
| **Thumb + Ring Pinch (Hold & Release)** | `ARMED` | Holds `Alt`, steps `Tab` every `0.55s` to cycle open windows, releases `Alt` on unpinch |
| **Open Palm Sweep (`Right -> Left`)** | `ARMED` | Next track (`playerctl`) |
| **Open Palm Sweep (`Left -> Right`)** | `ARMED` | Previous track (`playerctl`) |
| **Fist in Top-Left Zone** (`x < 0.28`, `y < 0.35`, hold `0.22s`) | `ARMED` | Launches **Brave Browser** |
| **Fist in Top-Right Zone** (`x > 0.72`, `y < 0.35`, hold `0.22s`) | `ARMED` | Launches **Spotify** (native or Flatpak) |
| **Fist Elsewhere** (hold `0.22s`) | `ARMED` | Toggles Play / Pause (`playerctl`) |
| **Thumb + Pinky Pinch** (hold `1.0s`) | `IDLE` / `ARMED` | Cleanly closes **Air-Desk AI** (or press `ESC`) |

---

## Project Structure

```text
Air-Desk AI/
├── main.py                  # Core tracking, gesture engine, Fedora controller & HUD
├── models/
│   └── hand_landmarker.task # MediaPipe Hand Landmarker task bundle
└── README.md
```

---

## Complete Fedora Wayland Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Viper2911/Air-Desk-AI.git](https://github.com/Viper2911/Air-Desk-AI.git) "/home/$USER/Documents/Air-Desk AI"
cd "/home/$USER/Documents/Air-Desk AI"
```

### 2. Install System Dependencies
Install `ydotool`, `brightnessctl`, `playerctl`, and `curl` via `dnf`:
```bash
sudo dnf install -y ydotool brightnessctl playerctl curl
```

### 3. Configure `/dev/uinput` and `ydotoold` Socket Permissions
Allow your user to access `/dev/uinput` and connect to `/tmp/.ydotool_socket` on Wayland:
```bash
sudo usermod -aG input "$USER"

echo 'KERNEL=="uinput", MODE="0660", GROUP="input", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/80-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

sudo mkdir -p /etc/systemd/system/ydotool.service.d
cat << 'EOF' | sudo tee /etc/systemd/system/ydotool.service.d/override.conf
[Service]
ExecStart=
ExecStart=/usr/bin/ydotoold --socket-path=/tmp/.ydotool_socket --socket-perm=0666
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ydotool.service
```
*(Log out and back in if `input` was newly added to your groups.)*

### 4. Create Python Virtual Environment & Install Packages
```bash
cd "/home/$USER/Documents/Air-Desk AI"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe numpy
```

### 5. Download the MediaPipe Hand Landmarker Model
```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  [https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
```

---

## Running Air-Desk AI

### Run from Terminal
```bash
cd "/home/$USER/Documents/Air-Desk AI"
source .venv/bin/activate
python main.py
```

### Optional: Bind a GNOME Keyboard Shortcut
1. Open **Settings -> Keyboard -> View and Customize Shortcuts -> Custom Shortcuts**.
2. Click **+** and set:
   - **Name:** `Air-Desk AI`
   - **Command:**
     ```bash
     bash -c 'cd "/home/Viper777/Documents/Air-Desk AI" && exec "/home/Viper777/Documents/Air-Desk AI/.venv/bin/python" "/home/Viper777/Documents/Air-Desk AI/main.py"'
     ```
   - **Shortcut:** Choose your preferred key combination (e.g., `Super + G`).

---

## Troubleshooting

* **`Alt + Tab` window switching does not respond:** Check that `ydotool.service` is active and `/tmp/.ydotool_socket` exists:
  ```bash
  systemctl status ydotool.service
  ls -la /tmp/.ydotool_socket
  ```
* **Brightness slider does not move:** Ensure your user has write permission to `/sys/class/backlight/*` (or belongs to the `video` group: `sudo usermod -aG video "$USER"`).
* **Webcam does not open:** Verify `/dev/video0` is not locked by another application (`fuser /dev/video0`).

---

## Architecture Overview

* **`OneEuroFilter`**: Velocity-adaptive low-pass filter smoothing palm horizontal displacement while keeping fast sweeps responsive.
* **`FedoraController`**: Non-blocking subprocess wrapper around `wpctl`, `brightnessctl`, `playerctl`, `ydotool` (with `YDOTOOL_SOCKET` fallback across `/run/user/1000/.ydotool_socket` and `/tmp/.ydotool_socket`), and desktop app launchers.
* **`GestureEngine`**: Hand-scale-normalized gesture classifier with pinch thresholds, hold timers, flicker tolerance windows, and safe `Alt` key-release cleanup on exit.
* **Persistent HUD Loop**: Renders real-time hand landmarks, active pinch lines, corner app zones, live volume/brightness meters, and keeps the HUD window topmost while external windows like Spotify or Brave are open.
