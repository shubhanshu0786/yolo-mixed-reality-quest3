# YOLO Mixed Reality — Real-Time Object Detection on Meta Quest 3

Real-time AI object detection running inside a Meta Quest 3 headset.
Detected objects are labeled with **floating 3D text in world space** via Unity passthrough camera.

---

## Demo

> *Wear the Quest 3, look around your room — every object gets a live AI label floating on it.*

![Architecture Overview](https://raw.githubusercontent.com/placeholder/placeholder/main/docs/demo.gif)

---

## How It Works

```
Quest 3 passthrough camera (1280×960)
        │
        ▼  downscale to 640×480
Unity (C#) — YOLODetectionClient
        │
        ▼  JPEG frames via WebSocket
Python Server — server.py (port 8765)
        │
        ▼  YOLOv11n inference
detections JSON { class_name, confidence, bbox, center }
        │
        ▼  back to Quest via WebSocket
MRDetectionVisualizer — places 3D labels in world space
```

1. Quest 3 passthrough camera captures real-world frames
2. Unity downscales and streams frames as JPEG via WebSocket to a PC Python server
3. Python runs **YOLOv11n** inference on each frame (~10–30ms on GPU)
4. Detection results are sent back to Unity
5. Unity places **billboarded 3D text labels** in world space at each detected object

---

## Tech Stack

| Layer | Technology |
|---|---|
| Headset | Meta Quest 3, Horizon OS |
| XR Framework | OpenXR, Meta XR SDK, OVR SDK |
| Game Engine | Unity 2022.3 LTS (C#) |
| AI Model | YOLOv11n (Ultralytics) |
| Server | Python 3.10+, asyncio, websockets |
| Communication | WebSocket (NativeWebSocket package) |

---

## Project Structure

```
yolo-mixed-reality-quest3/
├── server.py              # WebSocket server — receives frames, runs YOLO, returns detections
├── detector.py            # YOLODetector wrapper around Ultralytics
├── config.yaml            # All tuneable settings (model, confidence, server, webcam)
├── requirements.txt       # Python dependencies
├── unity-scripts/
│   ├── YOLODetectionClient.cs    # Captures passthrough frames, sends via WebSocket
│   ├── MRDetectionVisualizer.cs  # Spawns 3D world-space labels from detection results
│   └── DetectionVisualizer.cs    # Minimal example subscriber (for reference)
└── .gitignore
```

---

## Requirements

- Meta Quest 3
- PC on the **same WiFi network** as the Quest
- Python 3.10+
- Unity 2022.3 LTS with Meta XR SDK
- GPU recommended for inference (CPU works but is slower)

---

## Quick Start

### 1. Python Server (PC)

```bash
# Clone repo
git clone https://github.com/shubhanshu0786/yolo-mixed-reality-quest3.git
cd yolo-mixed-reality-quest3

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (auto-downloads on first run, or manual):
# https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# Start server
python server.py
```

The server listens on `ws://0.0.0.0:8765` by default.

### 2. Unity (Quest 3)

1. Open your Unity 2022.3 project
2. Copy the scripts from `unity-scripts/` into your `Assets` folder
3. Select the **YOLO Manager** GameObject → set `Server Url` to your PC's local IP:
   ```
   ws://192.168.X.X:8765
   ```
   *(find your PC IP by running `ipconfig` in a terminal)*
4. Assign a `PassthroughCameraAccess` component and set `Capture Mode` to **Passthrough**
5. Build → Android → install APK on Quest 3

### 3. Grant Camera Permission (one-time)

The headset camera permission is not surfaced as an OS dialog on first launch.  
Grant it manually via ADB after installing the APK:

```bash
adb shell pm grant <your.package.name> horizonos.permission.HEADSET_CAMERA
```

ADB is included in the Android SDK bundled with Unity, or install it via [Android Studio](https://developer.android.com/studio).

---

## Configuration (`config.yaml`)

```yaml
model:
  path: "yolo11n.pt"        # swap to yolo11s/m/l for more accuracy
  confidence: 0.3           # detection threshold (lower = more detections)
  device: "cpu"             # "cpu" or "0" for GPU

server:
  host: "0.0.0.0"
  port: 8765
  source: "unity"           # "unity" (Quest sends frames) or "webcam" (PC webcam mode)
  return_annotated_frame: true
```

### Server Modes

| Mode | Description |
|---|---|
| `unity` | Quest 3 passthrough camera sends JPEG frames to the server |
| `webcam` | Server captures from PC webcam and pushes detections to Quest |

---

## Key Technical Challenge

`horizonos.permission.HEADSET_CAMERA` is **never surfaced as an OS dialog** on Horizon OS.

The fix is to grant it directly via ADB:
```bash
adb shell pm grant com.ari.AR horizonos.permission.HEADSET_CAMERA
```

This took hours to diagnose. Adding it here so nobody else loses that time.

---

## Possible Improvements

- [ ] Add **Scene Understanding raycast** so labels land on actual surfaces instead of at a fixed depth
- [ ] Upgrade model: `yolo11n` → `yolo11s` or `yolo11m` for better accuracy
- [ ] Lower `defaultDepth` from 2m → 1.5m for labels that feel closer to objects
- [ ] Add **label persistence** — smooth/lerp label positions instead of destroying each frame
- [ ] Support **multiple YOLO classes filtered by config** (e.g. only show `person`, `chair`, `laptop`)

---

## License

MIT

---

*Built with Meta XR SDK + Ultralytics YOLO + Unity 2022.3*
