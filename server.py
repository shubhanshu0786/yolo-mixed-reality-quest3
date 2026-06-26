"""
WebSocket server for real-time YOLO detection.

TWO MODES (set in config.yaml → server.source):

  "unity"  (default) — Unity sends JPEG frames, server returns detections.
  "webcam"           — Server captures from a local webcam, pushes detections
                       to all connected Unity clients automatically. Unity does
                       NOT send frames in this mode; it just receives results.

Webcam mode is useful when the headset's passthrough camera is inaccessible
(e.g. Meta Quest 3 with OpenXR), so YOLO runs on the PC's own webcam instead.

Response JSON schema (same in both modes):
{
  "frame_id": int,
  "detections": [ { "class_id", "class_name", "confidence", "bbox",
                     "center", "size" }, ... ],
  "inference_time_ms": float,
  "frame_size": {"width": int, "height": int},
  "annotated_frame": str | null
}
"""

import asyncio
import base64
import json
import logging
import threading
import time

import cv2
import numpy as np
import websockets
import yaml

from detector import YOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Shared state for webcam mode
_connected_clients: set = set()
_clients_lock = threading.Lock()


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_detector(cfg: dict) -> YOLODetector:
    m = cfg["model"]
    return YOLODetector(
        model_path=m["path"],
        confidence=m["confidence"],
        iou_threshold=m["iou_threshold"],
        device=m["device"],
        filter_classes=m.get("filter_classes"),
        max_detections=m.get("max_detections", 100),
    )


# ── UNITY MODE ────────────────────────────────────────────────────────────────

async def handle_unity_client(websocket, detector: YOLODetector, cfg: dict):
    """Unity sends JPEG frames → server detects → server responds."""
    client_addr = websocket.remote_address
    log.info("Client connected: %s", client_addr)
    frame_id = 0
    return_annotated = cfg["server"].get("return_annotated_frame", False)
    jpeg_quality = cfg["server"].get("annotated_frame_quality", 60)

    try:
        async for message in websocket:
            frame_id += 1

            if not isinstance(message, bytes):
                await websocket.send(json.dumps({"error": "Expected binary JPEG", "frame_id": frame_id}))
                continue

            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send(json.dumps({"error": "Failed to decode image", "frame_id": frame_id}))
                continue

            result = detector.detect(frame)
            result["frame_id"] = frame_id

            if return_annotated:
                annotated = detector.annotate(frame, result)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                result["annotated_frame"] = base64.b64encode(buf).decode("utf-8")
                if frame_id % 50 == 0:
                    cv2.imwrite(f"debug_frame_{frame_id}.jpg", annotated)
                    log.info("Saved debug_frame_%d.jpg", frame_id)
            else:
                result["annotated_frame"] = None

            await websocket.send(json.dumps(result))

            if frame_id % 100 == 0:
                log.info("[%s] frame %d | %d detections | %.1f ms",
                         client_addr, frame_id,
                         len(result["detections"]),
                         result["inference_time_ms"])

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning("Connection closed with error from %s: %s", client_addr, e)
    finally:
        log.info("Client disconnected: %s (served %d frames)", client_addr, frame_id)


# ── WEBCAM MODE ───────────────────────────────────────────────────────────────

async def handle_webcam_client(websocket):
    """Unity connects, registers itself, waits for pushed detections."""
    client_addr = websocket.remote_address
    log.info("Client connected (webcam mode): %s", client_addr)
    with _clients_lock:
        _connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        with _clients_lock:
            _connected_clients.discard(websocket)
        log.info("Client disconnected: %s", client_addr)


def webcam_capture_loop(detector: YOLODetector, cfg: dict, loop: asyncio.AbstractEventLoop):
    """Runs in a background thread: captures webcam, runs YOLO, pushes JSON to all clients."""
    webcam_index  = cfg["server"].get("webcam_index", 0)
    target_fps    = cfg["server"].get("webcam_fps", 10)
    return_annotated = cfg["server"].get("return_annotated_frame", False)
    jpeg_quality  = cfg["server"].get("annotated_frame_quality", 60)
    cap_w = cfg["server"].get("webcam_width",  640)
    cap_h = cfg["server"].get("webcam_height", 480)
    interval = 1.0 / target_fps

    cap = cv2.VideoCapture(webcam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)

    if not cap.isOpened():
        log.error("Could not open webcam index %d. Check webcam_index in config.yaml.", webcam_index)
        return

    log.info("Webcam %d opened at %dx%d, target %d fps", webcam_index, cap_w, cap_h, target_fps)

    frame_id = 0
    while True:
        t0 = time.monotonic()
        ok, frame = cap.read()
        if not ok:
            log.warning("Webcam read failed — retrying...")
            time.sleep(0.5)
            continue

        frame_id += 1
        result = detector.detect(frame)
        result["frame_id"] = frame_id

        if return_annotated:
            annotated = detector.annotate(frame, result)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            result["annotated_frame"] = base64.b64encode(buf).decode("utf-8")
            if frame_id % 50 == 0:
                cv2.imwrite(f"debug_webcam_{frame_id}.jpg", annotated)
                log.info("Saved debug_webcam_%d.jpg", frame_id)
        else:
            result["annotated_frame"] = None

        if frame_id % 30 == 0:
            log.info("webcam frame %d | %d detections | %.1f ms",
                     frame_id, len(result["detections"]), result["inference_time_ms"])

        payload = json.dumps(result)
        with _clients_lock:
            clients = list(_connected_clients)

        if clients:
            async def push(ws, msg):
                try:
                    await ws.send(msg)
                except Exception:
                    pass

            for ws in clients:
                asyncio.run_coroutine_threadsafe(push(ws, payload), loop)

        elapsed = time.monotonic() - t0
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    cap.release()


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main():
    cfg = load_config()
    detector = build_detector(cfg)

    host   = cfg["server"]["host"]
    port   = cfg["server"]["port"]
    source = cfg["server"].get("source", "unity")

    log.info("Starting YOLO WebSocket server on ws://%s:%d", host, port)
    log.info("Model: %s | Confidence: %s | Device: %s | Source: %s",
             cfg["model"]["path"], cfg["model"]["confidence"],
             cfg["model"]["device"], source)

    if source == "webcam":
        loop = asyncio.get_event_loop()
        t = threading.Thread(target=webcam_capture_loop, args=(detector, cfg, loop), daemon=True)
        t.start()
        handler = handle_webcam_client
    else:
        handler = lambda ws: handle_unity_client(ws, detector, cfg)

    async with websockets.serve(handler, host, port, max_size=10 * 1024 * 1024):
        log.info("Server ready. Waiting for Unity client...")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
