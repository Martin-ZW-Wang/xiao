import os
import time
import threading
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import requests
import torch

from flask import Flask, Response, jsonify
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

PI_STREAM_URL = (
    "http://raspberrypi.local:5001/raw_feed"
)

MODEL_PATH = "models/best.pt"

IMG_SIZE = 640

CONFIDENCE = 0.30
IOU_THRESHOLD = 0.45

WEB_STREAM_FPS = 20
JPEG_QUALITY = 75

HOST = "0.0.0.0"
PORT = 5002


CLASS_NAMES = {
    0: "vehicle",
    1: "motorcycle",
    2: "bicycle",
    3: "pedestrian",
}


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0
    USE_HALF = True

    torch.backends.cudnn.benchmark = True

else:

    DEVICE = "cpu"
    USE_HALF = False


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


@app.after_request
def cors(response):

    response.headers[
        "Access-Control-Allow-Origin"
    ] = "*"

    return response


# ============================================================
# Shared data
# ============================================================

frame_lock = threading.Lock()
detection_lock = threading.Lock()
status_lock = threading.Lock()

latest_frame = None
latest_jpeg = None
latest_detections = []

running = True


status_data = {
    "vehicle": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "pedestrian": 0,

    "fps": 0.0,
    "inference_ms": 0.0,
    "source_fps": 0.0,

    "source_online": False,
    "model_ready": False,

    "device": str(DEVICE),

    "error": None,
}


event_log = deque(
    maxlen=50
)

last_event_time = {
    "vehicle": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "pedestrian": 0,
}

EVENT_COOLDOWN = 2.0


# ============================================================
# Raspberry Pi MJPEG reader
# ============================================================

def camera_reader():

    global latest_frame

    while running:

        try:

            print(
                "Connecting to Raspberry Pi Camera:"
            )

            print(
                PI_STREAM_URL
            )

            response = requests.get(
                PI_STREAM_URL,
                stream=True,
                timeout=(5, 30),
            )

            response.raise_for_status()

            buffer = b""

            previous_time = (
                time.perf_counter()
            )

            print(
                "Raspberry Pi camera connected."
            )

            for chunk in response.iter_content(
                chunk_size=4096
            ):

                if not running:
                    break

                if not chunk:
                    continue

                buffer += chunk

                while True:

                    start = buffer.find(
                        b"\xff\xd8"
                    )

                    end = buffer.find(
                        b"\xff\xd9",
                        start + 2
                    )

                    if (
                        start == -1
                        or end == -1
                    ):
                        break

                    jpg = buffer[
                        start:end + 2
                    ]

                    buffer = buffer[
                        end + 2:
                    ]

                    array = np.frombuffer(
                        jpg,
                        dtype=np.uint8
                    )

                    frame = cv2.imdecode(
                        array,
                        cv2.IMREAD_COLOR
                    )

                    if frame is None:
                        continue

                    now = time.perf_counter()

                    elapsed = (
                        now - previous_time
                    )

                    previous_time = now

                    if elapsed > 0:

                        source_fps = (
                            1.0 / elapsed
                        )

                    else:

                        source_fps = 0.0

                    with frame_lock:
                        latest_frame = (
                            frame.copy()
                        )

                    with status_lock:

                        status_data[
                            "source_online"
                        ] = True

                        status_data[
                            "source_fps"
                        ] = round(
                            source_fps,
                            2
                        )

                        status_data[
                            "error"
                        ] = None

        except Exception as e:

            print(
                "Camera connection error:",
                e
            )

            with status_lock:

                status_data[
                    "source_online"
                ] = False

                status_data[
                    "error"
                ] = str(e)

            time.sleep(1)


# ============================================================
# Drawing
# ============================================================

def draw_detections(frame):

    with detection_lock:

        detections = list(
            latest_detections
        )

    for det in detections:

        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]

        name = det["class"]
        conf = det["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{name} {conf:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(20, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    return frame


# ============================================================
# Web stream renderer
# ============================================================

def renderer():

    global latest_jpeg

    interval = (
        1.0 / WEB_STREAM_FPS
    )

    while running:

        start = time.perf_counter()

        with frame_lock:

            if latest_frame is None:
                frame = None

            else:
                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.05)
            continue

        frame = draw_detections(
            frame
        )

        with status_lock:

            yolo_fps = (
                status_data["fps"]
            )

            source_fps = (
                status_data["source_fps"]
            )

        cv2.putText(
            frame,
            f"CAMERA: {source_fps:.1f} FPS",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"YOLO: {yolo_fps:.1f} FPS",
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                JPEG_QUALITY,
            ],
        )

        if ok:

            latest_jpeg = (
                buffer.tobytes()
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        sleep_time = (
            interval - elapsed
        )

        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================
# Event log
# ============================================================

def update_events(detections):

    now = time.time()

    grouped = {
        "vehicle": [],
        "motorcycle": [],
        "bicycle": [],
        "pedestrian": [],
    }

    for det in detections:

        name = det["class"]

        if name in grouped:
            grouped[name].append(det)

    for name, items in grouped.items():

        if not items:
            continue

        if (
            now
            - last_event_time[name]
            < EVENT_COOLDOWN
        ):
            continue

        confidence = max(
            item["confidence"]
            for item in items
        )

        event_log.appendleft({
            "time": datetime.now().strftime(
                "%H:%M:%S"
            ),

            "class": name,

            "count": len(items),

            "confidence": round(
                confidence,
                2
            ),
        })

        last_event_time[name] = now


# ============================================================
# Flask API
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "service": "Windows YOLO Server",
        "status": "running",
    })


@app.route("/status")
def status():

    with status_lock:

        data = status_data.copy()

    data["events"] = list(
        event_log
    )

    return jsonify(data)


@app.route("/events")
def events():

    return jsonify(
        list(event_log)
    )


@app.route("/health")
def health():

    with status_lock:

        ok = (
            status_data["source_online"]
            and
            status_data["model_ready"]
        )

    return jsonify({
        "ok": ok
    })


def video_generator():

    while running:

        jpeg = latest_jpeg

        if jpeg is None:

            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
            + b"\r\n"
        )

        time.sleep(
            1.0 / WEB_STREAM_FPS
        )


@app.route("/video_feed")
def video_feed():

    return Response(
        video_generator(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


def flask_server():

    app.run(
        host=HOST,
        port=PORT,
        threaded=True,
        use_reloader=False,
    )


# ============================================================
# YOLO
# ============================================================

def main():

    global latest_detections

    print()
    print("==============================")
    print("Windows YOLO Server")
    print("==============================")

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Loading model:",
        MODEL_PATH
    )

    model = YOLO(
        MODEL_PATH
    )

    print(
        "Classes:",
        model.names
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    dummy = np.zeros(
        (
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        dtype=np.uint8,
    )

    print("YOLO warm-up...")

    model.predict(
        source=dummy,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False,
    )

    print("YOLO ready.")

    with status_lock:
        status_data[
            "model_ready"
        ] = True

    # --------------------------------------------------------
    # Camera reader
    # --------------------------------------------------------

    threading.Thread(
        target=camera_reader,
        daemon=True,
    ).start()

    # --------------------------------------------------------
    # Renderer
    # --------------------------------------------------------

    threading.Thread(
        target=renderer,
        daemon=True,
    ).start()

    # --------------------------------------------------------
    # Flask
    # --------------------------------------------------------

    threading.Thread(
        target=flask_server,
        daemon=True,
    ).start()

    print(
        f"YOLO API: "
        f"http://0.0.0.0:{PORT}"
    )

    print(
        "Waiting for Raspberry Pi Camera..."
    )

    while running:

        with frame_lock:

            if latest_frame is None:
                frame = None

            else:
                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.05)
            continue

        start = time.perf_counter()

        results = model.predict(
            source=frame,

            imgsz=IMG_SIZE,

            conf=CONFIDENCE,

            iou=IOU_THRESHOLD,

            device=DEVICE,

    
            max_det=100,

            verbose=False,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        inference_ms = (
            elapsed * 1000
        )

        if elapsed > 0:

            inference_fps = (
                1.0 / elapsed
            )

        else:

            inference_fps = 0.0

        result = results[0]

        counts = {
            "vehicle": 0,
            "motorcycle": 0,
            "bicycle": 0,
            "pedestrian": 0,
        }

        detections = []

        height, width = (
            frame.shape[:2]
        )

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                if (
                    class_id
                    not in CLASS_NAMES
                ):
                    continue

                name = (
                    CLASS_NAMES[
                        class_id
                    ]
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                x1 = max(
                    0,
                    min(
                        int(x1),
                        width - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        int(x2),
                        width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        int(y1),
                        height - 1
                    )
                )

                y2 = max(
                    0,
                    min(
                        int(y2),
                        height - 1
                    )
                )

                counts[name] += 1

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "class": name,

                    "confidence":
                        confidence,
                })

        with detection_lock:

            latest_detections = (
                detections
            )

        update_events(
            detections
        )

        with status_lock:

            status_data.update({
                "vehicle":
                    counts["vehicle"],

                "motorcycle":
                    counts["motorcycle"],

                "bicycle":
                    counts["bicycle"],

                "pedestrian":
                    counts["pedestrian"],

                "fps":
                    round(
                        inference_fps,
                        2
                    ),

                "inference_ms":
                    round(
                        inference_ms,
                        1
                    ),

                "error":
                    None,
            })


if __name__ == "__main__":

    main()
