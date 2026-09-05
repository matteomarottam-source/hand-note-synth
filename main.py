import argparse
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

NOTE_FREQS = {
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "F": 349.23,
    "G": 392.00,
    "A": 440.00,
    "B": 493.88,
}

GESTURE_TO_NOTE = {
    frozenset(["thumb"]): "C",
    frozenset(["index"]): "D",
    frozenset(["middle"]): "E",
    frozenset(["ring"]): "F",
    frozenset(["pinky"]): "G",
    frozenset(["thumb", "index"]): "A",
    frozenset(["thumb", "middle"]): "B",
}

SAMPLE_RATE = 44100
TONE_DURATION = 0.4

TIPS_PIPS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def play_tone(freq):
    t = np.linspace(0, TONE_DURATION, int(SAMPLE_RATE * TONE_DURATION), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)

    fade_samples = int(SAMPLE_RATE * 0.02)
    envelope = np.ones_like(wave)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    wave *= envelope

    sd.play(wave, SAMPLE_RATE)


def distance(landmarks, a, b):
    return ((landmarks[a].x - landmarks[b].x) ** 2 + (landmarks[a].y - landmarks[b].y) ** 2) ** 0.5


def extended_fingers(landmarks):
    extended = []
    for name, (tip, pip) in TIPS_PIPS.items():
        if landmarks[tip].y < landmarks[pip].y:
            extended.append(name)

    # thumb: orientation-independent check based on spread from pinky base,
    # avoids depending on mediapipe's left/right handedness label
    if distance(landmarks, 4, 17) > distance(landmarks, 2, 17):
        extended.append("thumb")

    return extended


def note_from_gesture(extended):
    return GESTURE_TO_NOTE.get(frozenset(extended))


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Scarico il modello hand_landmarker.task (solo la prima volta)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)


MAX_WIDTH = 640


class CameraStream:
    """Reads frames in a background thread and always exposes the latest one.

    A plain cv2.VideoCapture().read() call processes frames in the order the
    stream delivers them, so if processing (MediaPipe) is slower than the
    incoming framerate - very common with a phone streamed over WiFi - the
    displayed frame falls further and further behind real time. Grabbing
    continuously in the background and dropping stale frames keeps the lag
    from accumulating.
    """

    def __init__(self, source):
        # DirectShow is more reliable than the default Media Foundation backend
        # for third-party virtual cameras on Windows (e.g. Iriun Webcam).
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def isOpened(self):
        return self.cap.isOpened()

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ok, None
            return self.ok, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Hand gesture note synth")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index (e.g. 0, 1) or stream URL (e.g. for an IP-camera app). Default: 0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera

    ensure_model()

    cap = CameraStream(source)
    if not cap.isOpened():
        print(f"Impossibile aprire la camera '{source}'. Prova un altro indice con --camera.")
        return
    cap.start()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO,
    )

    last_note = None
    start_time = time.time()
    last_timestamp_ms = -1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if frame.shape[1] > MAX_WIDTH:
                scale = MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = max(int((time.time() - start_time) * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_note = None

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks)
                extended = extended_fingers(landmarks)
                current_note = note_from_gesture(extended)

            if current_note is not None and current_note != last_note:
                play_tone(NOTE_FREQS[current_note])
            last_note = current_note

            cv2.putText(
                frame,
                f"Nota: {current_note or '-'}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
            cv2.imshow("Hand Note Synth", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
