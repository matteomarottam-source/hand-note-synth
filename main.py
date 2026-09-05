import argparse
import os
import threading
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

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

GESTURE_LABELS = {
    frozenset(["thumb"]): "Pollice",
    frozenset(["index"]): "Indice",
    frozenset(["middle"]): "Medio",
    frozenset(["ring"]): "Anulare",
    frozenset(["pinky"]): "Mignolo",
    frozenset(["thumb", "index"]): "Pollice + Indice",
    frozenset(["thumb", "middle"]): "Pollice + Medio",
}

SAMPLE_RATE = 44100
TONE_DURATION = 1.6

# relative amplitude of each harmonic; the rapid roll-off plus the percussive
# envelope in play_note is what makes an additive tone read as "piano" rather
# than as a plain synth beep
PIANO_HARMONICS = [
    (1, 1.00),
    (2, 0.55),
    (3, 0.32),
    (4, 0.18),
    (5, 0.10),
    (6, 0.06),
    (7, 0.03),
    (8, 0.02),
]

TIPS_PIPS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}

MAX_WIDTH = 640

# a gesture must dominate this many of the last frames before it triggers a
# note, which rejects the single-frame misreads that a laggy camera produces
HISTORY_LEN = 7
STABILITY_THRESHOLD = 4

TEXT_COLOR = (250, 240, 230)
ACCENT_COLOR = (255, 178, 74)
LANDMARK_COLOR = (232, 150, 60)
PANEL_COLOR = (18, 18, 24)

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_NOTE = load_font(38)
FONT_LEGEND = load_font(17)


def play_note(freq):
    n_samples = int(SAMPLE_RATE * TONE_DURATION)
    t = np.linspace(0, TONE_DURATION, n_samples, endpoint=False)

    wave = np.zeros(n_samples)
    for harmonic, amplitude in PIANO_HARMONICS:
        wave += amplitude * np.sin(2 * np.pi * freq * harmonic * t)

    envelope = np.exp(-3.0 * t)
    attack_samples = int(SAMPLE_RATE * 0.006)
    envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)
    wave *= envelope

    wave *= 0.35 / np.max(np.abs(wave))
    sd.play(wave, SAMPLE_RATE)


def distance(landmarks, a, b):
    return ((landmarks[a].x - landmarks[b].x) ** 2 + (landmarks[a].y - landmarks[b].y) ** 2) ** 0.5


def extended_fingers(landmarks):
    extended = []

    # hand size is the reference for every threshold below, so detection does
    # not change with how close the hand is to the camera
    hand_size = distance(landmarks, 0, 9)
    if hand_size == 0:
        return extended

    for name, (tip, pip) in TIPS_PIPS.items():
        if (landmarks[pip].y - landmarks[tip].y) / hand_size > 0.15:
            extended.append(name)

    # thumb: orientation-independent check based on spread from pinky base,
    # avoids depending on mediapipe's left/right handedness label
    thumb_spread = (distance(landmarks, 4, 17) - distance(landmarks, 2, 17)) / hand_size
    if thumb_spread > 0.20:
        extended.append("thumb")

    return extended


def note_from_gesture(extended):
    return GESTURE_TO_NOTE.get(frozenset(extended))


def stable_note(history):
    note, count = Counter(history).most_common(1)[0]
    return note if count >= STABILITY_THRESHOLD else None


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Scarico il modello hand_landmarker.task (solo la prima volta)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, LANDMARK_COLOR[::-1], -1)


def legend_lines():
    return [f"{note}   {GESTURE_LABELS[combo]}" for combo, note in GESTURE_TO_NOTE.items()]


def draw_overlay(frame, note):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")

    draw.rounded_rectangle((14, 14, 250, 78), radius=10, fill=PANEL_COLOR + (190,))
    draw.text((30, 26), f"Nota  {note or '—'}", font=FONT_NOTE, fill=ACCENT_COLOR)

    lines = legend_lines()
    panel_top = 92
    panel_height = 20 + len(lines) * 24
    draw.rounded_rectangle(
        (14, panel_top, 250, panel_top + panel_height),
        radius=10,
        fill=PANEL_COLOR + (170,),
    )

    y = panel_top + 12
    for line in lines:
        is_active = note is not None and line.startswith(note)
        draw.text((30, y), line, font=FONT_LEGEND, fill=ACCENT_COLOR if is_active else TEXT_COLOR)
        y += 24

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


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
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO,
    )

    history = deque(maxlen=HISTORY_LEN)
    playing_note = None
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

            detected_note = None
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks)
                detected_note = note_from_gesture(extended_fingers(landmarks))

            history.append(detected_note)
            current_note = stable_note(history)

            if current_note is not None and current_note != playing_note:
                play_note(NOTE_FREQS[current_note])
            playing_note = current_note

            cv2.imshow("Hand Note Synth", draw_overlay(frame, current_note))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
