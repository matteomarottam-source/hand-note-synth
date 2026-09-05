import argparse
import os
import threading
import time
import urllib.request
from collections import deque

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

# (international name, italian name, frequency in Hz)
NOTES = [
    ("C", "Do", 261.63),
    ("D", "Re", 293.66),
    ("E", "Mi", 329.63),
    ("F", "Fa", 349.23),
    ("G", "Sol", 392.00),
    ("A", "La", 440.00),
    ("B", "Si", 493.88),
]

SAMPLE_RATE = 44100
NOTE_DURATION = 1.6

# relative amplitude of each harmonic; the fast roll-off plus the percussive
# envelope in build_note_waves is what makes an additive tone read as "piano"
PIANO_HARMONICS = [(1, 1.00), (2, 0.55), (3, 0.32), (4, 0.18), (5, 0.10), (6, 0.06), (7, 0.03)]

# the hand is detected on a small copy of the frame while the HUD is drawn at
# display size: landmarks are normalized, so the smaller image costs less CPU
# without moving anything on screen
DISPLAY_WIDTH = 800
DETECT_WIDTH = 384

# pinch distance relative to hand size, with hysteresis so a hand held near the
# threshold does not retrigger the note over and over
PINCH_ON = 0.35
PINCH_OFF = 0.50

CURSOR_SMOOTHING = 0.45
FLASH_SECONDS = 0.28

COLOR_PANEL = (24, 20, 17)
COLOR_KEY = (46, 40, 35)
COLOR_TEXT = (244, 240, 236)
COLOR_MUTED = (168, 160, 152)
COLOR_ACCENT = (66, 170, 255)
COLOR_CURSOR = (255, 206, 120)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONTS = {
    "hero": load_font(52),
    "label": load_font(24),
    "small": load_font(15),
}

_sprite_cache = {}


def text_sprite(text, font_key, color):
    """Renders text once through PIL and caches it as a BGR + alpha pair.

    Converting the whole frame to PIL and back every frame was the most
    expensive part of the old HUD; small cached sprites blit with plain numpy
    instead, which keeps the text crisp without the per-frame conversion.
    """
    key = (text, font_key, color)
    sprite = _sprite_cache.get(key)
    if sprite is not None:
        return sprite

    font = FONTS[font_key]
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = measure.textbbox((0, 0), text, font=font)

    image = Image.new("RGBA", (right - left + 4, bottom - top + 4), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((2 - left, 2 - top), text, font=font, fill=color[::-1] + (255,))

    rgba = np.array(image)
    bgr = rgba[:, :, 2::-1].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0

    sprite = (bgr, alpha)
    _sprite_cache[key] = sprite
    return sprite


def blit(frame, sprite, x, y, center=False):
    bgr, alpha = sprite
    h, w = alpha.shape[:2]
    if center:
        x -= w // 2
        y -= h // 2

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 >= x2 or y1 >= y2:
        return

    src_bgr = bgr[y1 - y:y2 - y, x1 - x:x2 - x]
    src_alpha = alpha[y1 - y:y2 - y, x1 - x:x2 - x]
    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * (1 - src_alpha) + src_bgr * src_alpha).astype(np.uint8)


def fill_panel(frame, x1, y1, x2, y2, color, opacity):
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x1 >= x2 or y1 >= y2:
        return

    roi = frame[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), opacity, roi, 1 - opacity, 0, dst=roi)


def build_note_waves():
    """Synthesizes every note once at startup instead of on each trigger."""
    t = np.linspace(0, NOTE_DURATION, int(SAMPLE_RATE * NOTE_DURATION), endpoint=False)
    envelope = np.exp(-3.0 * t)
    attack = int(SAMPLE_RATE * 0.006)
    envelope[:attack] *= np.linspace(0, 1, attack)

    waves = []
    for _, _, freq in NOTES:
        wave = np.zeros_like(t)
        for harmonic, amplitude in PIANO_HARMONICS:
            wave += amplitude * np.sin(2 * np.pi * freq * harmonic * t)
        wave *= envelope
        wave *= 0.35 / np.max(np.abs(wave))
        waves.append(wave.astype(np.float32))

    return waves


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Scarico il modello hand_landmarker.task (solo la prima volta)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def keyboard_layout(width, height):
    margin = int(width * 0.05)
    key_height = int(height * 0.17)
    top = height - key_height - int(height * 0.06)
    key_width = (width - 2 * margin) / len(NOTES)

    rects = []
    for i in range(len(NOTES)):
        x1 = int(margin + i * key_width) + 4
        x2 = int(margin + (i + 1) * key_width) - 4
        rects.append((x1, top, x2, top + key_height))

    return rects, margin, key_width


def key_at(cursor_x, width, margin, key_width):
    index = int((cursor_x * width - margin) / key_width)
    return max(0, min(index, len(NOTES) - 1))


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], COLOR_PANEL, 4, cv2.LINE_AA)
        cv2.line(frame, points[a], points[b], COLOR_MUTED, 1, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 3, COLOR_TEXT, -1, cv2.LINE_AA)


def draw_cursor(frame, position, pinching, keyboard_top):
    x, y = position
    cv2.line(frame, (x, y), (x, keyboard_top), COLOR_CURSOR, 1, cv2.LINE_AA)

    if pinching:
        cv2.circle(frame, (x, y), 16, COLOR_ACCENT, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 24, COLOR_ACCENT, 2, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), 12, COLOR_CURSOR, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3, COLOR_CURSOR, -1, cv2.LINE_AA)


def draw_keyboard(frame, rects, active_key, flashing_key):
    for i, (x1, y1, x2, y2) in enumerate(rects):
        international, italian, _ = NOTES[i]

        if i == flashing_key:
            fill_panel(frame, x1, y1, x2, y2, COLOR_ACCENT, 0.85)
            label_color, sub_color = COLOR_PANEL, COLOR_PANEL
        elif i == active_key:
            fill_panel(frame, x1, y1, x2, y2, COLOR_KEY, 0.85)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ACCENT, 2, cv2.LINE_AA)
            label_color, sub_color = COLOR_TEXT, COLOR_ACCENT
        else:
            fill_panel(frame, x1, y1, x2, y2, COLOR_PANEL, 0.72)
            label_color, sub_color = COLOR_TEXT, COLOR_MUTED

        center_x = (x1 + x2) // 2
        blit(frame, text_sprite(italian, "label", label_color), center_x, y1 + 30, center=True)
        blit(frame, text_sprite(international, "small", sub_color), center_x, y2 - 22, center=True)


def draw_hud(frame, note_index, fps, hand_visible):
    width = frame.shape[1]

    fill_panel(frame, 24, 24, 300, 116, COLOR_PANEL, 0.78)
    if note_index is None:
        blit(frame, text_sprite("—", "hero", COLOR_MUTED), 44, 36)
        blit(frame, text_sprite("nessuna nota", "small", COLOR_MUTED), 46, 96)
    else:
        international, italian, freq = NOTES[note_index]
        blit(frame, text_sprite(italian, "hero", COLOR_ACCENT), 44, 36)
        blit(frame, text_sprite(f"{international} · {freq:.0f} Hz", "small", COLOR_MUTED), 46, 96)

    status = "mano rilevata" if hand_visible else "mostra la mano alla camera"
    status_color = COLOR_TEXT if hand_visible else COLOR_ACCENT
    blit(frame, text_sprite(status, "small", status_color), 24, 130)

    fps_sprite = text_sprite(f"{fps:.0f} FPS", "small", COLOR_MUTED)
    blit(frame, fps_sprite, width - 24 - fps_sprite[1].shape[1], 28)

    hint = "Muovi la mano per scegliere la nota  ·  pizzica pollice e indice per suonare  ·  Q per uscire"
    blit(frame, text_sprite(hint, "small", COLOR_MUTED), frame.shape[1] // 2, frame.shape[0] - 22, center=True)


class CameraStream:
    """Reads frames in a background thread and always exposes the latest one.

    A plain cv2.VideoCapture().read() call processes frames in the order the
    stream delivers them, so if processing is slower than the incoming
    framerate - very common with a phone streamed over WiFi - the displayed
    frame falls further and further behind real time. Grabbing continuously in
    the background and dropping stale frames keeps the lag from accumulating.
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
    parser = argparse.ArgumentParser(description="Hand gesture piano")
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
    waves = build_note_waves()

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

    cursor = None
    pinching = False
    playing_key = None
    flash_until = 0.0
    frame_times = deque(maxlen=30)
    start_time = time.time()
    last_timestamp_ms = -1
    layout_cache = None

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok or frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if frame.shape[1] != DISPLAY_WIDTH:
                scale = DISPLAY_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            if layout_cache is None or layout_cache[0] != (width, height):
                layout_cache = ((width, height), *keyboard_layout(width, height))
            _, key_rects, margin, key_width = layout_cache
            keyboard_top = key_rects[0][1]

            detect_scale = DETECT_WIDTH / width
            small = cv2.resize(frame, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
            )

            timestamp_ms = max(int((time.time() - start_time) * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            active_key = None
            hand_visible = bool(result.hand_landmarks)

            if hand_visible:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks)

                thumb, index = landmarks[4], landmarks[8]
                target = ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2)
                cursor = target if cursor is None else (
                    cursor[0] + (target[0] - cursor[0]) * CURSOR_SMOOTHING,
                    cursor[1] + (target[1] - cursor[1]) * CURSOR_SMOOTHING,
                )

                hand_size = ((landmarks[0].x - landmarks[9].x) ** 2 + (landmarks[0].y - landmarks[9].y) ** 2) ** 0.5
                pinch = (((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5) / max(hand_size, 1e-6)

                active_key = key_at(cursor[0], width, margin, key_width)

                if not pinching and pinch < PINCH_ON:
                    pinching = True
                elif pinching and pinch > PINCH_OFF:
                    pinching = False
                    playing_key = None

                # retriggering while pinched lets you slide across keys
                if pinching and active_key != playing_key:
                    sd.play(waves[active_key], SAMPLE_RATE)
                    playing_key = active_key
                    flash_until = loop_start + FLASH_SECONDS

                draw_cursor(
                    frame,
                    (int(cursor[0] * width), int(cursor[1] * height)),
                    pinching,
                    keyboard_top,
                )
            else:
                cursor = None
                pinching = False
                playing_key = None

            frame_times.append(max(time.time() - loop_start, 1e-6))
            fps = len(frame_times) / sum(frame_times)

            draw_keyboard(frame, key_rects, active_key, playing_key if loop_start < flash_until else None)
            draw_hud(frame, active_key, fps, hand_visible)

            cv2.imshow("Hand Piano", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
