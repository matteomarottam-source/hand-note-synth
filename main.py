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

SAMPLE_RATE = 44100
NOTE_DURATION = 2.6
MAX_VOICES = 24
BLOCK_SIZE = 512

STRING_COUNT = 15
BASE_MIDI = 48  # C3, a warm harp register

# a pentatonic scale has no dissonant pair, so any strum across the strings
# sounds musical - the diatonic scale is one keypress away for full melodies
SCALES = [
    ("Pentatonica", [0, 2, 4, 7, 9]),
    ("Completa", [0, 2, 4, 5, 7, 9, 11]),
]

SEMITONE_NAMES = {0: "Do", 2: "Re", 4: "Mi", 5: "Fa", 7: "Sol", 9: "La", 11: "Si"}

# the hand is detected on a small copy of the frame while the HUD is drawn at
# display size: landmarks are normalized, so the smaller image costs less CPU
# without moving anything on screen
DISPLAY_WIDTH = 860
DETECT_WIDTH = 384

FINGERTIPS = [4, 8, 12, 16, 20]
PLUCK_COOLDOWN = 0.12

STRING_DECAY = 0.86
STRING_SPEED = 34.0
RIPPLE_LIFE = 0.45
LABEL_LIFE = 0.85

COLOR_DIM = (18, 14, 12)
COLOR_PANEL = (26, 21, 18)
COLOR_TEXT = (246, 242, 238)
COLOR_MUTED = (162, 154, 146)
COLOR_STRING = (150, 138, 128)
COLOR_ACCENT = (86, 176, 255)
COLOR_GLOW = (255, 208, 128)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"),
    ("C:/Windows/Fonts/calibrib.ttf", "C:/Windows/Fonts/calibri.ttf"),
    ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"),
]


def load_font(size, bold=True):
    for bold_path, regular_path in FONT_CANDIDATES:
        path = bold_path if bold else regular_path
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONTS = {
    "title": load_font(26),
    "note": load_font(30),
    "label": load_font(16),
    "small": load_font(14, bold=False),
}

_sprite_cache = {}


def text_sprite(text, font_key, color):
    """Renders text once through PIL and caches it as a BGR + alpha pair.

    Converting the whole frame to PIL and back on every frame was the most
    expensive part of the earlier HUD; small cached sprites blit with plain
    numpy instead, which keeps the text crisp without the conversion.
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
    sprite = (rgba[:, :, 2::-1].astype(np.float32), rgba[:, :, 3:4].astype(np.float32) / 255.0)
    _sprite_cache[key] = sprite
    return sprite


def blit(frame, sprite, x, y, center=False, opacity=1.0):
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
    if opacity < 1.0:
        src_alpha = src_alpha * opacity

    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * (1 - src_alpha) + src_bgr * src_alpha).astype(np.uint8)


def fill_panel(frame, x1, y1, x2, y2, color, opacity):
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x1 >= x2 or y1 >= y2:
        return

    roi = frame[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), opacity, roi, 1 - opacity, 0, dst=roi)


def mix_color(color_a, color_b, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))


def note_name(midi):
    semitone = midi % 12
    return f"{SEMITONE_NAMES.get(semitone, '?')}{midi // 12 - 1}"


def scale_midi_notes(offsets):
    notes = []
    for i in range(STRING_COUNT):
        octave, step = divmod(i, len(offsets))
        notes.append(BASE_MIDI + 12 * octave + offsets[step])
    return notes


def build_waves(midi_notes):
    """Synthesizes every string once at startup instead of on each pluck."""
    t = np.linspace(0, NOTE_DURATION, int(SAMPLE_RATE * NOTE_DURATION), endpoint=False)
    attack = int(SAMPLE_RATE * 0.004)
    attack_ramp = np.linspace(0, 1, attack)

    waves = []
    for midi in midi_notes:
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        wave = np.zeros_like(t)

        for harmonic in range(1, 8):
            # higher harmonics fade faster than the fundamental, which is what
            # separates a plucked string from a flat synth tone
            decay = (1.4 + 2.0 * freq / 440.0) * harmonic ** 0.7
            amplitude = 1.0 / harmonic ** 1.6
            # slight inharmonicity, as in a real stretched string
            partial = freq * harmonic * (1 + 0.0004 * harmonic ** 2)
            wave += amplitude * np.sin(2 * np.pi * partial * t) * np.exp(-decay * t)

        wave[:attack] *= attack_ramp
        wave *= 0.5 / np.max(np.abs(wave))
        waves.append(wave.astype(np.float32))

    return waves


class Synth:
    """Polyphonic mixer so plucked strings keep ringing over each other."""

    def __init__(self):
        self.voices = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self.stream.start()

    def pluck(self, wave, gain):
        with self.lock:
            if len(self.voices) >= MAX_VOICES:
                self.voices.pop(0)
            self.voices.append([wave, 0, gain])

    def _callback(self, outdata, frames, time_info, status):
        buffer = np.zeros(frames, dtype=np.float32)

        with self.lock:
            for voice in self.voices:
                wave, position, gain = voice
                chunk = wave[position:position + frames]
                if chunk.size:
                    buffer[:chunk.size] += chunk * gain
                voice[1] = position + frames
            self.voices = [v for v in self.voices if v[1] < v[0].size]

        np.clip(buffer, -1.0, 1.0, out=buffer)
        outdata[:, 0] = buffer

    def close(self):
        self.stream.stop()
        self.stream.close()


class Harp:
    """String geometry, vibration state and pluck detection."""

    def __init__(self, width, height):
        margin = int(width * 0.075)
        self.xs = np.linspace(margin, width - margin, STRING_COUNT)
        self.top = int(height * 0.20)
        self.bottom = int(height * 0.86)

        self.amplitude = np.zeros(STRING_COUNT)
        self.phase = np.zeros(STRING_COUNT)

        # a standing wave: no displacement at the ends, maximum in the middle
        steps = np.linspace(0, 1, 22)
        self.ys = (self.top + steps * (self.bottom - self.top)).astype(np.int32)
        self.profile = np.sin(np.pi * steps)

    def update(self, dt):
        self.amplitude *= STRING_DECAY ** (dt * 60)
        self.phase += STRING_SPEED * dt
        self.amplitude[self.amplitude < 0.4] = 0.0

    def excite(self, index, strength):
        self.amplitude[index] = 9.0 + 16.0 * strength
        self.phase[index] = 0.0

    def crossings(self, previous, current):
        """Strings whose x lies between the fingertip's last and current position."""
        if not (self.top <= current[1] <= self.bottom):
            return []

        low, high = sorted((previous[0], current[0]))
        return [i for i, x in enumerate(self.xs) if low <= x <= high]

    def draw(self, frame):
        for i, x in enumerate(self.xs):
            amplitude = self.amplitude[i]
            if amplitude > 0:
                offsets = amplitude * self.profile * np.sin(self.phase[i])
                points = np.stack([(x + offsets).astype(np.int32), self.ys], axis=1)
                glow = min(amplitude / 22.0, 1.0)
                cv2.polylines(frame, [points], False, mix_color(COLOR_STRING, COLOR_GLOW, glow), 2, cv2.LINE_AA)
            else:
                cv2.line(frame, (int(x), self.top), (int(x), self.bottom), COLOR_STRING, 1, cv2.LINE_AA)


class CameraStream:
    """Reads frames in a background thread and always exposes the latest one.

    A plain cv2.VideoCapture().read() call processes frames in the order the
    stream delivers them, so if processing is slower than the incoming
    framerate - very common with a phone streamed over WiFi - the displayed
    frame falls further and further behind real time. Grabbing continuously in
    the background and dropping stale frames keeps the lag from accumulating.
    """

    def __init__(self, source, capture_width):
        # DirectShow is more reliable than the default Media Foundation backend
        # for third-party virtual cameras on Windows (e.g. Iriun Webcam).
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        # asking the device for a smaller image is the one lever that reduces
        # what a phone streams over WiFi, rather than paying for pixels we
        # immediately scale away
        if capture_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, round(capture_width * 9 / 16))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.counter = 0
        self.stamps = deque(maxlen=30)
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
                self.counter += 1
                self.stamps.append(time.time())

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ok, None, self.counter
            return self.ok, self.frame.copy(), self.counter

    def fps(self):
        with self.lock:
            stamps = list(self.stamps)
        if len(stamps) < 2:
            return 0.0
        return (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Scarico il modello hand_landmarker.task (solo la prima volta)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_hand(frame, points):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], COLOR_DIM, 4, cv2.LINE_AA)
        cv2.line(frame, points[a], points[b], COLOR_MUTED, 1, cv2.LINE_AA)

    for tip in FINGERTIPS:
        cv2.circle(frame, points[tip], 9, COLOR_GLOW, 1, cv2.LINE_AA)
        cv2.circle(frame, points[tip], 4, COLOR_GLOW, -1, cv2.LINE_AA)


def draw_ripples(frame, ripples, now):
    for ripple in ripples:
        age = (now - ripple["born"]) / RIPPLE_LIFE
        radius = int(12 + 46 * age * ripple["strength"])
        cv2.circle(frame, ripple["pos"], radius, mix_color(COLOR_GLOW, COLOR_DIM, age), 2, cv2.LINE_AA)


def draw_labels(frame, labels, now):
    for label in labels:
        age = (now - label["born"]) / LABEL_LIFE
        x, y = label["pos"]
        blit(frame, text_sprite(label["text"], "note", COLOR_GLOW), x, y - int(40 * age), center=True, opacity=1 - age)


def draw_hud(frame, harp, names, scale_name, recent, fps, camera_fps, hand_visible, now):
    width = frame.shape[1]

    fill_panel(frame, 0, 0, width, 64, COLOR_PANEL, 0.72)
    blit(frame, text_sprite("ARPA", "title", COLOR_ACCENT), 26, 16)
    blit(frame, text_sprite("sfiora le corde con le dita", "small", COLOR_MUTED), 104, 26)

    x = width - 26
    # camera and render rates are shown apart: a low camera rate is the phone
    # link, a low render rate is this machine
    fps_color = COLOR_MUTED if camera_fps >= 15 else COLOR_ACCENT
    fps_sprite = text_sprite(f"cam {camera_fps:.0f} · app {fps:.0f} FPS", "small", fps_color)
    x -= fps_sprite[1].shape[1]
    blit(frame, fps_sprite, x, 24)

    scale_sprite = text_sprite(f"scala {scale_name.lower()}", "small", COLOR_MUTED)
    x -= scale_sprite[1].shape[1] + 22
    blit(frame, scale_sprite, x, 24)

    for note, played_at in reversed(recent):
        sprite = text_sprite(note, "label", COLOR_GLOW)
        x -= sprite[1].shape[1] + 14
        blit(frame, sprite, x, 22, opacity=max(0.15, 1 - (now - played_at) / 4.0))

    for i, name in enumerate(names):
        color = mix_color(COLOR_MUTED, COLOR_GLOW, min(harp.amplitude[i] / 22.0, 1.0))
        blit(frame, text_sprite(name, "small", color), int(harp.xs[i]), harp.bottom + 20, center=True)

    if not hand_visible:
        message = text_sprite("mostra la mano alla camera", "label", COLOR_ACCENT)
        blit(frame, message, width // 2, frame.shape[0] // 2, center=True)

    hint = "muovi la mano tra le corde per suonare  ·  S cambia scala  ·  Q esci"
    blit(frame, text_sprite(hint, "small", COLOR_MUTED), width // 2, frame.shape[0] - 20, center=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Hand tracked air harp")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index (e.g. 0, 1) or stream URL (e.g. for an IP-camera app). Default: 0",
    )
    parser.add_argument(
        "--capture-width",
        type=int,
        default=960,
        help="Resolution requested from the camera; lower it for a laggy wireless camera. Default: 960",
    )
    parser.add_argument(
        "--detect-width",
        type=int,
        default=DETECT_WIDTH,
        help=f"Width the hand detector runs at; lower is faster. Default: {DETECT_WIDTH}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera

    ensure_model()

    scale_index = 0
    scales = []
    for name, offsets in SCALES:
        midi_notes = scale_midi_notes(offsets)
        scales.append((name, build_waves(midi_notes), [note_name(m) for m in midi_notes]))

    cap = CameraStream(source, args.capture_width)
    if not cap.isOpened():
        print(f"Impossibile aprire la camera '{source}'. Prova un altro indice con --camera.")
        return
    cap.start()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO,
    )

    synth = Synth()
    harp = None
    previous_tips = {}
    cooldowns = {}
    ripples = []
    labels = []
    recent = deque(maxlen=5)
    frame_times = deque(maxlen=30)
    start_time = time.time()
    last_timestamp_ms = -1
    last_frame_at = time.time()
    last_detect_at = time.time()
    last_counter = -1
    hands = []

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                now = time.time()
                dt = min(now - last_frame_at, 0.1)
                last_frame_at = now

                ok, frame, counter = cap.read()
                if not ok or frame is None:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                # the render loop runs faster than a wireless camera delivers
                # frames, so detection only reruns on genuinely new images
                # while the animation keeps drawing at full rate
                fresh = counter != last_counter
                last_counter = counter

                if frame.shape[1] != DISPLAY_WIDTH:
                    scale = DISPLAY_WIDTH / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                if harp is None or harp.xs[-1] > width:
                    harp = Harp(width, height)

                if fresh:
                    detect_scale = args.detect_width / width
                    small = cv2.resize(frame, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                    )

                    timestamp_ms = max(int((now - start_time) * 1000), last_timestamp_ms + 1)
                    last_timestamp_ms = timestamp_ms
                    hands = landmarker.detect_for_video(mp_image, timestamp_ms).hand_landmarks

                # the camera feed is a backdrop, so dim it to let the strings read
                fill_panel(frame, 0, 0, width, height, COLOR_DIM, 0.45)

                scale_name, waves, names = scales[scale_index]
                harp.update(dt)

                hand_visible = bool(hands)
                seen_tips = set()

                detect_dt = max(now - last_detect_at, 1e-3)
                for hand_index, landmarks in enumerate(hands):
                    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
                    draw_hand(frame, points)

                    if not fresh:
                        continue

                    for tip in FINGERTIPS:
                        key = (hand_index, tip)
                        seen_tips.add(key)
                        current = points[tip]
                        previous = previous_tips.get(key)
                        previous_tips[key] = current
                        if previous is None:
                            continue

                        speed = abs(current[0] - previous[0]) / detect_dt
                        strength = min(max(speed / 1400.0, 0.22), 1.0)

                        for string_index in harp.crossings(previous, current):
                            if now - cooldowns.get((key, string_index), 0) < PLUCK_COOLDOWN:
                                continue
                            cooldowns[(key, string_index)] = now

                            synth.pluck(waves[string_index], strength)
                            harp.excite(string_index, strength)
                            ripples.append({"pos": current, "born": now, "strength": strength})
                            labels.append({"pos": current, "born": now, "text": names[string_index]})
                            recent.append((names[string_index], now))

                if fresh:
                    last_detect_at = now
                    for key in list(previous_tips):
                        if key not in seen_tips:
                            del previous_tips[key]

                harp.draw(frame)

                ripples = [r for r in ripples if now - r["born"] < RIPPLE_LIFE]
                labels = [l for l in labels if now - l["born"] < LABEL_LIFE]
                draw_ripples(frame, ripples, now)
                draw_labels(frame, labels, now)

                frame_times.append(max(time.time() - now, 1e-6))
                draw_hud(
                    frame,
                    harp,
                    names,
                    scale_name,
                    list(recent),
                    len(frame_times) / sum(frame_times),
                    cap.fps(),
                    hand_visible,
                    now,
                )

                cv2.imshow("Air Harp", frame)

                pressed = cv2.waitKey(1) & 0xFF
                if pressed == ord("q"):
                    break
                if pressed == ord("s"):
                    scale_index = (scale_index + 1) % len(scales)
    finally:
        synth.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
