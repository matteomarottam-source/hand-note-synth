import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Impossibile aprire la camera '{source}'. Prova un altro indice con --camera.")
        return

    last_note = None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_note = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                extended = extended_fingers(hand_landmarks.landmark)
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
