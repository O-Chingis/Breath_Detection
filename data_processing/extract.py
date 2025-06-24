import cv2
import numpy as np
import random

IMG_SIZE = 128
SEQUENCE_LENGTH = 16


def augment_frame(frame):
    # случайная горизонтальная зеркалка
    if random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    # случайное увеличение/уменьшение яркости
    if random.random() < 0.5:
        factor = 0.5 + random.uniform(0, 1.5)
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)

    # случайный поворот (±10 градусов)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (IMG_SIZE, IMG_SIZE))

    return frame

def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH, augment=False):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, sequence_length).astype(int)

    frames = []
    current_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            if augment:
                frame = augment_frame(frame)
            frames.append(frame / 255.0)
        current_idx += 1
    cap.release()
    return np.array(frames) if len(frames) == sequence_length else None
