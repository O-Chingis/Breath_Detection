import os
import cv2
import numpy as np
import random

INPUT_DIRS = {
    "chest": "data/chest",
    "belly": "data/belly"
}
OUTPUT_BASE_DIR = "data_augmented"
AUGMENTATIONS_PER_VIDEO = 70  # 70 аугментаций на исходное видео
IMG_SIZE = 128
FPS = 15  # частота кадров для сохранения видео


def augment_frame(frame):
    frame = frame.astype(np.float32)

    # Горизонтальный флип с вероятностью 0.5
    if random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    # Изменение яркости и контрастности
    if random.random() < 0.7:
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3)
        frame = frame * contrast_factor + (brightness_factor - 1) * 127
        frame = np.clip(frame, 0, 255)

    # Поворот ±15 градусов
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT101)

    # Масштабирование (zoom in/out) 0.9 - 1.1
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), 0, scale)
        frame = cv2.warpAffine(frame, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT101)

    # Сдвиг ±5 пикселей по X и Y
    if random.random() < 0.5:
        tx = random.uniform(-5, 5)
        ty = random.uniform(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT101)

    # Добавление шума Гаусса
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, frame.shape).astype(np.float32)
        frame = frame + noise
        frame = np.clip(frame, 0, 255)

    return frame.astype(np.uint8)


def augment_and_save_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = augment_frame(frame)
        frames.append(frame)

    cap.release()

    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        for f in frames:
            out.write(f)
        out.release()


def augment_class_videos(class_name, input_dir, output_base_dir, max_videos):
    output_dir = os.path.join(output_base_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)

    videos = [f for f in os.listdir(input_dir) if f.endswith('.mp4')][:max_videos]
    print(f"Обрабатываем класс '{class_name}', найдено {len(videos)} видео.")

    for video_name in videos:
        input_path = os.path.join(input_dir, video_name)
        for j in range(AUGMENTATIONS_PER_VIDEO):
            output_name = f"{os.path.splitext(video_name)[0]}_aug_{j}.mp4"
            output_path = os.path.join(output_dir, output_name)
            augment_and_save_video(input_path, output_path)
            print(f"✅ Сохранено: {output_path}")


if __name__ == "__main__":
    for class_name, input_dir in INPUT_DIRS.items():
        if not os.path.exists(input_dir):
            print(f"⚠️ Папка не найдена: {input_dir}")
            continue

        # Устанавливаем разные лимиты видео для классов
        max_videos = 10 if class_name == "belly" else 6
        augment_class_videos(class_name, input_dir, OUTPUT_BASE_DIR, max_videos)
