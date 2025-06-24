import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

IMG_SIZE = 128
SEQUENCE_LENGTH = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "cnn_lstm_breath.keras")
MODEL_PATH = os.path.normpath(MODEL_PATH)


_model = None  # ← добавляем глобальную переменную

def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH):
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
            frames.append(frame / 255.0)
        current_idx += 1
    cap.release()
    return np.array(frames) if len(frames) == sequence_length else None

def build_model(input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)):
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)), input_shape=input_shape),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4))),
        TimeDistributed(MaxPooling2D(2, 2)),
        TimeDistributed(Flatten()),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_labels_for_augmented(data_dir_original, data_dir_augmented):
    labels = {}
    for label_name, label_value in [("belly", 1), ("chest", 0)]:
        folder = os.path.join(data_dir_original, label_name)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".mp4"):
                labels[f] = label_value

    labels_aug = {}
    for label_name in ["belly", "chest"]:
        folder = os.path.join(data_dir_augmented, label_name)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".mp4"):
                original_name = f.split("_aug")[0] + ".mp4"
                if original_name in labels:
                    labels_aug[f] = labels[original_name]
                else:
                    print(f"⚠️ Не найдена метка для аугментированного видео {f}, оригинал {original_name}")
    return labels_aug

def load_data(data_dir, labels_dict):
    X, y = [], []
    for label_name in ["belly", "chest"]:
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Папка не найдена: {folder_path}")
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".mp4"):
                if file not in labels_dict:
                    print(f"⚠️ Метка не найдена для файла {file}")
                    continue
                seq = extract_frames(os.path.join(folder_path, file))
                if seq is not None:
                    X.append(seq)
                    y.append(labels_dict[file])
                else:
                    print(f"⚠️ Не удалось извлечь кадры из {file}")
    return np.array(X), np.array(y)

def train_model(train_dir, val_dir, labels_dict, epochs=10, batch_size=2):
    print("🔍 Загружаем обучающие данные...")
    X_train, y_train = load_data(train_dir, labels_dict)
    print("🔍 Загружаем валидационные данные...")
    X_val, y_val = load_data(val_dir, labels_dict)

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print("📊 Train:")
    for u, c in zip(unique_train, counts_train):
        print(f"  {'Chest' if u == 0 else 'Belly'}: {c}")
    print("📊 Validation:")
    for u, c in zip(unique_val, counts_val):
        print(f"  {'Chest' if u == 0 else 'Belly'}: {c}")

    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ Нет данных для обучения или валидации!")
        return

    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH, save_format='keras')
    print("✅ Модель сохранена.")

def load_model_once():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

def predict_on_sequence(sequence):
    model = load_model_once()
    sequence = np.expand_dims(sequence, axis=0)
    pred = model.predict(sequence)[0][0]
    print(f"[DEBUG] Raw prediction: {pred:.4f}")
    return "Chest" if pred < 0.5 else "Belly"

