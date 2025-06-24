import os
from train_model import cnn_lstm_model
from sklearn.metrics import classification_report

def evaluate(test_dir, true_labels):
    y_true = []
    y_pred = []

    for video_file, true_label in true_labels.items():
        video_path = os.path.join(test_dir, video_file)
        if not os.path.exists(video_path):
            print(f"Файл {video_file} не найден в {test_dir}")
            continue

        seq = cnn_lstm_model.extract_frames(video_path)
        if seq is None:
            print(f"Не удалось извлечь кадры из {video_file}")
            continue

        prediction = cnn_lstm_model.predict_on_sequence(seq)
        pred_label = 0 if prediction == "Chest" else 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        print(f"{video_file}: Истинная метка = {true_label}, Предсказание = {pred_label}")

    print("\nОтчет по классификации:")
    print(classification_report(y_true, y_pred, target_names=["Chest", "Belly"]))
