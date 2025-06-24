import os
from train_model import cnn_lstm_model
from train_model.cnn_lstm_model import create_labels_for_augmented
from inference import real_time_detection
from evaluation import evaluate
import argparse

# Тестовые метки (chest=0, belly=1)
TEST_LABELS = {
    "test1.mp4": 0,
    "test2.mp4": 0,
    "test6.mp4": 1,
    "test7.mp4": 0,
    "test100.mp4": 1,
    "test10.mp4": 0,
    "test11.mp4": 0,
    "test12.mp4": 0,
    "test13.mp4": 0,
    "test14.mp4": 0,
    "test21.mp4": 1,
    "test22.mp4": 1,
    "test23.mp4": 1,
    "test24.mp4": 1,
    "test25.mp4": 1,
    "test26.mp4": 1,
    "test27.mp4": 1,
}

def main():
    parser = argparse.ArgumentParser(description="Breathing detection project")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'infer'], required=True,
                        help="Запуск в режиме: train (обучение), eval (оценка), infer (реальное время)")
    parser.add_argument('--data_dir', type=str, default='./data', help="Папка с оригинальными данными (belly, chest, test)")
    parser.add_argument('--aug_data_dir', type=str, default='./data_augmented', help="Папка с аугментированными данными")
    parser.add_argument('--epochs', type=int, default=10, help="Количество эпох обучения")
    parser.add_argument('--batch_size', type=int, default=2, help="Размер батча")

    args = parser.parse_args()

    if args.mode == 'train':
        train_split_dir = os.path.join("data_split", "train")
        val_split_dir = os.path.join("data_split", "val")

        train_labels = create_labels_for_augmented(args.data_dir, train_split_dir)
        val_labels = create_labels_for_augmented(args.data_dir, val_split_dir)

        all_labels = {**train_labels, **val_labels}

        cnn_lstm_model.train_model(train_split_dir, val_split_dir, all_labels, epochs=args.epochs, batch_size=args.batch_size)

    elif args.mode == 'eval':
        test_path = os.path.join(args.data_dir, "test")
        evaluate.evaluate(test_path, TEST_LABELS)

    elif args.mode == 'infer':
        real_time_detection.real_time_detection()

if __name__ == "__main__":
    main()
