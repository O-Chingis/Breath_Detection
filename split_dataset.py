import os
import shutil
import random

# Пути к исходным аугментированным данным
AUGMENTED_DIR = "data_augmented"
TRAIN_DIR = "data_train"
VAL_DIR = "data_val"

# Параметры разбиения
SPLITS = {
    "chest": (280, 70),  # train, val
    "belly": (280, 70), # train, val
}

def make_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def split_and_copy(class_name, src_dir, dst_train, dst_val, n_train, n_val):
    files = [f for f in os.listdir(src_dir) if f.endswith(".mp4")]
    if len(files) < n_train + n_val:
        raise ValueError(f"❌ Не хватает файлов для класса '{class_name}': нужно минимум {n_train + n_val}, найдено {len(files)}")

    random.shuffle(files)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]

    for f in train_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_train, f))

    for f in val_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_val, f))

    print(f"✅ Класс '{class_name}': скопировано {n_train} в train, {n_val} в val.")

def main():
    random.seed(42)

    for class_name in ["chest", "belly"]:
        src_class_dir = os.path.join(AUGMENTED_DIR, class_name)
        dst_train_dir = os.path.join(TRAIN_DIR, class_name)
        dst_val_dir = os.path.join(VAL_DIR, class_name)

        make_clean_dir(dst_train_dir)
        make_clean_dir(dst_val_dir)

        n_train, n_val = SPLITS[class_name]
        split_and_copy(class_name, src_class_dir, dst_train_dir, dst_val_dir, n_train, n_val)

if __name__ == "__main__":
    main()
