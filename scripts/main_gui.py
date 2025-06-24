import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont  # ### ИЗМЕНЕНО ### Добавлены ImageDraw и ImageFont
import threading
import os
import time
from collections import deque
import platform  # ### ИЗМЕНЕНО ### Для определения ОС

# --- Импорты из вашего проекта ---
from train_model.cnn_lstm_model import predict_on_sequence, load_model_once, IMG_SIZE, SEQUENCE_LENGTH

# --- Константы ---
OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ### ИЗМЕНЕНО ### Функция для поиска подходящего шрифта
def get_font_path():
    system = platform.system()
    if system == "Windows":
        # Arial - стандартный шрифт Windows, поддерживает кириллицу
        return "C:/Windows/Fonts/arial.ttf"
    elif system == "Darwin":  # macOS
        return "/System/Library/Fonts/Supplemental/Arial.ttf"
    else:  # Linux
        # Попробуем найти один из распространенных шрифтов
        common_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    # Если ничего не найдено, возвращаем None
    return None


class AnalysisWindow(tk.Toplevel):
    """Окно для отображения и анализа видео."""
    TEXT_BELLY = "Техника соблюдена. Дыхание ровное"
    TEXT_CHEST_WEBCAM = ["Пожалуйста, расслабьтесь и дышите глубже.", "Откиньте спину назад и расслабьте мышцы тела."]
    TEXT_CHEST_FILE = "Человек напряжен или в стрессе. Ему следует дышать глубже."

    def __init__(self, parent, source):
        super().__init__(parent)
        self.source = source
        self.is_webcam = (source == 0)

        title_prefix = "Веб-камера" if self.is_webcam else "Видеофайл"
        self.title(f"Анализ дыхания: {title_prefix}")
        self.geometry("800x650")

        # ### ИЗМЕНЕНО ### Загружаем шрифт при инициализации
        font_path = get_font_path()
        if font_path:
            self.font_big = ImageFont.truetype(font_path, 24)
            self.font_medium = ImageFont.truetype(font_path, 18)
            self.font_small = ImageFont.truetype(font_path, 16)
        else:
            messagebox.showwarning("Шрифт не найден",
                                   "Не удалось найти кириллический шрифт. Текст может отображаться некорректно.")
            self.font_big = self.font_medium = self.font_small = ImageFont.load_default()

        self.is_running = threading.Event()
        self.thread = None
        self.processed_frames_for_saving = []
        self.video_fps = 30

        self.model = load_model_once()
        if self.model is None:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель!")
            self.destroy()
            return

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ### ИЗМЕНЕНО ### Переписано с использованием Pillow
    def draw_text_with_background(self, draw, text, position, font, text_color, bg_color):
        """Рисует текст с прямоугольным фоном."""
        x, y = position
        # Получаем размеры текста
        text_bbox = draw.textbbox((x, y), text, font=font)
        # Добавляем отступы
        padding = 5
        bg_bbox = (text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding)
        draw.rectangle(bg_bbox, fill=bg_color)
        draw.text((x, y), text, font=font, fill=text_color)

    # ### ИЗМЕНЕНО ### Переписано с использованием Pillow
    def draw_centered_text(self, draw, frame_size, text_lines, font):
        """Рисует многострочный текст по центру."""
        w, h = frame_size

        # Вычисляем высоту блока текста
        line_height = font.getbbox("test")[3] + 10
        total_text_height = len(text_lines) * line_height
        y_start = (h - total_text_height) // 2

        current_y = y_start
        for line in text_lines:
            text_width = font.getbbox(line)[2]
            text_x = (w - text_width) // 2
            self.draw_text_with_background(draw, line, (text_x, current_y), font, (255, 255, 0),
                                           "black")  # Желтый текст на черном фоне
            current_y += line_height

    def video_loop(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror("Ошибка", f"Не удалось открыть источник видео: {self.source}"))
            return

        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        prediction_text = "Belly"

        while self.is_running.is_set():
            ret, frame_bgr = cap.read()
            if not ret:
                if not self.is_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            resized = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
            normalized = resized / 255.0
            frame_buffer.append(normalized)

            if len(frame_buffer) == SEQUENCE_LENGTH:
                sequence = np.array(frame_buffer)
                prediction_text = predict_on_sequence(sequence)

            # ### ИЗМЕНЕНО ### Вся отрисовка текста теперь через Pillow
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)

            w, h = pil_img.size

            if prediction_text == "Belly":
                self.draw_text_with_background(draw, self.TEXT_BELLY, (20, 20), self.font_medium, (0, 255, 0), "black")
            else:  # prediction_text == "Chest"
                if self.is_webcam:
                    self.draw_centered_text(draw, pil_img.size, self.TEXT_CHEST_WEBCAM, self.font_big)
                else:
                    self.draw_text_with_background(draw, self.TEXT_CHEST_FILE, (20, h - 50), self.font_small,
                                                   (255, 165, 0), "black")

            # Конвертируем обратно в формат OpenCV для сохранения
            final_frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            self.processed_frames_for_saving.append(final_frame_bgr)
            self.after(0, self.update_video_display, np.array(pil_img))  # Передаем RGB массив
            time.sleep(1 / (self.video_fps * 2))

        cap.release()
        self.after(0, self.stop_analysis)

    def update_video_display(self, frame_rgb):  # ### ИЗМЕНЕНО ### Принимает RGB массив
        if not self.winfo_exists(): return

        # Конвертация больше не нужна, уже в правильном формате
        img = Image.fromarray(frame_rgb)

        lbl_w, lbl_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if lbl_w > 1 and lbl_h > 1:
            img_w, img_h = img.size
            scale = min(lbl_w / img_w, lbl_h / img_h)
            new_size = (int(img_w * scale), int(img_h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    # Остальные функции без изменений
    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        self.video_label = ttk.Label(main_frame, background="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        control_frame.columnconfigure([0, 1, 2], weight=1)
        self.btn_start = ttk.Button(control_frame, text="▶ Старт", command=self.start_analysis)
        self.btn_start.grid(row=0, column=0, padx=5, sticky="ew")
        self.btn_stop = ttk.Button(control_frame, text="⏹ Стоп", command=self.stop_analysis, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=1, padx=5, sticky="ew")
        self.btn_save = ttk.Button(control_frame, text="💾 Сохранить", command=self.save_video, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=2, padx=5, sticky="ew")

    def start_analysis(self):
        if self.thread and self.thread.is_alive(): return
        self.is_running.set()
        self.processed_frames_for_saving.clear()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def stop_analysis(self):
        self.is_running.clear()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.processed_frames_for_saving: self.btn_save.config(state=tk.NORMAL)

    def save_video(self):
        if not self.processed_frames_for_saving: return
        filename = filedialog.asksaveasfilename(initialdir=OUTPUT_DIR, title="Сохранить видео как...",
                                                defaultextension=".mp4", filetypes=[("MP4 видео", "*.mp4")])
        if not filename: return
        self.btn_save.config(text="Сохранение...", state=tk.DISABLED)
        self.update_idletasks()
        frames_copy = list(self.processed_frames_for_saving)
        save_thread = threading.Thread(target=self._save_video_worker, args=(filename, frames_copy), daemon=True)
        save_thread.start()

    def _save_video_worker(self, filename, frames_to_save):
        try:
            height, width, _ = frames_to_save[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, self.video_fps, (width, height))
            for frame in frames_to_save:
                out.write(frame)
            out.release()

            def on_save_success():
                messagebox.showinfo("Успех", f"Видео сохранено в {filename}")
                self.destroy()

            self.after(0, on_save_success)
        except Exception as e:
            def on_save_error():
                messagebox.showerror("Ошибка сохранения", str(e))
                self.btn_save.config(text="💾 Сохранить", state=tk.NORMAL)

            self.after(0, on_save_error)

    def on_close(self):
        self.is_running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(0.5)
        self.destroy()


# --- Класс App и секция if __name__ == "__main__" остаются без изменений ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Система анализа дыхания")
        self.geometry("500x300")
        style = ttk.Style(self)
        style.configure("TButton", font=("Helvetica", 12), padding=10)
        style.configure("TLabel", font=("Helvetica", 14), padding=10)
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure([0, 1, 2, 3], weight=1)
        main_frame.columnconfigure(0, weight=1)
        ttk.Label(main_frame, text="Выберите режим работы:", anchor="center").grid(row=0, column=0)
        btn_from_file = ttk.Button(main_frame, text="Загрузить видео с диска", command=self.run_from_file)
        btn_from_file.grid(row=1, column=0, sticky="ew", padx=20, pady=5)
        btn_from_webcam = ttk.Button(main_frame, text="Анализ с веб-камеры", command=self.run_from_webcam)
        btn_from_webcam.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
        btn_exit = ttk.Button(main_frame, text="Выйти", command=self.destroy)
        btn_exit.grid(row=3, column=0, sticky="ew", padx=20, pady=(15, 5))

    def run_from_file(self):
        filepath = filedialog.askopenfilename(title="Выберите видеофайл",
                                              filetypes=[("Видеофайлы", "*.mp4 *.avi *.mov"), ("Все файлы", "*.*")])
        if filepath: AnalysisWindow(self, source=filepath)

    def run_from_webcam(self):
        AnalysisWindow(self, source=0)


if __name__ == "__main__":
    print("Загрузка модели... Пожалуйста, подождите.")
    if load_model_once() is not None:
        print("Модель успешно загружена.")
        app = App()
        app.mainloop()
    else:
        print(f"Критическая ошибка: не удалось загрузить модель!")