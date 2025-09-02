import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
from PIL import Image, ImageTk
import threading
import time
import matplotlib.pyplot as plt

# ------------------------- تنظیمات -------------------------
IMG_SIZE = (224, 224)                 # سایز ورودی مدل
CLASS_NAMES = ['Normal', 'Pneumonia'] # کلاس‌ها
model = None                          # مدل (بعداً بارگذاری میشه)

# ------------------------- تابع پیش‌بینی -------------------------
def predict_image(img_path, img_size=IMG_SIZE):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    if predictions.shape[1] == 1:
        prob = predictions[0][0]
        return {
            CLASS_NAMES[0]: float(1 - prob),
            CLASS_NAMES[1]: float(prob)
        }
    else:
        return {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}

# ------------------------- بارگذاری مدل با Progress Bar -------------------------
def load_model_with_progress(path):
    global model
    progress_bar["value"] = 0
    status_label.config(text="در حال بارگذاری مدل...")

    # غیر فعال کردن دکمه‌ها
    select_model_btn.config(state="disabled")
    select_image_btn.config(state="disabled")

    def task():
        nonlocal path
        for i in range(1, 101):
            time.sleep(0.02)  # فقط برای زیبایی
            progress_bar["value"] = i
            root.update_idletasks()

        try:
            loaded_model = tf.keras.models.load_model(path)
            status_label.config(text="✅ مدل با موفقیت بارگذاری شد")
            global model
            model = loaded_model
            select_image_btn.config(state="normal")  # فعال کردن انتخاب تصویر
        except Exception as e:
            status_label.config(text=f"❌ خطا در بارگذاری مدل: {e}")
        finally:
            select_model_btn.config(state="normal")

    threading.Thread(target=task).start()

# ------------------------- انتخاب مدل -------------------------
def open_model():
    file_path = filedialog.askopenfilename(
        filetypes=[("Model files", "*.h5 *.keras")]
    )
    if file_path:
        load_model_with_progress(file_path)

# ------------------------- انتخاب تصویر -------------------------
def open_file():
    if model is None:
        status_label.config(text="⚠️ ابتدا یک مدل انتخاب کنید")
        return

    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    # نمایش تصویر انتخاب‌شده
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # پیش‌بینی
    results = predict_image(file_path)

    # نمایش نتیجه متنی
    result_text = ""
    for cls, prob in results.items():
        result_text += f"{cls}: {prob*100:.2f}%\n"
    result_label.config(text=result_text)

    # 📊 رسم نمودار میله‌ای
    plt.figure(figsize=(6, 4))
    plt.bar(results.keys(), [p * 100 for p in results.values()], color=['green', 'red'])
    plt.title("Prediction Probabilities")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()

# ------------------------- رابط کاربری -------------------------
root = tk.Tk()
root.title("تشخیص سینه‌پهلو از تصویر")

# دکمه انتخاب مدل
select_model_btn = Button(root, text="انتخاب مدل", command=open_model, font=("Arial", 12))
select_model_btn.pack(pady=10)

# Progress Bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=5)

status_label = Label(root, text="هیچ مدلی انتخاب نشده", font=("Arial", 10))
status_label.pack(pady=5)

# دکمه انتخاب تصویر
select_image_btn = Button(root, text="انتخاب تصویر", command=open_file, font=("Arial", 12), state="disabled")
select_image_btn.pack(pady=10)

# جای تصویر
image_label = Label(root)
image_label.pack()

# نتیجه متنی
result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
