import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ------------------------- تنظیمات -------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Pneumonia']
model = None

# ------------------------- پیش‌بینی تصویر -------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    if predictions.shape[1] == 1:
        prob = predictions[0][0]
        return {CLASS_NAMES[0]: float(1 - prob), CLASS_NAMES[1]: float(prob)}
    else:
        return {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}

# ------------------------- بارگذاری مدل با Progress Bar -------------------------
def load_model_with_progress(path):
    global model
    progress_bar["value"] = 0
    status_label.config(text="در حال بارگذاری مدل...")
    select_model_btn.config(state="disabled")
    select_image_btn.config(state="disabled")
    def task():
        nonlocal path
        for i in range(1, 101):
            time.sleep(0.01)
            progress_bar["value"] = i
            root.update_idletasks()
        try:
            loaded_model = tf.keras.models.load_model(path)
            global model
            model = loaded_model
            status_label.config(text="✅ مدل بارگذاری شد")
            select_image_btn.config(state="normal")
        except Exception as e:
            status_label.config(text=f"❌ خطا: {e}")
        finally:
            select_model_btn.config(state="normal")
    threading.Thread(target=task).start()

# ------------------------- انتخاب مدل -------------------------
def open_model():
    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5 *.keras")])
    if file_path:
        load_model_with_progress(file_path)

# ------------------------- انتخاب تصویر -------------------------
def open_file():
    if model is None:
        status_label.config(text="⚠️ ابتدا مدل را انتخاب کنید")
        return
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # نمایش تصویر با قاب
    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # پیش‌بینی
    results = predict_image(file_path)

    # نمایش درصدها با رنگ
    result_text = ""
    for cls, prob in results.items():
        color = "green" if cls=="Normal" else "red"
        result_text += f"{cls}: {prob*100:.2f}%\n"
    result_label.config(text=result_text, fg="blue", font=("Arial", 14, "bold"))

    # رسم نمودار میله‌ای داخل GUI
    fig.clf()
    ax = fig.add_subplot(111)
    ax.bar(results.keys(), [p*100 for p in results.values()], color=['green','red'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Prediction Probabilities")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    canvas.draw()

# ------------------------- ساخت پنجره -------------------------
root = tk.Tk()
root.title("💉 تشخیص سینه‌پهلو از تصویر")
root.geometry("650x600")
root.configure(bg="#f0f8ff")

# دکمه انتخاب مدل
select_model_btn = tk.Button(root, text="🔹 انتخاب مدل", command=open_model, font=("Arial", 12, "bold"), bg="#4682B4", fg="white", width=20)
select_model_btn.pack(pady=10)

# نوار پیشرفت
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=5)

status_label = tk.Label(root, text="هیچ مدلی انتخاب نشده", font=("Arial", 10), bg="#f0f8ff")
status_label.pack(pady=5)

# دکمه انتخاب تصویر
select_image_btn = tk.Button(root, text="📁 انتخاب تصویر", command=open_file, font=("Arial", 12, "bold"), bg="#32CD32", fg="white", width=20, state="disabled")
select_image_btn.pack(pady=10)

# نمایش تصویر
image_label = tk.Label(root, bg="white", relief="solid", bd=2)
image_label.pack(pady=10)

# نتیجه متنی
result_label = tk.Label(root, text="", font=("Arial", 12), bg="#f0f8ff")
result_label.pack(pady=5)

# نمودار پیش‌بینی داخل GUI
fig = plt.Figure(figsize=(5,3), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

root.mainloop()
