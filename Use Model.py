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
from matplotlib.patches import Rectangle

# ------------------------- تنظیمات -------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Pneumonia']
model = None
tooltip = None  # Tooltip widget

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

# ------------------------- بارگذاری مدل -------------------------
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

    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    results = predict_image(file_path)
    result_text = ""
    for cls, prob in results.items():
        result_text += f"{cls}: {prob*100:.2f}%\n"
    result_label.config(text=result_text, fg="blue", font=("Arial", 14, "bold"))

    animate_gradient_glow_bar_chart(results)

# ------------------------- نمودار با Glow، لرزش و Tooltip -------------------------
def animate_gradient_glow_bar_chart(results):
    fig.clf()
    ax = fig.add_subplot(111)
    bars = ax.bar(results.keys(), [0,0], color=['white','white'], edgecolor='black', linewidth=1.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Prediction Probabilities")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    canvas.draw()

    target_values = [p*100 for p in results.values()]
    current_values = [0,0]
    step = 1
    colors = [("limegreen", "darkgreen"), ("red", "darkred")]

    def update_bars():
        done = True
        ax.cla()
        for i, bar in enumerate(bars):
            if current_values[i] < target_values[i]:
                current_values[i] += step
                if current_values[i] > target_values[i]:
                    current_values[i] = target_values[i]
                done = False
            gradient_glow_rect(ax, i, current_values[i], colors[i])
            ax.text(i, current_values[i]+1, f"{current_values[i]:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(results.keys(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Prediction Probabilities")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        canvas.draw()
        if not done:
            root.after(20, update_bars)
        else:
            shake_glow_bars(ax, current_values, colors, steps=5, magnitude=2)
            add_tooltip(ax, current_values, colors)

    update_bars()

# ------------------------- رسم میله با Glow -------------------------
def gradient_glow_rect(ax, idx, height, color_pair):
    n = 50
    for i in range(n):
        h = height*(i+1)/n
        base_color = interpolate_color(color_pair[0], color_pair[1], i/n)
        glow_color = np.clip(np.array(base_color)+0.2,0,1)
        rect = Rectangle((idx-0.4, h - height/n), 0.8, height/n, color=glow_color, linewidth=0)
        ax.add_patch(rect)

# ------------------------- ترکیب رنگ -------------------------
def interpolate_color(c1, c2, t):
    import matplotlib.colors as mcolors
    rgb1 = np.array(mcolors.to_rgb(c1))
    rgb2 = np.array(mcolors.to_rgb(c2))
    rgb = rgb1*(1-t)+rgb2*t
    return rgb

# ------------------------- لرزش میله‌ها با Glow -------------------------
def shake_glow_bars(ax, heights, colors, steps=5, magnitude=2):
    def shake_step(step_count):
        ax.cla()
        for i, h in enumerate(heights):
            shift = magnitude*np.sin(step_count*np.pi/steps)
            gradient_glow_rect(ax, i, h+shift, colors[i])
            ax.text(i, h+shift+1, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Prediction Probabilities")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        canvas.draw()
        if step_count < steps:
            root.after(50, lambda: shake_step(step_count+1))
    shake_step(0)

# ------------------------- Tooltip روی میله‌ها -------------------------
def add_tooltip(ax, heights, colors):
    global tooltip
    if tooltip:
        tooltip.destroy()
        tooltip = None

    tooltip = tk.Label(root, text="", bg="yellow", fg="black", font=("Arial",10,"bold"), bd=1, relief="solid")
    def motion(event):
        for i, bar in enumerate(ax.patches):
            bbox = bar.get_bbox()
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            inv = ax.transData.inverted()
            xdata, ydata = inv.transform([event.x, event.y])
            if x0 <= xdata <= x1 and y0 <= ydata <= y1:
                tooltip.config(text=f"{CLASS_NAMES[i]}: {heights[i]:.1f}%")
                tooltip.place(x=event.x_root - root.winfo_rootx() + 10,
                              y=event.y_root - root.winfo_rooty() + 10)
                return
        tooltip.place_forget()

    canvas.get_tk_widget().bind("<Motion>", motion)
    canvas.get_tk_widget().bind("<Leave>", lambda e: tooltip.place_forget())

# ------------------------- ساخت پنجره -------------------------
root = tk.Tk()
root.title("💉 تشخیص سینه‌پهلو از تصویر")
root.geometry("700x750")
root.configure(bg="#f0f8ff")

select_model_btn = tk.Button(root, text="🔹 انتخاب مدل", command=open_model, font=("Arial", 12, "bold"), bg="#4682B4", fg="white", width=20)
select_model_btn.pack(pady=10)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=5)

status_label = tk.Label(root, text="هیچ مدلی انتخاب نشده", font=("Arial", 10), bg="#f0f8ff")
status_label.pack(pady=5)

select_image_btn = tk.Button(root, text="📁 انتخاب تصویر", command=open_file, font=("Arial", 12, "bold"), bg="#32CD32", fg="white", width=20, state="disabled")
select_image_btn.pack(pady=10)

image_label = tk.Label(root, bg="white", relief="solid", bd=2)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), bg="#f0f8ff")
result_label.pack(pady=5)

fig = plt.Figure(figsize=(6,4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

root.mainloop()
