# final_gui_with_sound_tooltip.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import time
import base64
import io
import platform
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# optional: simpleaudio for cross-platform wav-from-bytes playback
try:
    import simpleaudio as sa
except Exception:
    sa = None

# ------------------------- Embeded WAV (Base64) - small ding -------------------------
# (یک ding موجی کوتاه و سبک که داخل کد قرار گرفت)
ding_base64 = (
    b'UklGRlIAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAAAgP8A/wD/'
    b'AP8A/wD/AP8A/wD/AP8A/wD/AP8A/wD/AP8A/wD/AP8A/wD/AP8A'
)
# ------------------------- تابع پخش صدا -------------------------
def play_sound_embedded():
    # تلاش می‌کنیم اول با simpleaudio پخش کنیم (cross-platform)
    try:
        wav_bytes = base64.b64decode(ding_base64)
        if sa is not None:
            # simpleaudio expects bytes-like for WaveObject.from_wave_read; use io.BytesIO + WaveObject.from_wave_file
            wav_stream = io.BytesIO(wav_bytes)
            # simpleaudio has WaveObject.from_wave_file which accepts a file-like object in recent versions
            try:
                wave_obj = sa.WaveObject.from_wave_file(wav_stream)
                wave_obj.play()
                return
            except Exception:
                # fallback: write to temp file? prefer winsound if windows
                pass
        # if simpleaudio not available or failed, on Windows use winsound
        if platform.system() == "Windows":
            try:
                import winsound
                # produce two beeps approximating ding
                winsound.Beep(880, 120)
                winsound.Beep(1320, 140)
                return
            except Exception:
                pass
    except Exception as e:
        print("⚠️ play_sound_embedded error:", e)

# ------------------------- تنظیمات -------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Pneumonia']
model = None
tooltip_label = None
current_patches = []  # نگهداری پچ‌ها برای tooltip

# ------------------------- تابع امن برای خواندن خروجی مدل -------------------------
def parse_model_prediction(preds):
    """
    preds: numpy array returned by model.predict(img_array)
    returns dict mapping class->probability (0..1)
    works for:
      - binary sigmoid: shape (1,) or (1,1)
      - multiclass softmax: shape (1, num_classes)
    """
    preds = np.array(preds)
    if preds.ndim == 1:
        # shape (num_classes,) or (1,)
        if preds.size == 1:
            p = float(preds[0])
            return {CLASS_NAMES[0]: 1 - p, CLASS_NAMES[1]: p}
        else:
            # multiclass as 1D
            idx = int(np.argmax(preds))
            probs = preds / preds.sum()  # normalize just in case
            return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    elif preds.ndim == 2:
        if preds.shape[1] == 1:
            p = float(preds[0][0])
            return {CLASS_NAMES[0]: 1 - p, CLASS_NAMES[1]: p}
        else:
            probs = preds[0]
            idx = int(np.argmax(probs))
            probs = probs / probs.sum()
            return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    else:
        raise ValueError("پاسخ مدل شکل غیرقابل پشتیبانی دارد: " + str(preds.shape))

# ------------------------- تابع پیش‌بینی تصویر -------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return parse_model_prediction(preds)

# ------------------------- بارگذاری مدل با Progress و غیرفعال‌سازی دکمه‌ها -------------------------
def load_model_with_progress(path):
    global model
    progress_bar['value'] = 0
    status_label.config(text="⏳ در حال بارگذاری مدل...")
    select_model_btn.config(state="disabled")
    select_image_btn.config(state="disabled")
    def task():
        nonlocal path
        try:
            # انیمیشن پر شدن نوار (فقط برای UI)
            for i in range(1, 101):
                time.sleep(0.008)
                progress_bar['value'] = i
                root.update_idletasks()
            # بارگذاری واقعی مدل (ممکن وقت بگیره)
            loaded = tf.keras.models.load_model(path)
            model_buf = loaded
            model_ready(model_buf)
        except Exception as e:
            status_label.config(text=f"❌ خطا در بارگذاری مدل: {e}")
        finally:
            select_model_btn.config(state="normal")
    threading.Thread(target=task, daemon=True).start()

def model_ready(loaded_model):
    global model
    model = loaded_model
    status_label.config(text="✅ مدل بارگذاری شد")
    play_sound_embedded()
    select_image_btn.config(state="normal")

# ------------------------- انتخاب مدل -------------------------
def open_model():
    file_path = filedialog.askopenfilename(title="انتخاب فایل مدل", filetypes=[("Keras model", "*.h5 *.keras"), ("All files","*.*")])
    if file_path:
        load_model_with_progress(file_path)

# ------------------------- انتخاب تصویر -------------------------
def open_file():
    if model is None:
        status_label.config(text="⚠️ ابتدا مدل را انتخاب کنید")
        return
    file_path = filedialog.askopenfilename(title="انتخاب تصویر", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")])
    if not file_path:
        return
    # غیرفعال کردن موقت دکمه‌ها تا پردازش تمام شود
    select_model_btn.config(state="disabled")
    select_image_btn.config(state="disabled")
    status_label.config(text="⏳ در حال پیش‌بینی...")
    def task():
        try:
            # نمایش تصویر
            img_pil = Image.open(file_path).convert("RGB").resize((320, 320))
            img_tk = ImageTk.PhotoImage(img_pil)
            image_label.config(image=img_tk)
            image_label.image = img_tk

            # پیش‌بینی
            results = predict_image(file_path)
            # نمایش متنی
            text = ""
            for cls, p in results.items():
                text += f"{cls}: {p*100:.2f}%\n"
            result_label.config(text=text)
            # رسم و انیمیشن نمودار (از طریق main thread)
            root.after(100, lambda: animate_gradient_glow_bar_chart(results))
        except Exception as e:
            status_label.config(text=f"❌ خطا در پیش‌بینی: {e}")
        finally:
            # دکمه‌ها بعد از تمام شدن انیمیشن (در تابع animate مجدداً فعال می‌شوند)
            pass
    threading.Thread(target=task, daemon=True).start()

# ------------------------- انیمیشن نمودار (گرادیان + glow + لرزش) -------------------------
def animate_gradient_glow_bar_chart(results):
    # disable buttons during animation
    select_model_btn.config(state="disabled")
    select_image_btn.config(state="disabled")
    status_label.config(text="⏳ در حال نمایش نتیجه...")

    fig.clf()
    ax = fig.add_subplot(111)
    canvas.draw()

    target_values = [v*100 for v in results.values()]
    current_values = [0.0 for _ in target_values]
    colors = [("limegreen", "darkgreen"), ("red", "darkred")]
    steps_fill = 60  # تعداد گام‌های پر شدن
    delay_ms = 15    # فاصله‌ی بروزرسانی (میلی‌ثانیه)

    # تابعی برای رسم با گرادیان و glow
    def draw_bars(vals):
        ax.cla()
        ax.set_ylim(0, 110)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Prediction Probabilities")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # draw each bar as many thin rectangles to simulate gradient + glow
        n = 40
        current_patches.clear()
        for i, val in enumerate(vals):
            for j in range(n):
                h1 = val * j / n
                h2 = val * (j+1) / n
                t = j / (n-1) if n>1 else 0
                base = interpolate_color(colors[i][0], colors[i][1], t)
                # make glow by brightening slightly for upper stripes
                glow = np.clip(np.array(base) + 0.15*(1 - t), 0, 1)
                rect = Rectangle((i-0.35, h1), 0.7, h2-h1, color=glow, linewidth=0)
                ax.add_patch(rect)
            # draw percentage label on top
            ax.text(i, val + 2, f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11, color='#222')
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(list(results.keys()), fontsize=12, fontweight='bold')
        canvas.draw()
        # update current_patches used for tooltip hit-testing
        current_patches[:] = ax.patches.copy()

    # animate fill
    def fill_step(k):
        for i in range(len(target_values)):
            current_values[i] = min(target_values[i], target_values[i] * (k / steps_fill))
        draw_bars(current_values)
        if k < steps_fill:
            root.after(delay_ms, lambda: fill_step(k+1))
        else:
            # start shake after small pause
            root.after(120, lambda: shake_bars(0))

    # shake animation
    def shake_bars(step_count, max_steps=8, magnitude=2.5):
        # oscillate heights slightly up/down
        offsets = [magnitude * np.sin(np.pi * step_count / max_steps) for _ in target_values]
        drawn = [max(0.0, current_values[i] + offsets[i]) for i in range(len(current_values))]
        draw_bars(drawn)
        if step_count < max_steps:
            root.after(60, lambda: shake_bars(step_count+1, max_steps, magnitude))
        else:
            # finish: enable tooltip and play sound and re-enable buttons
            add_tooltip_connection()
            play_sound_embedded()
            status_label.config(text="✅ آماده")
            select_model_btn.config(state="normal")
            select_image_btn.config(state="normal")

    # wrapper to call shake
    def shake_bars_wrapper(step_count):
        shake_bars(step_count)

    # start filling
    fill_step(0)

# ------------------------- رنگ بین دو رنگ (interpolate) -------------------------
def interpolate_color(c1, c2, t):
    import matplotlib.colors as mcolors
    rgb1 = np.array(mcolors.to_rgb(c1))
    rgb2 = np.array(mcolors.to_rgb(c2))
    rgb = rgb1*(1-t) + rgb2*t
    return rgb

# ------------------------- Tooltip: از اتصال matplotlib استفاده می‌کنیم -------------------------
tooltip_widget = None
tooltip_cid = None

def add_tooltip_connection():
    global tooltip_cid, tooltip_widget
    # disconnect previous if any
    if tooltip_cid is not None:
        try:
            canvas.mpl_disconnect(tooltip_cid)
        except Exception:
            pass
    # create tooltip widget (hidden)
    if tooltip_widget is None:
        tooltip_widget = tk.Label(root, text="", bg="#FFFFE0", bd=1, relief="solid", font=("Arial", 9))
    # handler uses matplotlib event with xdata,ydata
    def on_move(event):
        # event.xdata, event.ydata are None when outside axes
        if event.xdata is None or event.ydata is None:
            tooltip_widget.place_forget()
            return
        ax = event.inaxes
        if ax is None:
            tooltip_widget.place_forget()
            return
        # check which bar (by x coordinate)
        x = event.xdata
        y = event.ydata
        # bars are centered at integers 0..n-1, width ~0.7
        for i, patch in enumerate(ax.patches):
            # we created patches as vertical stripes; to detect bar area, check x near integer
            # simpler: compute distance to bar center
            center = (i)  # index corresponds to bar center
            if abs(x - center) <= 0.4:
                # find top y of that bar by reading text above (or use first patch stack)
                # We'll compute value by scanning patches belonging to that bar: patches are added in groups; approximate by comparing x0
                # Use bounding box of patches that have x0 close to center-0.35
                for p in ax.patches:
                    bx0 = p.get_x()
                    if abs(bx0 - (center - 0.35)) < 0.05:
                        # approximate height using p.get_y()+p.get_height of the top-most patch in that group
                        pass
                # simpler: read the text labels on top of bars to get shown percentage
                texts = [t for t in ax.texts if t.get_position()[0] == center]
                if texts:
                    txt = texts[0].get_text()
                else:
                    txt = ""
                # position tooltip near mouse (screen coords)
                try:
                    canvas_widget = canvas.get_tk_widget()
                    x_root = canvas_widget.winfo_rootx() + int(event.guiEvent.x)
                    y_root = canvas_widget.winfo_rooty() + int(event.guiEvent.y)
                    # place relative to root window
                    tooltip_widget.config(text=f"{CLASS_NAMES[i]}: {txt}")
                    tooltip_widget.place(x=x_root - root.winfo_rootx() + 12, y=y_root - root.winfo_rooty() + 12)
                except Exception:
                    pass
                return
        tooltip_widget.place_forget()

    tooltip_cid = canvas.mpl_connect("motion_notify_event", on_move)

# ------------------------- ساخت پنجره -------------------------
root = tk.Tk()
root.title("💉 تشخیص سینه‌پهلو از تصویر")
root.geometry("760x820")
root.configure(bg="#f8fbff")

# Title
title_lbl = tk.Label(root, text="📌 سیستم تشخیص سینه‌پهلو (Pneumonia Detector)", font=("Arial", 16, "bold"), bg="#f8fbff", fg="#222")
title_lbl.pack(pady=12)

# select model button
select_model_btn = tk.Button(root, text="🔹 انتخاب مدل", command=open_model, font=("Arial", 12, "bold"), bg="#2E86AB", fg="white", width=22)
select_model_btn.pack(pady=8)

# progress
progress_bar = ttk.Progressbar(root, orient="horizontal", length=520, mode="determinate")
progress_bar.pack(pady=6)

status_label = tk.Label(root, text="هیچ مدلی انتخاب نشده", font=("Arial", 11), bg="#f8fbff")
status_label.pack(pady=4)

# select image button
select_image_btn = tk.Button(root, text="📁 انتخاب تصویر", command=open_file, font=("Arial", 12, "bold"), bg="#28A745", fg="white", width=22, state="disabled")
select_image_btn.pack(pady=8)

# image display
image_label = tk.Label(root, bg="white", relief="solid", bd=2, width=320, height=320)
image_label.pack(pady=10)

# textual result
result_label = tk.Label(root, text="", font=("Arial", 13, "bold"), bg="#f8fbff", fg="#333", justify="center")
result_label.pack(pady=6)

# matplotlib figure embedded
fig = plt.Figure(figsize=(6,4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

# run
root.mainloop()
