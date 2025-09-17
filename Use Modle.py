# -*- coding: utf-8 -*-
"""
Pneumonia Detector – Dark Only + True Glass (Acrylic/Mica) + Menus + Hover White + Social Media Buttons
"""

import os, io, threading, time, ctypes, webbrowser
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =================== Design (Dark Only) ===================
COLORS = {
    "BG": "#000000",
    "FG": "#FFFFFF",
    "PRIMARY": "#5682B1",
    "SECONDARY": "#739EC9",
    "ACCENT": "#7CFC00",
    "CARD": "#121212",
    "WARN": "#F44336",
    "OK": "#22c55e",
    "BTN_BG": "#1a1a1a",
    "BTN_FG": "#FFFFFF",
    "BTN_HOVER_BG": "#FFFFFF",
    "BTN_HOVER_FG": "#000000",
}
SP = dict(xs=4, sm=8, md=12, lg=16, xl=24)

# =================== Fonts ===================
def pick_font():
    try:
        fams = tkfont.families()
    except tk.TclError:
        return "Arial"
    for f in ["Vazirmatn", "IRANSans", "Vazir", "Segoe UI", "Arial"]:
        if f in fams:
            return f
    return "Arial"

# =================== ML ===================
IMG_SIZE = (150, 150)  # تغییر به اندازه تصویر مدل اصلی (150x150)
CLASS_NAMES = ["Normal", "Pneumonia"]
model = None
THRESHOLD = 0.5

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        arr = image.img_to_array(img)
        arr = arr / 255.0
        arr = np.expand_dims(arr, 0)
        preds = model.predict(arr)
        print(f"خروجی خام مدل: {preds}")  # چاپ خروجی خام
        p = float(preds[0][0])
        return {"Normal": 1 - p, "Pneumonia": p}
    except Exception as e:
        raise Exception(f"خطا در پیش‌بینی تصویر: {str(e)}")

# =================== Windows Glass (True Blur) ===================
def enable_true_glass(hwnd):
    try:
        DWMWA_SYSTEMBACKDROP = 38
        DWMSBT_MAINWINDOW = 2
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            ctypes.wintypes.DWORD(DWMWA_SYSTEMBACKDROP),
            ctypes.byref(ctypes.wintypes.DWORD(DWMSBT_MAINWINDOW)),
            ctypes.sizeof(ctypes.wintypes.DWORD)
        )
        return
    except Exception:
        pass

    class ACCENTPOLICY(ctypes.Structure):
        _fields_ = [
            ("AccentState", ctypes.c_int),
            ("AccentFlags", ctypes.c_int),
            ("GradientColor", ctypes.c_uint32),
            ("AnimationId", ctypes.c_int)
        ]
    class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
        _fields_ = [
            ("Attribute", ctypes.c_int),
            ("Data", ctypes.c_void_p),
            ("SizeOfData", ctypes.c_size_t)
        ]
    WCA_ACCENT_POLICY = 19
    ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
    a = 0x90
    r, g, b = (17, 17, 17)
    gradient = (a<<24) | (b<<16) | (g<<8) | r
    policy = ACCENTPOLICY(ACCENT_ENABLE_ACRYLICBLURBEHIND, 0, gradient, 0)
    data = WINDOWCOMPOSITIONATTRIBDATA(
        WCA_ACCENT_POLICY,
        ctypes.byref(policy),
        ctypes.sizeof(policy)
    )
    try:
        set_wca = ctypes.windll.user32.SetWindowCompositionAttribute
        set_wca(ctypes.wintypes.HWND(hwnd), ctypes.byref(data))
    except Exception:
        pass

def get_hwnd_from_tk(root):
    root.update_idletasks()
    return int(root.frame(), 16) if hasattr(root, "frame") else root.winfo_id()

# =================== Donut Chart ===================
def render_donut(results, dpi=120, size=(4.0, 4.0)):
    labels = list(results.keys())
    vals = [p * 100 for p in results.values()]
    colors = [COLORS["PRIMARY"], COLORS["SECONDARY"]]
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    fig.patch.set_facecolor(COLORS["BG"])
    ax.set_facecolor(COLORS["BG"])
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=colors, pctdistance=0.82,
           textprops={"color": COLORS["FG"], "weight": "bold", "fontsize": 11})
    ax.add_artist(plt.Circle((0, 0), 0.68, fc=COLORS["BG"]))
    ax.axis("equal")
    ax.set_title("📊 توزیع پیش‌بینی", fontsize=12, color=COLORS["FG"], weight="bold")
    fig.tight_layout()
    return fig

# =================== App ===================
class App:
    def __init__(self, root):
        self.root = root
        self.fontfam = pick_font()
        self.fonts = dict(title=(self.fontfam, 16, "bold"),
                          label=(self.fontfam, 12, "bold"),
                          button=(self.fontfam, 13, "bold"),
                          social=(self.fontfam, 11, "bold"))
        self.chart_canvas = None
        self.chart_fig = None
        self.current_img = None
        self.preview_label = None
        self.last_prediction = None

        self.setup_window()
        self.build_menu()
        self.build_ui()
        self.style_widgets()
        self.enable_glass()

        self.set_status("🚀 سیستم آماده است — لطفاً مدل و سپس تصویر خود را انتخاب کنید.", OK=True)

        self.bind_shortcuts()

    def setup_window(self):
        self.root.title("🩺 تشخیص سینه‌پهلو | Pro Dark + Glass")
        self.root.geometry("980x900")
        self.root.minsize(820, 820)
        self.root.configure(bg=COLORS["BG"],
                            highlightbackground=COLORS["ACCENT"],
                            highlightcolor=COLORS["ACCENT"],
                            highlightthickness=4)

    def enable_glass(self):
        try:
            hwnd = get_hwnd_from_tk(self.root)
            enable_true_glass(hwnd)
        except Exception:
            pass

    def build_menu(self):
        menubar = tk.Menu(self.root, tearoff=0, background=COLORS["BG"], foreground=COLORS["FG"])
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="📁 Open Image…\tCtrl+O", command=self.open_image)
        filem.add_command(label="🧠 Open Model…\tCtrl+M", command=self.open_model)
        filem.add_separator()
        filem.add_command(label="💾 Export Chart as PNG…", command=self.export_chart)
        filem.add_separator()
        filem.add_command(label="❌ Exit\tCtrl+Q", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filem)

        modelm = tk.Menu(menubar, tearoff=0)
        modelm.add_command(label="Load Model…", command=self.open_model)
        modelm.add_command(label="Model Info", command=self.show_model_info)
        menubar.add_cascade(label="Model", menu=modelm)

        viewm = tk.Menu(menubar, tearoff=0)
        viewm.add_command(label="Reset Layout", command=self.reset_layout)
        menubar.add_cascade(label="View", menu=viewm)

        toolsm = tk.Menu(menubar, tearoff=0)
        toolsm.add_command(label="🗂️ Batch Predict (Placeholder)", command=lambda: messagebox.showinfo("Batch", "اسکلت آماده است."))
        menubar.add_cascade(label="Tools", menu=toolsm)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="Shortcuts", command=lambda: messagebox.showinfo("Shortcuts", "Ctrl+O, Ctrl+M, Ctrl+Q"))
        helpm.add_command(label="About", command=lambda: messagebox.showinfo("About", "Dark + True Glass + Hover White\nMade with ❤️"))
        menubar.add_cascade(label="Help", menu=helpm)
        self.root.config(menu=menubar)

    def glass_card(self, parent):
        f = tk.Frame(parent, bg=COLORS["CARD"], bd=0, highlightbackground="#FFFFFF", highlightthickness=1)
        return f

    def build_ui(self):
        pad = SP["lg"]
        self.canvas = tk.Canvas(self.root, bg=COLORS["BG"], bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.container = tk.Frame(self.canvas, bg=COLORS["BG"], bd=0)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        self.canvas.configure(scrollregion=(0, 0, self.root.winfo_width(), 500))
        self.container.bind("<Configure>", self.update_scrollregion)
        self.canvas.bind("<Configure>", self.update_canvas_width)

        self.header = self.glass_card(self.container)
        self.header.pack(fill="x", pady=(0, pad))
        self.title_lbl = tk.Label(self.header, text="🧬 تشخیص سینه‌پهلو از تصویر", font=self.fonts["title"], bg=COLORS["CARD"], fg=COLORS["FG"])
        self.title_lbl.pack(expand=True, anchor="center", padx=SP["md"], pady=(SP["md"], SP["sm"]))

        social_frame = tk.Frame(self.header, bg=COLORS["CARD"])
        social_frame.pack(fill="x", padx=SP["md"], pady=(0, SP["sm"]))
        social_buttons = [
            ("✈️ تلگرام", "https://t.me/SinaBanifazl"),
            ("📸 اینستاگرام", "https://www.instagram.com/SinaBanifazl"),
            ("🛠️ گیت‌هاب", "https://github.com/SinaBanifazl")
        ]
        for text, url in social_buttons:
            btn = tk.Button(social_frame, text=text, font=self.fonts["social"],
                            bg=COLORS["BTN_BG"], fg=COLORS["BTN_FG"], bd=0, relief="flat",
                            padx=12, pady=6, cursor="hand2",
                            command=lambda u=url: webbrowser.open(u))
            btn.pack(side="left", padx=SP["xs"])
            if not hasattr(self, "social_buttons"):
                self.social_buttons = []
            self.social_buttons.append(btn)

        self.controls = self.glass_card(self.container)
        self.controls.pack(fill="x", pady=(0, pad))
        row1 = tk.Frame(self.controls, bg=COLORS["CARD"])
        row1.pack(fill="x", padx=SP["md"], pady=(SP["md"], SP["sm"]))
        self.model_btn = tk.Button(row1, text="🧠 انتخاب مدل", command=self.open_model, font=self.fonts["button"],
                                   bg=COLORS["BTN_BG"], fg=COLORS["BTN_FG"], bd=0, relief="flat", padx=18, pady=8, cursor="hand2")
        self.model_btn.pack(side="left", padx=(0, SP["md"]))
        self.image_btn = tk.Button(row1, text="🖼️ انتخاب تصویر", command=self.open_image, font=self.fonts["button"],
                                   state="disabled", bg=COLORS["BTN_BG"], fg=COLORS["BTN_FG"], bd=0, relief="flat", padx=18, pady=8, cursor="hand2")
        self.image_btn.pack(side="left")

        row2 = tk.Frame(self.controls, bg=COLORS["CARD"])
        row2.pack(fill="x", padx=SP["md"], pady=(SP["xs"], SP["md"]))
        self.progress = ttk.Progressbar(row2, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, SP["md"]))
        self.status_label = tk.Label(row2, text="", font=self.fonts["label"], bg=COLORS["CARD"], fg=COLORS["OK"])
        self.status_label.pack(side="left")

        self.preview = self.glass_card(self.container)
        self.preview.pack(fill="x", pady=(0, pad))
        tk.Label(self.preview, text="🔍 پیش‌نمایش تصویر", font=self.fonts["label"], bg=COLORS["CARD"], fg=COLORS["FG"]).pack(anchor="w", padx=SP["md"], pady=(SP["md"], SP["sm"]))
        self.preview_label = tk.Label(self.preview, bg=COLORS["CARD"])
        self.preview_label.pack(padx=SP["md"], pady=(0, SP["md"]))

        self.result = self.glass_card(self.container)
        self.result.pack(fill="x", pady=(0, pad))
        tk.Label(self.result, text="📣 نتیجه پیش‌بینی", font=self.fonts["label"], bg=COLORS["CARD"], fg=COLORS["FG"]).pack(anchor="w", padx=SP["md"], pady=(SP["md"], SP["sm"]))
        self.result_label = tk.Label(self.result, text="—", font=self.fonts["label"], bg=COLORS["CARD"], fg=COLORS["FG"], anchor="e", justify="right")
        self.result_label.pack(fill="x", padx=SP["md"], pady=(0, SP["md"]))

        self.chart = self.glass_card(self.container)
        self.chart.pack(fill="both", expand=True)
        tk.Label(self.chart, text="نمودار دوناتی احتمال‌ها", font=self.fonts["label"], bg=COLORS["CARD"], fg=COLORS["FG"]).pack(anchor="w", padx=SP["md"], pady=(SP["md"], SP["sm"]))
        self.chart_holder = tk.Frame(self.chart, bg=COLORS["CARD"])
        self.chart_holder.pack(fill="both", expand=True, padx=SP["md"], pady=(0, SP["md"]))

    def update_scrollregion(self, event=None):
        self.canvas.update_idletasks()
        width = self.root.winfo_width() - self.scrollbar.winfo_width()
        height = max(self.container.winfo_reqheight(), 500)
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def update_canvas_width(self, event=None):
        self.canvas.update_idletasks()
        width = self.root.winfo_width() - self.scrollbar.winfo_width()
        self.canvas.itemconfigure(self.canvas_frame, width=width)
        self.update_scrollregion()

    def style_widgets(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except:
            pass
        s.configure("TProgressbar", thickness=10, troughcolor=COLORS["CARD"], background=COLORS["SECONDARY"])

        def add_hover(b):
            def on_enter(e):
                b.configure(bg=COLORS["BTN_HOVER_BG"], fg=COLORS["BTN_HOVER_FG"])
            def on_leave(e):
                b.configure(bg=COLORS["BTN_BG"], fg=COLORS["BTN_FG"])
            b.bind("<Enter>", on_enter)
            b.bind("<Leave>", on_leave)
        for b in [self.model_btn, self.image_btn] + (getattr(self, "social_buttons", [])):
            add_hover(b)

    def set_status(self, msg, WARN=False, OK=False):
        color = COLORS["WARN"] if WARN else COLORS["OK"] if OK else COLORS["FG"]
        self.status_label.configure(text=msg, fg=color)

    def open_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5 *.keras")])
        if not path:
            return
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        self.set_status("⏳ در حال بارگذاری مدل...", OK=False)
        self.model_btn.configure(state="disabled")
        self.image_btn.configure(state="disabled")
        def task():
            try:
                global model
                tf.keras.backend.clear_session()
                loaded = tf.keras.models.load_model(path, compile=False)  # بدون کامپایل برای انعطاف‌پذیری
                # بررسی ورودی مدل
                input_shape = loaded.input_shape
                if input_shape[1:3] != IMG_SIZE:
                    raise ValueError(f"مدل انتظار ورودی {input_shape[1:3]} دارد، اما IMG_SIZE={IMG_SIZE} تنظیم شده است.")
                # بررسی خروجی مدل
                output_shape = loaded.output_shape[-1]
                if output_shape != 1:
                    raise ValueError(f"مدل باید خروجی باینری داشته باشد (sigmoid)، اما خروجی {output_shape} بعدی دارد.")
                # تلاش برای بارگذاری class_names در کنار فایل مدل
                try:
                    import json
                    base_dir = os.path.dirname(path)
                    cn_path = os.path.join(base_dir, "class_names.json")
                    th_path = os.path.join(base_dir, "threshold.json")
                    if os.path.exists(cn_path):
                        with open(cn_path, "r", encoding="utf-8") as jf:
                            names = json.load(jf)
                        if isinstance(names, list) and len(names) >= 2:
                            mapped = [str(n).title() for n in names]
                            if len(mapped) == 2:
                                global CLASS_NAMES
                                CLASS_NAMES = mapped
                    if os.path.exists(th_path):
                        with open(th_path, "r", encoding="utf-8") as jf:
                            th_obj = json.load(jf)
                        t = th_obj.get("threshold")
                        if isinstance(t, (int, float)) and 0.0 <= float(t) <= 1.0:
                            global THRESHOLD
                            THRESHOLD = float(t)
                except Exception as ex:
                    print(f"هشدار: خطا در بارگذاری فایل‌های متادیتا: {ex}")
                time.sleep(0.2)
                self.root.after(0, lambda: self._model_loaded(loaded))
            except Exception as ex:
                error_msg = f"❌ خطا در بارگذاری مدل: {str(ex)}"
                self.root.after(0, lambda: self.set_status(error_msg, WARN=True))
                print(error_msg)
            finally:
                self.root.after(0, lambda: (self.progress.stop(), self.progress.configure(mode="determinate", value=0),
                                            self.model_btn.configure(state="normal")))
        threading.Thread(target=task, daemon=True).start()

    def _model_loaded(self, loaded):
        global model
        model = loaded
        self.set_status("✅ مدل با موفقیت بارگذاری شد — لطفاً تصویر خود را انتخاب کنید.", OK=True)
        self.image_btn.configure(state="normal")

    def reset_ui(self):
        if self.preview_label:
            self.preview_label.configure(image='')
            self.preview_label.image = None
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
            self.chart_canvas = None
        if self.chart_fig:
            plt.close(self.chart_fig)
            self.chart_fig = None
        self.result_label.configure(text="—")
        self.last_prediction = None
        self.set_status("🚀 تصویر جدید انتخاب کنید.", OK=True)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        if model is None:
            self.set_status("⚠️ ابتدا مدل را انتخاب کنید.", WARN=True)
            return
        self.reset_ui()
        self.current_img = path
        try:
            img = Image.open(path)
            img = ImageOps.fit(img, (460, 460), Image.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=tkimg)
            self.preview_label.image = tkimg
            self.run_prediction(path)
        except Exception as e:
            self.set_status(f"❌ خطا در بارگذاری تصویر: {e}", WARN=True)

    def run_prediction(self, path):
        def task():
            try:
                self.root.after(0, lambda: self.progress.configure(mode="indeterminate"))
                self.root.after(0, lambda: self.progress.start(10))
                self.root.after(0, lambda: self.set_status("⏳ در حال پردازش تصویر...", OK=False))
                res = predict_image(path)
                self.last_prediction = res
                text = "\n".join([f"{k}: {v*100:.2f}%" for k, v in res.items()])
                self.root.after(0, lambda: self.result_label.configure(text=text))
                self.root.after(0, lambda: self.draw_chart(res))
                pneu = res.get("Pneumonia", 0.0)
                thr = THRESHOLD
                self.root.after(0, lambda: self.set_status("🚨 احتمال پنومونی بالاست" if pneu >= thr else "✅ احتمال پنومونی پایین",
                                                           WARN=pneu >= thr, OK=pneu < thr))
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f"❌ خطا در پیش‌بینی: {e}", WARN=True))
            finally:
                self.root.after(0, lambda: (self.progress.stop(), self.progress.configure(mode="determinate", value=0)))
        threading.Thread(target=task, daemon=True).start()

    def draw_chart(self, results):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
            self.chart_canvas = None
        if self.chart_fig:
            plt.close(self.chart_fig)
            self.chart_fig = None
        self.chart_fig = render_donut(results)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=self.chart_holder)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_chart(self):
        if not self.chart_fig:
            messagebox.showinfo("Export", "ابتدا پیش‌بینی انجام دهید.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                          filetypes=[("PNG Image", "*.png")],
                                          initialfile=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        if not path:
            return
        self.chart_fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=COLORS["BG"])
        self.set_status("🖼️ نمودار ذخیره شد.", OK=True)

    def show_model_info(self):
        global model
        if model is None:
            messagebox.showinfo("Model Info", "هیچ مدلی بارگذاری نشده.")
            return
        try:
            params = model.count_params()
            input_shape = model.input_shape
            output_shape = model.output_shape
            info = f"تعداد پارامترها: {params:,}\nورودی مدل: {input_shape}\nخروجی مدل: {output_shape}"
            messagebox.showinfo("Model Info", info)
        except Exception as e:
            messagebox.showerror("Model Info", f"خطا:\n{e}")

    def reset_layout(self):
        self.canvas.delete(self.canvas_frame)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.container, anchor="nw")
        self.update_scrollregion()

    def bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-m>", lambda e: self.open_model())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

# =================== Run ===================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()