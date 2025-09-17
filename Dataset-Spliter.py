import os
import shutil
import random
import customtkinter as ctk
from tkinter import filedialog

# ظاهر
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# متغیرهای عمومی
selected_folder = ""
images = []

# تابع تقسیم عکس‌ها
def split_dataset(status_label, progress_bar, count_label, start_button):
    total = len(images)
    if total == 0:
        status_label.configure(text="❌ No valid image files found", text_color="red")
        return

    random.shuffle(images)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    output_dirs = {
        "train": os.path.join(selected_folder, "train"),
        "val": os.path.join(selected_folder, "val"),
        "test": os.path.join(selected_folder, "test"),
    }

    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    try:
        for idx, img in enumerate(images):
            src = os.path.join(selected_folder, img)

            if idx < train_size:
                dst = os.path.join(output_dirs["train"], img)
            elif idx < train_size + val_size:
                dst = os.path.join(output_dirs["val"], img)
            else:
                dst = os.path.join(output_dirs["test"], img)

            shutil.copy2(src, dst)

            # پیشرفت
            progress = (idx + 1) / total
            progress_bar.set(progress)
            status_label.configure(text=f"Processing image {idx + 1} of {total}...")
            status_label.update()
            progress_bar.update()

        status_label.configure(text=f"✅ Successfully split {total} images", text_color="green")
        start_button.configure(state="disabled")

    except Exception as e:
        for d in output_dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        status_label.configure(text=f"❌ Error: {e}", text_color="red")

# انتخاب پوشه
def select_folder(status_label, progress_bar, count_label, start_button):
    global selected_folder, images
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        selected_folder = folder_selected
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        images = [f for f in os.listdir(selected_folder)
                  if os.path.isfile(os.path.join(selected_folder, f)) and f.lower().endswith(image_extensions)]

        count_label.configure(text=f"Total images: {len(images)}")
        progress_bar.set(0)
        status_label.configure(text="")
        if len(images) > 0:
            start_button.configure(state="normal")
        else:
            start_button.configure(state="disabled")

# رابط گرافیکی
app = ctk.CTk()
app.title("Image Dataset Splitter")
app.geometry("500x330")
app.resizable(False, False)

title_label = ctk.CTkLabel(app, text="Split dataset into train / val / test", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=(20, 10))

select_button = ctk.CTkButton(app, text="📁 Select Image Folder", command=lambda: select_folder(status_label, progress_bar, count_label, start_button))
select_button.pack(pady=10)

progress_bar = ctk.CTkProgressBar(app, width=400)
progress_bar.set(0)
progress_bar.pack(pady=(10, 5))

count_label = ctk.CTkLabel(app, text="Total images: -")
count_label.pack(pady=(0, 5))

start_button = ctk.CTkButton(app, text="🚀 Start Splitting", command=lambda: split_dataset(status_label, progress_bar, count_label, start_button))
start_button.pack(pady=(10, 5))
start_button.configure(state="disabled")

status_label = ctk.CTkLabel(app, text="")
status_label.pack(pady=(10, 5))

app.mainloop()
