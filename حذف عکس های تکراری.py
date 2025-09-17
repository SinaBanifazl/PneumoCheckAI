import os
import hashlib

# تعریف مسیر پوشه تصاویر
train_dir = r"C:\Users\Sina Banifazl\Desktop\New folder (2)\chest_xray\train"

def check_duplicates(directory):
    hashes = {}
    duplicates = []
    for root, _, files in os.walk(directory):
        for fname in files:
            path = os.path.join(root, fname)
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in hashes:
                duplicates.append(path)
                print(f"Duplicate: {path} == {hashes[file_hash]}")
            else:
                hashes[file_hash] = path
    print(f"Found {len(duplicates)} duplicates")
    return duplicates

# فراخوانی تابع برای شناسایی
duplicates = check_duplicates(train_dir)

# حذف فایل‌های تکراری
if duplicates:
    print("Starting to delete duplicates...")
    for dup in duplicates:
        try:
            os.remove(dup)
            print(f"Deleted: {dup}")
        except Exception as e:
            print(f"Error deleting {dup}: {e}")
    print("Deletion completed.")
else:
    print("No duplicates to delete.")

# چک تعداد فایل‌ها بعد از حذف
print(f"Train: NORMAL={len(os.listdir(os.path.join(train_dir, 'NORMAL')))}, PNEUMONIA={len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))}")