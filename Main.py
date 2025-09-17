import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels
import seaborn as sns

# تنظیمات اولیه برای استفاده از GPU
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU detected and configured for use.")
else:
    print("No GPU detected. Training will use CPU.")

# متغیرهای قابل تنظیم
dataset_path = r'C:\Users\Sina Banifazl\Desktop\New folder (2)\chest_xray'  # آدرس دیتاست (با ساختار train/test/val و زیرپوشه‌های NORMAL و PNEUMONIA)
output_dir = r'C:\Users\Sina Banifazl\Desktop\جدید\output'   # آدرس پوشه خروجی
epochs = 10                           # تعداد epoch
batch_size = 64                        # اندازه بچ (اختیاری، می‌توانید تغییر دهید)
img_height = 150                       # ارتفاع تصویر
img_width = 150                        # عرض تصویر
learning_rate = 0.001                  # نرخ یادگیری

# ایجاد پوشه خروجی اگر وجود نداشته باشد
os.makedirs(output_dir, exist_ok=True)

# آماده‌سازی داده‌ها با افزایش داده
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # تقسیم برای validation اگر val جدا نباشد
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # برای طبقه‌بندی باینری (NORMAL vs PNEUMONIA)
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),  # اگر val جدا نیست، از train استفاده کنید
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # برای محاسبه دقیق confusion matrix
)

# ساخت مدل CNN بهینه (برای دقت بالا، از یک مدل ساده اما موثر استفاده می‌کنیم)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # خروجی باینری
])

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ذخیره خلاصه مدل در فایل txt
with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# آموزش مدل با early stopping برای بهینه‌سازی
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# ارزیابی مدل روی داده تست
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
print(f'Test Loss: {test_loss * 100:.2f}%')  # درصد تابع خطا

# پیش‌بینی روی داده تست
predictions = model.predict(test_generator)
y_pred = np.round(predictions).astype(int).flatten()
y_true = test_generator.classes

# محاسبه متریک‌ها
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-Score: {f1 * 100:.2f}%')

# گزارش طبقه‌بندی و ذخیره در فایل txt
report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'])
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# ماتریس confusion و ذخیره به عنوان png
cm = confusion_matrix(y_true, y_pred)
labels = unique_labels(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# نمودارهای matplotlib: accuracy و loss over epochs (نمودارهای "رگرسیون" ممکن است به معنای منحنی‌های یادگیری باشد)
plt.figure(figsize=(12, 4))

# نمودار accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# نمودار loss (تابع خطا)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'training_curves.png'))
plt.close()

# برای "نمودار رگرسیون"، اگر منظورتان ROC curve باشد (برای طبقه‌بندی)
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# ذخیره مدل اگر نیاز باشد (اختیاری)
model.save(os.path.join(output_dir, 'pneumonia_model.h5'))