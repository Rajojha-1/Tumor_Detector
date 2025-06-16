# === 1. Import Dependencies ===
import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# === 2. Set Dataset Paths ===
train_dir = 'Dataset/Training'
test_dir = 'Dataset/Testing'

# === 3. Constants ===
IMAGE_SIZE = 128
BATCH_SIZE = 20
EPOCHS = 5
CLASS_LABELS = sorted(os.listdir(train_dir))

# === 4. Load and Shuffle Data Paths ===
def load_paths_labels(directory):
    paths, labels = [], []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        for image in os.listdir(label_dir):
            paths.append(os.path.join(label_dir, image))
            labels.append(label)
    return shuffle(paths, labels)

train_paths, train_labels = load_paths_labels(train_dir)

# === 5. Augmentation & Loader ===
def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return np.array(image) / 255.0

def open_images(paths):
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = img_to_array(image)
        image = augment_image(image)
        images.append(image)
    return np.array(images)

def encode_label(labels):
    return np.array([CLASS_LABELS.index(label) for label in labels])

def datagen(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = encode_label(labels[i:i + batch_size])
            yield batch_images, batch_labels

# === 6. Build Model ===
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(CLASS_LABELS), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# === 7. Train Model ===
steps_per_epoch = len(train_paths) // BATCH_SIZE
history = model.fit(
    datagen(train_paths, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS),
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch
)

# === 8. Plot Accuracy & Loss ===
plt.figure(figsize=(8, 4))
plt.grid(True)
plt.plot(history.history['sparse_categorical_accuracy'], '.g-', linewidth=2)
plt.plot(history.history['loss'], '.r-', linewidth=2)
plt.title('Training History')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

# === 9. Save the Model ===
model.save('model.h5')
print("✅ Model saved as model.h5")
