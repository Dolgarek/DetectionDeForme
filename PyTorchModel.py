import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from ModelTrainer import CustomShapeDataset, data_transform

# Step 2: Data Loading and Preprocessing
# Modify your dataset so that it handles multiple labels
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    validation_split=0.2,
)

dataset = CustomShapeDataset("dataset", transform=data_transform)

batch_size = 32
image_size = (64, 64)
num_classes = len(dataset[0][1])

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)

validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
)

# Step 3: Model Definition
model = keras.Sequential([
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Add more convolutional layers as needed
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Step 4: Loss Function and Optimizer
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Step 5: Training Loop
num_epochs = 10
checkpoint = ModelCheckpoint('shape_classifier.h5', save_best_only=True)

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=num_epochs,
    callbacks=[checkpoint],
)

# Step 6: Save the model (already saved using ModelCheckpoint)
