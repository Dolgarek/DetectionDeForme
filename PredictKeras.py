import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from PyTorchModel import model

classes = ['circle', 'circle_rectangle', 'circle_triangle', 'circle_rectangle_triangle', 'rectangle', 'rectangle_triangle', 'triangle']

image_path = 'img.png'  # Path to your PNG image
target_size = (64, 64)  # Same size used during training

image = load_img(image_path, target_size=target_size)
image = img_to_array(image) / 255.0  # Normalize pixel values
image = image.reshape((1,) + image.shape)  # Reshape to (1, 64, 64, 3)

predictions = model.predict(image)

predicted_shape_index = np.argmax(predictions)
predicted_shape = classes[predicted_shape_index]

print(predicted_shape)