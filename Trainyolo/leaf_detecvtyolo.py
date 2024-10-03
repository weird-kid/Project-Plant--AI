import os
from ultralytics import YOLO
# Path to your saved model
model_path = 'train/weights/best.pt'

# Load the model
model = YOLO(model_path)

results = model.predict(source='test')

import matplotlib.pyplot as plt

for result in results:
    num_objects = len(result.boxes)  # Count objects in the current image
    print("Number of leaves:", num_objects) # Removed extra space before print
    img = result.plot()  # Get the plotted image from the result object
    plt.imshow(img)  # Display the image using Matplotlib
    plt.axis('off')  # Hide axes
    plt.show()  # Show the plot (waits for user to close the window)