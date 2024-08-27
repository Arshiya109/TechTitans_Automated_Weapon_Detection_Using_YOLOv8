from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = YOLO('best.pt')

# Perform detection on an image
results = model.predict(r'C:\Users\arshi\OneDrive\Documents\IIIT HYDERABAD\5b4c9e09-img2.jpg')

# Extract the image and annotations
img = results.pandas().xyxy[0]  # Get detections in pandas DataFrame format

# Print the detections
print(img)

# Display the image with annotations
image = cv2.imread(r'C:\Users\arshi\OneDrive\Documents\IIIT HYDERABAD\5b4c9e09-img2.jpg')
for _, row in img.iterrows():
    # Draw bounding boxes
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
