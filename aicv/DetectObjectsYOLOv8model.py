import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")

image_path = "V1.jpg"
img = cv2.imread(image_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = model(img_rgb)

annotated_img = result[0].plot()

cv2.imshow("Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
