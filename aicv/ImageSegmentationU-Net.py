import cv2
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

model = smp.Unet(
   encoder_name="resnet34",
   encoder_weights="imagenet",
   in_channels=3,
   classes=1
)

model.eval()

img = cv2.imread('V1.jpg')
img = cv2.resize(img,(128,128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.0

with torch.no_grad():
   mask = model(img_tensor)
   mask = torch.sigmoid(mask)
   mask_np = mask[0,0].numpy()

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Input Image")

plt.subplot(1,2,2)
plt.imshow(mask_np, cmap="gray")
plt.title("Predicted Mask")

plt.show()
