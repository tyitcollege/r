import cv2
import cv2

img = cv2.imread("img.jpg")

# original
cv2.imshow("Original", img)

# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

# crop
crop = img[50:200, 100:200]
cv2.imshow("Crop", crop)

# rotate
(h, w) = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
rot = cv2.warpAffine(img, M, (w, h))
cv2.imshow("Rotate", rot)

# resize
res = cv2.resize(img, (100,100))
cv2.imshow("Resize", res)

# blur
blur = cv2.GaussianBlur(img, (7,7), 0)
cv2.imshow("Blur", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
