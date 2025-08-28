import cv2
import matplotlib.pyplot as plt

# read picture
img = cv2.imread('fitgirl-10px.png')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert to binary
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
binary01 = (binary / 255).astype(int)

print("Array RGB - Channel R:")
print(img[:,:,2])

print("\nArray RGB - Channel G:")
print(img[:,:,1])

print("\nArray RGB - Channel B:")
print(img[:,:,0])

print("\nArray Grayscale:")
print(gray)

print("\nArray Binary (0 dan 1):")
print(binary01)

# show result
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 3, 3)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')

plt.show()