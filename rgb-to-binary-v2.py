import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread("image.png")  # ganti sesuai nama file

# Konversi ke float biar perhitungan lebih akurat
img = img.astype(float)

# Ekstrak channel (ingat OpenCV: BGR, jadi kita balik ke RGB)
B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

# Konversi manual ke grayscale
gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# Konversi ke binary dengan threshold
threshold = 128
binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
binary01 = (binary / 255).astype(np.uint8)

# Tampilkan array 2d di terminal
print("Array RGB - Channel R:")
print(R)

print("\nArray RGB - Channel G:")
print(G)

print("\nArray RGB - Channel B:")
print(B)

print("\nArray Grayscale:")
print(gray)

print("\nArray Binary (0 dan 1):")
print(binary01)

# Tampilkan hasil
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale (Manual)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(binary, cmap='gray')
plt.title('Binary (Manual)')
plt.axis('off')

plt.show()
