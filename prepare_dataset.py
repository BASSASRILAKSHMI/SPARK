from sklearn.datasets import fetch_olivetti_faces
import cv2
import os

os.makedirs("dataset", exist_ok=True)

print("Downloading Olivetti faces dataset...")
data = fetch_olivetti_faces()

print("Saving 80 images to ./dataset/...")
for i in range(80):
    img = (data.images[i] * 255).astype('uint8')
    cv2.imwrite(f"dataset/img_{i}.png", img)

print(f"Done! {len(os.listdir('dataset'))} images saved to ./dataset/")