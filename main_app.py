import cv2
import numpy as np
import os

if __name__ == "__main__":
    image_dir = "D:/Pictures/Video F/Predicted/Predicted"
    img_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    video = cv2.VideoWriter('D:/Pictures/Video F/Predicted/video1.avi', -1, 20, (800, 800))
    for i in range(len(img_names)):
        img = cv2.imread(f"{image_dir}/{img_names[i]}")
        video.write(img)
    video.release()