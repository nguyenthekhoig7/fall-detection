import cv2
import numpy as np


path = 'C://Users//bazzy//Desktop//FallDetection_Deploy//'
modelPath = path + ''
videoPath = path + ''
cap = cv2.VideoCapture(videoPath + 'video.mp4' )

img_array = []

for i in range (0,180):
    success, img = cap.read()
    (height, width, layers) = img.shape
    size = (width,height)
    img_array.append(img)

print(len(img_array))

vid_name = 'video_with_anotation.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(vid_name,fourcc, 20, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
