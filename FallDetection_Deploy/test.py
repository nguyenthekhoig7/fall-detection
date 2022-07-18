import cv2
import mediapipe as mp
import numpy as np
import threading
import joblib
from fastapi import FastAPI, File, UploadFile
import shutil

path = 'C://Users//bazzy//Desktop//FallDectiontion_Deploy//'
model_path = 'C:/Users/bazzy/Desktop/FallDetection_Deploy/'
modelPath = path + ''
testPath = path + ''
label = ""
n_time_steps = 1
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

print(modelPath + 'model_final.sav')
model = joblib.load(model_path + 'model_final.sav')
