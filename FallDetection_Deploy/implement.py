import cv2
import mediapipe as mp
import numpy as np
import threading
import joblib
from fastapi import FastAPI, File, UploadFile
import shutil
from fastapi.responses import StreamingResponse

app = FastAPI()

# uplaod and save file
@app.post('/upload_and_save_video/')
async def upload_file(file: UploadFile = File(...)):
    with open('video.mp4', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {'file': file.filename}


path = 'C://Users//bazzy//Desktop//FallDetection_Deploy//'
modelPath = path + ''
videoPath = path + ''
label = ""
n_time_steps = 1
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = joblib.load(modelPath + 'model_final.sav')
cap = cv2.VideoCapture(videoPath + 'video.mp4' )

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (255, 0, 0)
    thickness = 5
    lineType = 3
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    results = model.predict(lm_list)
    if results[0] == 1:
        label = "Fall"
    else:
        label = "No Fall"
    return label

# save video
img_array = []
video_length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))

# while True:
for i in range(0,video_length):

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        c_lm = make_landmark_timestep(results)

        lm_list.append(c_lm)
        if len(lm_list) == n_time_steps:
            # predict
            t1 = threading.Thread(target=detect, args=(model, lm_list,))
            t1.start()
            lm_list = []

        img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)

    # show video on local desktop
    cv2.imshow("Test", img)
    img_array.append(img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

# save video with label and landmarks
(height, width, layers) = img_array[0].shape
size = (width,height)

vid_name = 'video_with_anotation.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(vid_name,fourcc, 20, size)

for i in range(len(img_array)):
     out.write(img_array[i])

out.release()

cv2.destroyAllWindows()
