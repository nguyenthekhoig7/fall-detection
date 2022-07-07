from fastapi import FastAPI, File, UploadFile
from starlette.responses import HTMLResponse 
import cv2
import mediapipe as mp
import numpy as np
import threading
import joblib
import time
import shutil
import json

### functions for detecting falls from a video

label = ""
n_time_steps = 1
global lm_list 
# lm_list = []
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
# model = joblib.load('C://fall-urdataset-csv//model//model_final.sav')
model = joblib.load('model_final.sav')

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
        # print(id, lm)
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
    elif results[0] == 0:
        label = "No Fall"
    else:
        label = 'Unknown'
    return label

def get_labels(videoPath: str):

    cap = cv2.VideoCapture(videoPath)
    # global lm_lists
    global label
    lm_list = []
    start_time = time.time()
    detect_results = {}
    while True:
        # print("first loop turn")
        success, img = cap.read()
        if not success:
            break
        # print("success")
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
        else:
            label = 'No skeleton detected'
        img = draw_class_on_image(label, img)
        cv2.imshow("Test", img)

        vidtime = time.time()-start_time
        print(vidtime)#, ': ', label)
        detect_results[vidtime] = label
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return detect_results

### The app
app = FastAPI()


@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /file please"}

@app.get('/file', response_class=HTMLResponse)
def upload_file():
    return '''<form method="post" enctype="multipart/form-data"> 
    <input type="file" name="upfile">      
    <input type="submit"/> 
    </form>'''
    # <input type="text" maxlength="28" name="text" value="Text Emotion to be tested"/>



@app.post('/file') #text: str = Form(...), 
def handle_file(upfile: UploadFile = File(...)):

    with open('video.mp4', 'wb') as buffer:
        shutil.copyfileobj(upfile.file, buffer)

    detect_results = get_labels('video.mp4') # a dictionary


    return {"Name of UpFile" : upfile.filename, 
            "File Content Type": upfile.content_type, 
            "Labels": json.dumps(detect_results, indent=4)}

    # return_str = "Name of UpFile: " + str(upfile.filename) + "\nFile Content Type: " + str(upfile.content_type) + "\nLabels" + json.dumps(detect_results, indent=4)
    # return return_str


