import cv2
import mediapipe as mp
import numpy as np
import threading
import joblib

path = 'C://python//train//'
modelPath = path + 'model//'
testPath = path + 'test//'
label = ""
n_time_steps = 1
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = joblib.load(modelPath + 'model_final.sav')
cap = cv2.VideoCapture(testPath + 'test.mp4')


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


while True:

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
    cv2.imshow("Test", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
