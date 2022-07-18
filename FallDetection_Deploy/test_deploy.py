# put everything together
import asyncio
import cv2
import mediapipe as mp
import numpy as np
import threading
import joblib
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import shutil
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get('/')
async def main_page():
    return {'welcome to Human Fall Detection program'}

# uplaod and save file
@app.post('/upload_and_save_video/')
async def upload_file(file: UploadFile = File(...)):
    with open('video.mp4', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {'file': file.filename}

# process video by call and run the script 'implement.py'
@app.get("/process_video/")
async def subprocess_test(background_tasks: BackgroundTasks):
  background_tasks.add_task(run_subprocess)

async def run_subprocess(): # run implement.py from cmd promt
  proc = await asyncio.create_subprocess_shell(
    'python implement.py',
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
  )

  stdout, stderr = await proc.communicate()

  if stderr:
    print(stderr)
  else:
    print(stdout)

    return {'processing video'}

# stream video
path = 'C://Users//bazzy//Desktop//FallDetection_Deploy//'
videoPath = path + ''
@app.get("/stream_video")
def stream_video():
    def iterfile():  #
        with open(videoPath + 'video_with_anotation.mp4', mode="rb") as file_like:  #
            yield from file_like  #

    return StreamingResponse(iterfile(), media_type="video/mp4")
