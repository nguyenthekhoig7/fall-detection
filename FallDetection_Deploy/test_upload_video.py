from fastapi import FastAPI, File, UploadFile
app = FastAPI()

# upload file
@app.post('/upload_video/')
async def upload_file(file: UploadFile = File(...)):
    return {'file': file.filename}

# uplaod and save file
import shutil
@app.post('/upload_and_save_video/')
async def upload_file(file: UploadFile = File(...)):
    with open('video.mp4', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {'file': file.filename}

# stream video
