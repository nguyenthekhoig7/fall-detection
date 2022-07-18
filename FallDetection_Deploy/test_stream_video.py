from fastapi import FastAPI
from fastapi.responses import StreamingResponse

#some_file_path = "large-video-file.mp4"
some_file_path = 'C://Users//bazzy//Desktop//FallDetection_Deploy//video_with_anotation.mp4'
app = FastAPI()


@app.get("/stream_video")
def main():
    def iterfile():  #
        with open(some_file_path, mode="rb") as file_like:  #
            yield from file_like  #

    return StreamingResponse(iterfile(), media_type="video/mp4")
