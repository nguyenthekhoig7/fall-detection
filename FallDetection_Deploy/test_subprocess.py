import asyncio
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

@app.get("/process_video/")
async def subprocess_test(background_tasks: BackgroundTasks):
  background_tasks.add_task(run_subprocess)

# run a sub process from cmd prompt
async def run_subprocess():
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
