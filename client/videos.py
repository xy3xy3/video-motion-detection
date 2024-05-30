import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import datetime

router = APIRouter()
templates = Jinja2Templates(directory="templates")

VIDEO_DIR = "./static/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

class RenameRequest(BaseModel):
    oldName: str
    newName: str

@router.get("/", response_class=HTMLResponse)
async def videos(request: Request):
    return templates.TemplateResponse("videos.html", {"request": request})

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(VIDEO_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return JSONResponse(content={"status": "success", "filename": file.filename})

@router.get("/list")
async def list_videos(request: Request):
    # 从请求中获取分页参数
    page = int(request.query_params.get("page", 1))
    limit = int(request.query_params.get("limit", 10))

    # 计算分页的起始索引
    start_index = (page - 1) * limit
    end_index = page * limit

    videos = []
    i = 1
    for filename in os.listdir(VIDEO_DIR):
        if start_index <= len(videos) < end_index:
            file_path = os.path.join(VIDEO_DIR, filename)
            size = os.path.getsize(file_path)
            creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
            videos.append({
                "id": i,
                "name": filename,
                "size": f"{size / (1024 * 1024):.2f} MB",
                "creation_time": creation_time
            })
            i += 1
        else:
            break

    return JSONResponse(content={"code": 0,"msg":"ok", "data": videos, "count": len(videos)})
@router.delete("/{filename}")
async def delete_video(filename: str):
    file_path = os.path.join(VIDEO_DIR, filename)
    if (os.path.exists(file_path)):
        os.remove(file_path)
        return JSONResponse(content={"status": "success"})
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@router.put("/rename")
async def rename_video(request: RenameRequest):
    old_path = os.path.join(VIDEO_DIR, request.oldName)
    new_path = os.path.join(VIDEO_DIR, request.newName)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        return JSONResponse(content={"status": "success"})
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@router.get("/play/{filename}")
async def play_video(filename: str):
    file_path = os.path.join(VIDEO_DIR, filename)
    if (os.path.exists(file_path)):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Video not found")
