import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import datetime
from db import get_all_logs, get_frames_by_log_id, create_frame, delete_frame, get_frame

router = APIRouter()
templates = Jinja2Templates(directory="templates")

VIDEO_DIR = "./static/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

@router.get("/", response_class=HTMLResponse)
async def frames(request: Request):
    return templates.TemplateResponse("frames.html", {"request": request, "videos": get_all_logs()})

@router.get("/list")
async def list_frames(request: Request, log_id: int = 0, page: int = 1, limit: int = 10):
    query = get_frames_by_log_id(log_id)
    total = query.count()
    paginated_frames = query.offset((page - 1) * limit).limit(limit).all()

    frame_data = []
    for frame in paginated_frames:
        frame_data.append({
            "id": frame.id,
            "time": frame.time.strftime('%Y-%m-%d %H:%M:%S'),
            "base64": frame.base64,
            "data": frame.data
        })
    
    return JSONResponse(content={"code": 0, "msg": "ok", "data": frame_data, "count": total})

@router.delete("/{frame_id}")
async def delete_frame(frame_id: int):
    frame = get_frame(frame_id)
    if frame:
        delete_frame(frame_id)
        return JSONResponse(content={"status": "success"})
    else:
        raise HTTPException(status_code=404, detail="Frame not found")
