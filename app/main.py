"""
FastAPI 백엔드 — 영상 업로드 -> 24프레임 추출 -> 특징 계산 -> 4개 실험군 모델 추론.

로컬 실행:  uvicorn app.main:app --reload --port 8000
API 문서:   http://localhost:8000/docs
"""

import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import features, inference, video, visualize

app = FastAPI(title="AI 생성 영상 탐지 API")

# 프론트엔드(GitHub Pages)에서 호출할 수 있도록 CORS 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gahye0n.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models = None  # 앱 시작 시 한 번만 로드(무거운 작업)


@app.on_event("startup")
def _load_models_on_startup():
    global _models
    _models = inference.load_models()
    print(f"모델 {len(_models)}개 로드 완료:", list(_models.keys()))


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": _models is not None and len(_models)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if _models is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        frames_rgb = video.extract_frames_from_video(tmp_path)
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
        feature_row = features.extract_all_features(frames_rgb)
        results = inference.predict_all(feature_row, _models)
        contribution = inference.contribution_breakdown(_models)
        frame_thumbs = [visualize.frame_to_base64_jpeg(f) for f in frames_rgb]
        spectrum = visualize.spectrum_panel_base64(frames_rgb[0], frames_gray=frames_gray)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "results": results,
        "contribution": contribution,
        "frames": frame_thumbs,      # 24장, base64 JPEG (8x3 그리드용)
        "spectrum": spectrum,         # {"original":..., "fft":..., "wavelet":..., "temporal_fft":..., "temporal_wavelet":...}
        "features": feature_row,     # 예측에 쓰인 모든 원본 피처 값 {"framediff_0000": 0.01, ...}
        "feature_columns_by_model": {name: m.feature_columns for name, m in _models.items()},
    }
