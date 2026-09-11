---
title: AI 생성 영상 탐지
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 백엔드 — AI 생성 영상 탐지 API

노트북(`260808(최종일까).ipynb`) §25-1에서 저장한 4개 모델을 그대로 불러와, 업로드된 영상 하나를
24프레임 추출→특징 계산→4개 실험군(ResNet 단독/+Spatial-Level/+Temporal-native/Hybrid)으로 동시에
판정하는 FastAPI 서버입니다.

## 1. 모델 파일 복사

Colab에서 `MODEL_DIR`(`/content/drive/MyDrive/AI생성영상탐지_최종/models`) 안의 파일 8개를
이 폴더의 `models/`에 그대로 복사하세요.

```
models/
  ResNet_단독_기준선.joblib
  ResNet_단독_기준선_meta.json
  ResNet_+_Spatial-Level.joblib
  ResNet_+_Spatial-Level_meta.json
  ResNet_+_Temporal-native.joblib
  ResNet_+_Temporal-native_meta.json
  ResNet_+_Full_Artifact_Hybrid.joblib
  ResNet_+_Full_Artifact_Hybrid_meta.json
```

## 2. 시간 정렬 기준 확인 (중요)

`app/video.py`의 `TARGET_DURATION_SEC = 3.0`은 노트북 §4에서 출력된
"적용할 공통 목표 구간 길이: X.XX초" 값으로 **반드시 교체**하세요. 학습 때와 다른 값을 쓰면
프레임 추출 구간이 달라져서 Frame Difference·Optical Flow·Temporal FFT·Temporal Wavelet 값이
학습 분포와 어긋납니다.

## 3. 로컬 실행

```bash
cd web-backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

`http://localhost:8000/docs`에서 Swagger UI로 영상을 직접 업로드해 테스트할 수 있습니다.

## 4. Hugging Face Spaces 배포

1. huggingface.co에서 새 Space 생성 (SDK: **Docker**)
2. 이 `web-backend` 폴더 전체(모델 파일 포함)를 그 Space 저장소에 push
3. 빌드가 끝나면 `https://<user>-<space명>.hf.space/api/analyze`가 실제 API 주소

## API

```
GET  /api/health         -> {"status": "ok", "models_loaded": 4}
POST /api/analyze        -> multipart/form-data, key="file" (영상 파일)

응답 예시:
{
  "results": {
    "ResNet 단독 (기준선)": {"score": 0.83, "label": "fake", "auc": 0.944},
    "ResNet + Spatial-Level": {...},
    "ResNet + Temporal-native": {...},
    "ResNet + Full Artifact (=Hybrid)": {...}
  },
  "contribution": {"CNN(ResNet)": 77.1, "공간(손수설계)": 19.4, "시간(손수설계)": 3.4, "기타": 0.2}
}
```
