"""
저장된 4개 joblib 모델(§25-1에서 만든 것)을 로드하고, 특징 dict 하나로 4개 모델 전부를
한 번에 추론한다. 각 모델의 meta.json에 있는 feature_columns 순서를 그대로 따르므로,
어떤 조합(ResNet 단독/+Spatial/+Temporal/Hybrid)에 어떤 컬럼이 필요한지 이 파일에서
다시 정의할 필요가 없다 — 노트북의 FEATURE_GROUPS_USED와 항상 자동으로 일치한다.
"""

import json
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# 노트북 §19 ABLATION_ITEMS와 동일한 순서·이름 — save_model_for_web()이 만든 파일명 규칙과 맞아야 함.
ABLATION_ITEMS = [
    "ResNet 단독 (기준선)",
    "ResNet + Spatial-Level",
    "ResNet + Temporal-native",
    "ResNet + Full Artifact (=Hybrid)",
]


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")


class LoadedModel:
    __slots__ = ("name", "pipeline", "feature_columns", "threshold", "test_auc")

    def __init__(self, name, pipeline, meta):
        self.name = name
        self.pipeline = pipeline
        self.feature_columns = meta["feature_columns"]
        self.threshold = meta["threshold"]
        self.test_auc = meta.get("test_auc")


def load_models(model_dir: Path = MODEL_DIR) -> dict:
    """{조합 이름: LoadedModel} 딕셔너리를 반환. 앱 시작 시 한 번만 호출할 것(무겁다)."""
    models = {}
    for name in ABLATION_ITEMS:
        safe = _safe_name(name)
        pipeline = joblib.load(model_dir / f"{safe}.joblib")
        meta = json.loads((model_dir / f"{safe}_meta.json").read_text(encoding="utf-8"))
        models[name] = LoadedModel(name, pipeline, meta)
    return models


def predict_all(feature_row: dict, models: dict) -> dict:
    """feature_row: features.extract_all_features()가 만든 {"rgb_level_0000": ..., ...} 평탄화 dict.
    반환: {조합 이름: {"score": float, "label": "real"|"fake", "auc": float(참고용 test AUC)}}
    """
    results = {}
    for name, m in models.items():
        missing = [c for c in m.feature_columns if c not in feature_row]
        if missing:
            raise KeyError(f"[{name}] 특징 벡터에 없는 컬럼 {len(missing)}개(예: {missing[:3]}) — "
                            "features.py의 계산 로직이 학습 때와 달라졌을 가능성이 있습니다.")
        x = np.array([[feature_row[c] for c in m.feature_columns]], dtype=np.float32)
        score = float(m.pipeline.predict_proba(x)[0, 1])
        label = "fake" if score >= m.threshold else "real"
        results[name] = {"score": round(score, 4), "label": label, "auc": m.test_auc}
    return results


def contribution_breakdown(models: dict, hybrid_key: str = "ResNet + Full Artifact (=Hybrid)") -> dict:
    """Hybrid 모델의 계수 절댓값을 축(CNN/공간/시간/기타)별로 합산 — 표11과 동일한 로직.
    프론트엔드의 "판정 근거" 표시에 사용."""
    m = models[hybrid_key]
    lr = m.pipeline.named_steps.get("logisticregression")
    if lr is None:
        return {}
    coefs = np.abs(lr.coef_[0])

    TEMPORAL_PREFIXES = {"framediff", "motion", "tfft", "twav"}
    SPATIAL_PREFIXES = {"rgb_level", "rgb_dynamics", "fft_level", "fft_dynamics",
                         "wavelet_level", "wavelet_dynamics"}
    RESNET_PREFIXES = {"resnet_level", "resnet_dynamics"}

    axis_sum = {"CNN(ResNet)": 0.0, "공간(손수설계)": 0.0, "시간(손수설계)": 0.0, "기타": 0.0}
    for col, c in zip(m.feature_columns, coefs):
        prefix = col.rsplit("_", 1)[0]
        if prefix in RESNET_PREFIXES:
            axis_sum["CNN(ResNet)"] += c
        elif prefix in SPATIAL_PREFIXES:
            axis_sum["공간(손수설계)"] += c
        elif prefix in TEMPORAL_PREFIXES:
            axis_sum["시간(손수설계)"] += c
        else:
            axis_sum["기타"] += c

    total = sum(axis_sum.values()) or 1.0
    return {k: round(v / total * 100, 1) for k, v in axis_sum.items()}
