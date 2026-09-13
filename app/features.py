"""
학습 노트북(260808(최종일까).ipynb)의 §8~§11 특징 추출 코드를 그대로 이식한 모듈.

절대 로직을 바꾸지 말 것 — 여기서 계산되는 숫자가 저장된 joblib 모델이 학습 때 본 것과
정확히 같은 정의를 따라야 추론이 의미를 가진다. 함수 하나하나가 노트북 셀과 1:1로 대응한다.
"""

import numpy as np
import cv2
import pywt
import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 224
N_FRAMES = 24
WAVELET_GRID = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Render 무료 인스턴스(0.1 vCPU/512MB)에서는 스레드 풀 자체가 메모리를 잡아먹으므로 1개로 제한.
torch.set_num_threads(1)
cv2.setNumThreads(1)

# ---------------------------------------------------------------------------
# 공간(Spatial) 아티팩트 — 노트북 §8
# ---------------------------------------------------------------------------

def rgb_base_feature(frame_rgb):
    x = frame_rgb.astype(np.float32) / 255.0
    return np.array([x[..., c].mean() for c in range(3)] + [x[..., c].std() for c in range(3)], dtype=np.float32)


def _radial_bands(mag, n_bands, r_max=None):
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    if r_max is None:
        rr = np.sqrt(xx ** 2 + yy ** 2)
        r_max = min(h, w)
    else:
        cy, cx = h // 2, w // 2
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr_norm = np.clip(rr / r_max, 0, 1.0)
    bands = np.zeros(n_bands, dtype=np.float32)
    for b in range(n_bands):
        lo, hi = b / n_bands, (b + 1) / n_bands
        mask = (rr_norm >= lo) & (rr_norm <= hi) if b == n_bands - 1 else (rr_norm >= lo) & (rr_norm < hi)
        bands[b] = mag[mask].mean() if mask.any() else 0.0
    return bands


def fft_base_feature(gray, n_bands=24):
    f = np.fft.fft2(gray.astype(np.float32) / 255.0)
    mag = np.log1p(np.abs(np.fft.fftshift(f)))
    h, w = mag.shape
    return _radial_bands(mag, n_bands, r_max=min(h, w) / 2.0)


def hfwavelet_base_feature(gray, wavelet="haar", levels=2, grid=WAVELET_GRID):
    g = gray.astype(np.float32) / 255.0
    feats = []
    current = g
    for _ in range(levels):
        cA, (cH, cV, cD) = pywt.dwt2(current, wavelet)
        for band in [cH, cV, cD]:
            h, w = band.shape
            for rg in np.array_split(np.arange(h), grid):
                for cg in np.array_split(np.arange(w), grid):
                    sub = band[rg[0]:rg[-1] + 1, cg[0]:cg[-1] + 1]
                    a = np.abs(sub)
                    feats += [float(a.mean()), float(a.std()), float(np.mean(sub ** 2)), float(np.percentile(a, 90))]
        current = cA
    return np.array(feats, dtype=np.float32)


def level_stats(seq):
    seq = np.asarray(seq, dtype=np.float32)
    return np.concatenate([seq.mean(axis=0), seq.std(axis=0)]).astype(np.float32)


def dynamics_stats(seq):
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) >= 2:
        delta = np.abs(np.diff(seq, axis=0))
        return np.concatenate([delta.mean(axis=0), delta.std(axis=0)]).astype(np.float32)
    d = seq.shape[1] if seq.ndim > 1 else 1
    return np.zeros(d * 2, dtype=np.float32)


def spatial_features_for_video(frames_rgb, frames_gray):
    """DCT는 학습 시 계산은 했지만 어떤 조합에도 쓰이지 않아 여기서는 아예 생략한다
    (연구설명서·논문의 'DCT 완전 제외' 결정과 동일)."""
    rgb_seq, fft_seq, wav_seq = [], [], []
    for fr, gray in zip(frames_rgb, frames_gray):
        rgb_seq.append(rgb_base_feature(fr))
        fft_seq.append(fft_base_feature(gray))
        wav_seq.append(hfwavelet_base_feature(gray))
    out = {}
    for name, seq in [("rgb", rgb_seq), ("fft", fft_seq), ("wavelet", wav_seq)]:
        out[f"{name}_level"] = level_stats(seq)
        out[f"{name}_dynamics"] = dynamics_stats(seq)
    return out


# ---------------------------------------------------------------------------
# 시간(Temporal) 아티팩트 — 노트북 §9
# ---------------------------------------------------------------------------

def frame_diff_feature(frames_gray):
    diffs = np.array([np.mean(np.abs(frames_gray[t + 1].astype(np.float32) - frames_gray[t].astype(np.float32))) / 255.0
                       for t in range(len(frames_gray) - 1)], dtype=np.float32)
    return np.array([diffs.mean(), diffs.std(), np.percentile(diffs, 10), np.percentile(diffs, 50),
                      np.percentile(diffs, 90), diffs.min(), diffs.max()], dtype=np.float32)


def optical_flow_feature(frames_gray, move_threshold_px=1.0):
    mags = []
    for t in range(len(frames_gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(frames_gray[t], frames_gray[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))
    mags = np.concatenate([m.ravel() for m in mags]) if mags else np.zeros(1, dtype=np.float32)
    p10, p50, p90 = np.percentile(mags, [10, 50, 90])
    moving_ratio = float((mags > move_threshold_px).mean())
    return np.array([mags.mean(), mags.std(), p10, p50, p90, moving_ratio, mags.max(), p90 - p10], dtype=np.float32)


def temporal_fft_feature(frames_gray, size=64):
    stack = np.stack([cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA) for g in frames_gray], axis=0).astype(np.float32) / 255.0
    stack = stack - stack.mean(axis=0, keepdims=True)
    mag = np.abs(np.fft.rfft(stack, axis=0))[1:]
    n_bins = mag.shape[0]
    feats = []
    for k in range(n_bins):
        vals = mag[k].ravel()
        feats += [vals.mean(), vals.std(), np.percentile(vals, 90)]
    return np.array(feats, dtype=np.float32)


def temporal_wavelet_feature(frames_gray, size=64, wavelet="db2"):
    stack = np.stack([cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA) for g in frames_gray], axis=0).astype(np.float32) / 255.0
    Tn = stack.shape[0]
    max_level = max(pywt.dwt_max_level(Tn, wavelet), 1)
    coeffs = pywt.wavedec(stack.reshape(Tn, -1), wavelet, axis=0, level=max_level)
    energies = [float(np.mean(d ** 2)) for d in coeffs[1:]]
    return np.array(energies, dtype=np.float32)


# ---------------------------------------------------------------------------
# CNN(ResNet18, 고정) — 노트북 §10
# ---------------------------------------------------------------------------

_resnet_weights = tvmodels.ResNet18_Weights.IMAGENET1K_V1
_resnet = tvmodels.resnet18(weights=_resnet_weights)
_resnet.fc = nn.Identity()
_resnet = _resnet.to(DEVICE).eval()
_resnet_transform = _resnet_weights.transforms()


@torch.no_grad()
def resnet_feature_for_video(frames_rgb):
    imgs = [_resnet_transform(Image.fromarray(f)) for f in frames_rgb]
    feat = _resnet(torch.stack(imgs).to(DEVICE)).cpu().numpy()
    return {"resnet_level": level_stats(feat), "resnet_dynamics": dynamics_stats(feat)}


# ---------------------------------------------------------------------------
# 전체 파이프라인 — 노트북 §11 extract_all_features()와 동일한 조립 규칙
# ---------------------------------------------------------------------------

def extract_all_features(frames_rgb):
    """frames_rgb: 길이 N_FRAMES(24)의 RGB(H,W,3) uint8 ndarray 리스트.
    반환: {"framediff_0000": ..., "rgb_level_0000": ..., ...} 형태의 평탄화된 dict.
    """
    if len(frames_rgb) != N_FRAMES:
        raise ValueError(f"프레임 수가 {N_FRAMES}가 아닙니다: {len(frames_rgb)}")
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]

    sp = spatial_features_for_video(frames_rgb, frames_gray)
    fd = frame_diff_feature(frames_gray)
    of = optical_flow_feature(frames_gray)
    tfft = temporal_fft_feature(frames_gray)
    twav = temporal_wavelet_feature(frames_gray)
    rn = resnet_feature_for_video(frames_rgb)

    all_named = {"framediff": fd, "motion": of, "tfft": tfft, "twav": twav}
    all_named.update(sp)
    all_named.update(rn)

    row = {}
    for name, vec in all_named.items():
        for i, v in enumerate(vec):
            row[f"{name}_{i:04d}"] = float(v)
    return row
