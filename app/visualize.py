"""
프론트엔드에 실제로 보여줄 이미지(프레임 썸네일, FFT/Wavelet 시각화)를 만든다.
노트북 §8-1의 _fig_spectrum_sample() 로직과 동일한 계산을 재사용한다.
"""

import base64
import colorsys
import io

import cv2
import numpy as np
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def frame_to_base64_jpeg(frame_rgb, size=160, quality=82) -> str:
    small = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
    small_bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", small_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def extract_dominant_color_hex(frames_rgb, k=4, sample_every=4) -> str:
    """업로드된 영상에서 가장 넓은 면적을 차지하는 색(k-means 최다 클러스터의 중심)을 뽑는다.
    UI 강조색·퍼센트 바·스펙트럼 컬러맵에 전부 이 색 하나를 공유해서 "그 영상만의 색"으로 보이게 한다.
    너무 어둡거나 탁한 영상이어도 강조색으로 쓸 수 있게 채도·명도 하한선을 살짝 올린다."""
    samples = []
    for f in frames_rgb[::sample_every]:
        small = cv2.resize(f, (32, 32), interpolation=cv2.INTER_AREA)
        samples.append(small.reshape(-1, 3))
    data = np.concatenate(samples, axis=0).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    r, g, b = centers[np.argmax(counts)]

    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    s = max(s, 0.45)
    v = max(min(v, 0.92), 0.55)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def _fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, dpi=140, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _quad_composite_base64(cA, cH, cV, cD, figsize=(4.2, 4.2)) -> str:
    """근사(LL, 좌상) / 수평(LH, 우상) / 수직(HL, 좌하) / 대각(HH, 우하) 4개 부대역을
    하나의 정사각 이미지에 2x2로 배치한다."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    panels = [
        (axes[0, 0], cA, "gray"),
        (axes[0, 1], np.abs(cH), "viridis"),
        (axes[1, 0], np.abs(cV), "viridis"),
        (axes[1, 1], np.abs(cD), "viridis"),
    ]
    for ax, data, cmap in panels:
        ax.imshow(data, cmap=cmap)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0, right=1, top=1, bottom=0)
    return _fig_to_base64_png(fig)


def wavelet_decomposition_base64(gray, wavelet="haar", levels=2) -> dict:
    """features.hfwavelet_base_feature와 완전히 동일한 반복(2-level Haar DWT)을 그대로 따라가며,
    각 레벨에서 실제로 계산에 쓰이는 4개 부대역(근사 LL / 수평 LH / 수직 HL / 대각 HH)을
    2x2 사분할 이미지 한 장으로 합쳐 보여준다.
    "최종 결과"는 특징 추출에 실제로 들어가는 레벨1 고주파 합성(|H|+|V|+|D|)이다."""
    current = gray.astype(np.float32) / 255.0
    level_images = []
    final_b64 = None
    for lv in range(1, levels + 1):
        cA, (cH, cV, cD) = pywt.dwt2(current, wavelet)
        if lv == 1:
            composite = np.abs(cH) + np.abs(cV) + np.abs(cD)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(composite, cmap="viridis"); ax.axis("off")
            final_b64 = _fig_to_base64_png(fig)
        level_images.append({"level": lv, "quad": _quad_composite_base64(cA, cH, cV, cD)})
        current = cA
    return {"final": final_b64, "levels": level_images}


_DARK_TEXT = "#d8d8dc"
_DARK_SPINE = "#4a4a50"


def _style_for_dark_ui(ax):
    """다크 UI 카드 위에 얹히는 투명 배경 그림이므로, 텍스트·눈금·테두리를 밝은 톤으로 맞춘다."""
    ax.set_facecolor("none")
    ax.tick_params(colors=_DARK_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_DARK_TEXT)
    ax.yaxis.label.set_color(_DARK_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_DARK_SPINE)


def temporal_fft_spectrum_base64(frames_gray, accent_hex, size=64) -> str:
    """features.temporal_fft_feature와 동일한 계산(시간축 FFT)으로, 시간 주파수 bin별 평균 크기를
    막대그래프(스펙트럼)로 그린다. 한글 폰트가 없는 matplotlib 환경이라 축 라벨은 영문으로 둔다."""
    stack = np.stack([cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA) for g in frames_gray], axis=0).astype(np.float32) / 255.0
    stack = stack - stack.mean(axis=0, keepdims=True)
    mag = np.abs(np.fft.rfft(stack, axis=0))[1:]  # (n_bins, size, size), DC 성분 제외
    bin_mean = mag.reshape(mag.shape[0], -1).mean(axis=1)

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.bar(np.arange(1, len(bin_mean) + 1), bin_mean, color=accent_hex)
    ax.set_xlabel("temporal frequency bin", fontsize=9)
    ax.set_ylabel("mean magnitude", fontsize=9)
    _style_for_dark_ui(ax)
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def temporal_wavelet_spectrum_base64(frames_gray, accent_hex, size=64, wavelet="db2") -> str:
    """features.temporal_wavelet_feature와 동일한 계산(시간축 Wavelet)으로, 분해 레벨별 에너지를
    막대그래프(스펙트럼)로 그린다."""
    stack = np.stack([cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA) for g in frames_gray], axis=0).astype(np.float32) / 255.0
    T = stack.shape[0]
    max_level = max(pywt.dwt_max_level(T, wavelet), 1)
    coeffs = pywt.wavedec(stack.reshape(T, -1), wavelet, axis=0, level=max_level)
    energies = [float(np.mean(d ** 2)) for d in coeffs[1:]]

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.bar([f"L{i + 1}" for i in range(len(energies))], energies, color=accent_hex)
    ax.set_xlabel("decomposition level", fontsize=9)
    ax.set_ylabel("energy", fontsize=9)
    _style_for_dark_ui(ax)
    fig.tight_layout()
    return _fig_to_base64_png(fig)


def spectrum_panel_base64(frame_rgb, accent_hex, frames_gray=None) -> dict:
    """노트북 §8-1과 동일한 정의 — 원본 프레임 / FFT 파워 스펙트럼 / Wavelet(최종 결과 + 대역별 분해)에 더해
    frames_gray(24프레임 전체)가 주어지면 시간축 FFT/Wavelet 스펙트럼도 함께 반환한다.
    FFT/Wavelet 이미지는 원래 컬러맵(magma/viridis)을 유지하고, accent_hex(영상에서 뽑은 대표색)는
    시간축 스펙트럼 막대그래프(Temporal FFT/Wavelet)에만 적용한다."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # 원본 프레임
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    ax1.imshow(frame_rgb); ax1.axis("off")
    original_b64 = _fig_to_base64_png(fig1)

    # FFT 파워 스펙트럼(log) — features.fft_base_feature와 동일한 정의
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) / 255.0))
    mag = np.log1p(np.abs(f))
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.imshow(mag, cmap="magma"); ax2.axis("off")
    fft_b64 = _fig_to_base64_png(fig2)

    # Wavelet — features.hfwavelet_base_feature와 동일한 2-level Haar DWT, 최종 결과 + 대역별 분해 모두 포함
    wavelet = wavelet_decomposition_base64(gray)

    result = {"original": original_b64, "fft": fft_b64, "wavelet": wavelet}
    if frames_gray is not None:
        result["temporal_fft"] = temporal_fft_spectrum_base64(frames_gray, accent_hex)
        result["temporal_wavelet"] = temporal_wavelet_spectrum_base64(frames_gray, accent_hex)
    return result
