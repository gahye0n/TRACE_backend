"""
노트북 §5(24프레임 추출)의 compute_window / extract_uniform_frames_seq /
center_crop_square_resize를 그대로 이식. 업로드된 영상 파일 하나를 받아
학습 때와 동일한 방식으로 24프레임(RGB, 224x224)을 뽑는다.
"""

from collections import Counter

import cv2
import numpy as np

N_FRAMES = 24
IMG_SIZE = 224

# 학습 시 §4(Real·Fake 시간 밀도 진단)에서 자동 산출된 값을 그대로 써야 특징이 일관된다.
# 노트북에서 "적용할 공통 목표 구간 길이: X.XX초"로 출력된 실제 값을 여기 넣어 확정할 것.
# (TARGET_DURATION_SEC = max(min(fake_q25, real_q25), N_FRAMES/8.0) 이었고, 이 3.0은 그 하한값.
#  실제로 출력된 값이 이보다 크면 반드시 그 값으로 교체 — 학습·추론의 시간 정렬 기준이 달라지면
#  Frame Difference/Optical Flow/Temporal FFT/Temporal Wavelet 값이 학습 분포와 어긋난다.)
TARGET_DURATION_SEC = 3.0


def compute_window(cap, target_duration_sec, n_target=None):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps and fps > 0 and target_duration_sec is not None:
        window = int(round(fps * target_duration_sec))
    else:
        window = total_reported if total_reported > 0 else 10 ** 9
    if total_reported > 0:
        window = min(window, total_reported)
    if n_target is not None and total_reported > 0:
        window = max(window, min(n_target, total_reported))
    return max(window, 1)


def extract_uniform_frames_seq(cap, n_target, target_duration_sec=None):
    limit = compute_window(cap, target_duration_sec, n_target=n_target)
    target_idxs = np.linspace(0, limit - 1, num=n_target).astype(int)
    want_counts = Counter(target_idxs.tolist())
    frames_by_idx = {}
    idx = 0
    last_frame = None
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want_counts:
            frames_by_idx[idx] = frame
            last_frame = frame
        idx += 1
    frames = []
    for t in target_idxs:
        t = int(t)
        frames.append(frames_by_idx[t] if t in frames_by_idx else (last_frame.copy() if last_frame is not None else None))
    return [f for f in frames if f is not None]


def center_crop_square_resize(frame, size=IMG_SIZE):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(frame[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def extract_frames_from_video(video_path: str):
    """업로드된 영상 파일 경로 -> 24개의 RGB(224,224,3) uint8 ndarray 리스트."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    frames_bgr = extract_uniform_frames_seq(cap, N_FRAMES, target_duration_sec=TARGET_DURATION_SEC)
    cap.release()
    if len(frames_bgr) < N_FRAMES:
        raise ValueError(f"프레임을 {N_FRAMES}개 뽑지 못했습니다(영상이 너무 짧을 수 있음): {len(frames_bgr)}개만 추출됨")
    frames_rgb = [cv2.cvtColor(center_crop_square_resize(f), cv2.COLOR_BGR2RGB) for f in frames_bgr]
    return frames_rgb
