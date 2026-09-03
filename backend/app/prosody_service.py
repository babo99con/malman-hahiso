from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "data" / "prosody_classifier.json"
CLASS_LABELS = ("statement", "question", "request")
DISPLAY_LABELS = {
    "statement": "평서·설명",
    "question": "질문·확인",
    "request": "명령·요청",
    "uncertain": "불확실·재확인 필요",
}

_MODEL: dict | None = None
_MODEL_LOCK = threading.Lock()


def _model_path() -> Path:
    return Path(os.getenv("PROSODY_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def _resample(values: np.ndarray, size: int = 24) -> np.ndarray:
    if values.size == 0:
        return np.zeros(size, dtype=np.float32)
    if values.size == 1:
        return np.full(size, values[0], dtype=np.float32)
    old = np.linspace(0.0, 1.0, values.size)
    new = np.linspace(0.0, 1.0, size)
    return np.interp(new, old, values).astype(np.float32)


def pitch_features(raw_pitch: list[float] | np.ndarray) -> np.ndarray:
    pitch = np.asarray(raw_pitch, dtype=np.float32)
    pitch = pitch[np.isfinite(pitch) & (pitch >= 45.0) & (pitch <= 600.0)]
    if pitch.size < 8:
        raise ValueError("유효한 음높이 구간이 너무 짧습니다.")

    median = float(np.median(pitch))
    normalized = np.log2(np.maximum(pitch, 1.0) / max(median, 1.0))
    contour = _resample(normalized, 24)
    window = max(2, pitch.size // 8)
    start = float(np.median(normalized[:window]))
    before_end = float(np.median(normalized[-2 * window : -window]))
    end = float(np.median(normalized[-window:]))
    x = np.linspace(-1.0, 1.0, normalized.size)
    slope = float(np.polyfit(x, normalized, 1)[0]) if normalized.size > 1 else 0.0
    diff = np.diff(normalized)
    summary = np.asarray(
        [
            math.log1p(pitch.size),
            math.log2(max(median, 1.0) / 150.0),
            float(np.std(normalized)),
            float(np.percentile(normalized, 90) - np.percentile(normalized, 10)),
            slope,
            start,
            before_end,
            end,
            end - before_end,
            end - start,
            float(np.argmax(normalized) / max(1, normalized.size - 1)),
            float(np.mean(np.abs(diff))) if diff.size else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([contour, summary])


def _load_model() -> dict:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        path = _model_path()
        if not path.is_file():
            raise FileNotFoundError(f"억양 분류기 파일이 없습니다: {path}")
        _MODEL = json.loads(path.read_text(encoding="utf-8"))
        return _MODEL


def get_prosody_status() -> dict:
    path = _model_path()
    model = _MODEL
    return {
        "enabled": os.getenv("PROSODY_ENABLED", "1").strip() != "0",
        "model_path": str(path),
        "model_ready": path.is_file(),
        "loaded": model is not None,
        "classes": list(model.get("classes", CLASS_LABELS)) if model else list(CLASS_LABELS),
        "task": "gyeongsang-prosody-intent-classification",
    }


def _extract_pitch(path: Path) -> tuple[np.ndarray, float]:
    import torch
    from faster_whisper.audio import decode_audio
    from torchaudio.functional import detect_pitch_frequency

    sample_rate = 16_000
    waveform = decode_audio(str(path), sampling_rate=sample_rate)
    if waveform.size < sample_rate // 4:
        return np.asarray([], dtype=np.float32), waveform.size / sample_rate

    waveform = waveform[: sample_rate * int(os.getenv("ASR_MAX_AUDIO_SECONDS", "30"))]
    signal = torch.from_numpy(waveform).float().unsqueeze(0)
    signal = signal - signal.mean(dim=-1, keepdim=True)
    peak = signal.abs().amax()
    if float(peak) > 0:
        signal = signal / peak
    requested_device = os.getenv("PROSODY_DEVICE", "cuda").strip().lower()
    device = (
        torch.device("cuda")
        if requested_device == "cuda" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    signal = signal.to(device)
    with torch.inference_mode():
        pitch = detect_pitch_frequency(
            signal,
            sample_rate=sample_rate,
            frame_time=0.01,
            win_length=15,
            freq_low=55,
            freq_high=450,
        )[0]
    values = pitch.detach().cpu().numpy()
    values = values[np.isfinite(values) & (values >= 55.0) & (values <= 450.0)]
    return values.astype(np.float32), waveform.size / sample_rate


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def analyze_prosody(path: Path) -> dict:
    started = perf_counter()
    if os.getenv("PROSODY_ENABLED", "1").strip() == "0":
        return {
            "type": "uncertain",
            "label": DISPLAY_LABELS["uncertain"],
            "confidence": 0.0,
            "reason": "disabled",
            "elapsed_seconds": 0.0,
        }

    model = _load_model()
    pitch, audio_seconds = _extract_pitch(path)
    if pitch.size < 8:
        return {
            "type": "uncertain",
            "label": DISPLAY_LABELS["uncertain"],
            "confidence": 0.0,
            "reason": "insufficient_pitch",
            "pitch_points": int(pitch.size),
            "audio_seconds": round(audio_seconds, 3),
            "elapsed_seconds": round(perf_counter() - started, 3),
        }

    features = pitch_features(pitch)
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    scale = np.asarray(model["feature_scale"], dtype=np.float32)
    standardized = (features - mean) / np.maximum(scale, 1e-6)
    hidden = standardized
    if model.get("layers"):
        for index, layer in enumerate(model["layers"]):
            weights = np.asarray(layer["weights"], dtype=np.float32)
            bias = np.asarray(layer["bias"], dtype=np.float32)
            hidden = weights @ hidden + bias
            if index < len(model["layers"]) - 1:
                hidden = np.maximum(hidden, 0.0)
        logits = hidden
    else:
        weights = np.asarray(model["weights"], dtype=np.float32)
        bias = np.asarray(model["bias"], dtype=np.float32)
        logits = weights @ hidden + bias
    probabilities = _softmax(logits)
    classes = model.get("classes", list(CLASS_LABELS))
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    threshold = float(os.getenv("PROSODY_CONFIDENCE_THRESHOLD", "0.70"))
    predicted = classes[best_index] if confidence >= threshold else "uncertain"

    ending_delta = float(features[32])
    ending = "rising" if ending_delta >= 0.08 else "falling" if ending_delta <= -0.08 else "level"
    return {
        "type": predicted,
        "label": DISPLAY_LABELS[predicted],
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(probabilities[index]), 4)
            for index, name in enumerate(classes)
        },
        "ending": ending,
        "ending_delta": round(ending_delta, 4),
        "pitch_points": int(pitch.size),
        "audio_seconds": round(audio_seconds, 3),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "model_version": model.get("version", "unknown"),
    }
