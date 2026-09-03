from __future__ import annotations

import os
import threading
from pathlib import Path
from statistics import mean
from time import perf_counter


_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = threading.Lock()

DEFAULT_ADAPTER_PATH = Path(
    r"D:\AIData\malhaeye\runs\whisper-small-lora-v2-order-senior\final-adapter"
)
DEFAULT_CACHE_PATH = Path(r"D:\AIData\malhaeye\hf-cache")


def _backend() -> str:
    return os.getenv("ASR_BACKEND", "malhaeye-lora").strip().lower()


def _adapter_path() -> Path:
    return Path(os.getenv("ASR_ADAPTER_PATH", str(DEFAULT_ADAPTER_PATH)))


def get_asr_status() -> dict:
    adapter_path = _adapter_path()
    backend = _backend()
    return {
        "backend": backend,
        "model": (
            "malhaeye-whisper-small-lora-v2-order-senior"
            if backend == "malhaeye-lora"
            else os.getenv("ASR_MODEL", "small")
        ),
        "device": os.getenv(
            "ASR_DEVICE", "cuda" if backend == "malhaeye-lora" else "cpu"
        ),
        "adapter_path": str(adapter_path) if backend == "malhaeye-lora" else None,
        "adapter_ready": adapter_path.is_dir() if backend == "malhaeye-lora" else None,
        "loaded": _MODEL is not None,
    }


def _get_lora_model():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _PROCESSOR, _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _PROCESSOR, _MODEL

        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        adapter_path = _adapter_path()
        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"Whisper LoRA adapter not found: {adapter_path}"
            )

        base_model = os.getenv("ASR_BASE_MODEL", "openai/whisper-small")
        cache_dir = os.getenv("ASR_MODEL_CACHE", str(DEFAULT_CACHE_PATH))
        requested_device = os.getenv("ASR_DEVICE", "cuda").strip().lower()
        device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        processor = WhisperProcessor.from_pretrained(
            base_model,
            language="Korean",
            task="transcribe",
            cache_dir=cache_dir,
        )
        base = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )
        model = PeftModel.from_pretrained(base, str(adapter_path))
        model = model.merge_and_unload()
        model.to(device)
        model.eval()

        _PROCESSOR = processor
        _MODEL = model
        return _PROCESSOR, _MODEL


def _transcribe_lora(path: Path) -> tuple[str, dict]:
    import torch
    from faster_whisper.audio import decode_audio

    started = perf_counter()
    processor, model = _get_lora_model()
    sample_rate = 16_000
    max_seconds = int(os.getenv("ASR_MAX_AUDIO_SECONDS", "30"))
    waveform = decode_audio(str(path), sampling_rate=sample_rate)
    if waveform.size == 0:
        return "", {
            **get_asr_status(),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "average_log_probability": None,
            "audio_seconds": 0.0,
        }

    waveform = waveform[: sample_rate * max_seconds]
    inputs = processor.feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    device = next(model.parameters()).device
    input_features = inputs.input_features.to(
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    attention_mask = inputs.attention_mask.to(device)

    with _MODEL_LOCK, torch.inference_mode():
        generated = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            language="Korean",
            task="transcribe",
            max_new_tokens=128,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    transcript = processor.batch_decode(
        generated.sequences,
        skip_special_tokens=True,
    )[0].strip()
    average_log_probability = None
    if generated.scores:
        transition_scores = model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            normalize_logits=True,
        )
        if transition_scores.numel():
            average_log_probability = float(transition_scores.mean().item())

    return transcript, {
        **get_asr_status(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "average_log_probability": (
            round(average_log_probability, 4)
            if average_log_probability is not None
            else None
        ),
        "audio_seconds": round(waveform.size / sample_rate, 3),
    }


def _get_faster_whisper_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from faster_whisper import WhisperModel

                _MODEL = WhisperModel(
                    os.getenv("ASR_MODEL", "small"),
                    device=os.getenv("ASR_DEVICE", "cpu"),
                    compute_type=os.getenv("ASR_COMPUTE_TYPE", "int8"),
                    download_root=os.getenv(
                        "ASR_MODEL_CACHE", r"D:\AIData\malhaeye\models"
                    ),
                )
    return _MODEL


def _transcribe_faster_whisper(path: Path) -> tuple[str, dict]:
    started = perf_counter()
    segments_iter, info = _get_faster_whisper_model().transcribe(
        str(path),
        language="ko",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments = list(segments_iter)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    average_log_probability = (
        mean(segment.avg_logprob for segment in segments) if segments else None
    )
    return transcript, {
        **get_asr_status(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "language_probability": round(info.language_probability, 4),
        "average_log_probability": (
            round(average_log_probability, 4)
            if average_log_probability is not None
            else None
        ),
        "segments": len(segments),
    }


def transcribe_audio(path: Path) -> tuple[str, dict]:
    if _backend() == "malhaeye-lora":
        return _transcribe_lora(path)
    return _transcribe_faster_whisper(path)
