import os
import secrets
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .asr_service import get_asr_status, transcribe_audio
from .menu_store import menu_store
from .ollama_client import OllamaError, interpret_order
from .prosody_service import analyze_prosody, get_prosody_status
from .rule_engine import interpret_order_rules
from .schemas import OrderInterpretation


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

app = FastAPI(
    title="말해예 로컬 주문 추론 API",
    version="0.2.0",
    description="사투리 음성을 표준 주문문으로 변환하고 선택적으로 메뉴와 대조하는 시험용 API",
)


class InterpretRequest(BaseModel):
    utterance: str = Field(min_length=1, max_length=500)
    context: str | None = Field(default=None, max_length=1000)


class InterpretResponse(BaseModel):
    result: OrderInterpretation
    metrics: dict


class VoiceOrderResponse(BaseModel):
    transcript: str
    corrected_text: str
    result: OrderInterpretation
    prosody: dict
    metrics: dict


def require_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("LOCAL_INFERENCE_TOKEN", "").strip()
    if not expected:
        return

    supplied = ""
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid inference token")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "asr": get_asr_status(),
        "prosody": get_prosody_status(),
        "order_parser": {
            "primary": (
                os.getenv("KANANA_MODEL_ID", "kakaocorp/kanana-2-3b-instruct")
                if os.getenv("ORDER_MODEL_BACKEND", "kanana") == "kanana"
                else os.getenv("OLLAMA_MODEL", "qwen3:8b")
            ),
            "backend": os.getenv("ORDER_MODEL_BACKEND", "kanana"),
            "adapter": os.getenv("KANANA_ADAPTER_PATH") or None,
            "task": "dialect-to-standard-order-with-closed-menu-rag",
            "fallback": "rules for voice requests only",
        },
        "menu_store": menu_store.status(),
        "exposure": "local-only unless a protected tunnel is configured",
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get(
    "/api/v1/menus",
    dependencies=[Depends(require_token)],
)
def list_menus():
    return {
        "menus": menu_store.list_menus(),
        "status": menu_store.status(),
    }


@app.get(
    "/api/v1/menus/search",
    dependencies=[Depends(require_token)],
)
def search_menus(q: str, limit: int = 5):
    if not q.strip():
        raise HTTPException(status_code=422, detail="검색어를 입력해 주세요.")
    results = menu_store.search(q, limit=limit)
    return {
        "query": q,
        "results": [result.as_api_dict() for result in results],
        "retrieval": "closed-menu-hybrid",
    }


@app.post(
    "/api/v1/interpret-rules",
    response_model=InterpretResponse,
    dependencies=[Depends(require_token)],
)
def interpret_rules(request: InterpretRequest):
    result, metrics = interpret_order_rules(request.utterance)
    return InterpretResponse(result=result, metrics=metrics)


@app.post(
    "/api/v1/voice-order",
    response_model=VoiceOrderResponse,
    dependencies=[Depends(require_token)],
)
def voice_order(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    content = audio.file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="음성 파일은 20MB 이하여야 합니다.")

    temp_path: Path | None = None
    prosody = {
        "type": "uncertain",
        "label": "불확실·재확인 필요",
        "confidence": 0.0,
        "reason": "not_analyzed",
    }
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        try:
            prosody = analyze_prosody(temp_path)
        except Exception as exc:
            prosody = {
                **prosody,
                "reason": "analysis_failed",
                "error": str(exc),
            }
        transcript, asr_metrics = transcribe_audio(temp_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"음성 인식에 실패했습니다: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    if not transcript:
        result = OrderInterpretation(
            intent="clarification",
            standard_order="",
            items=[],
            missing_fields=["speech"],
            needs_clarification=True,
            clarification_question="말씀을 듣지 못했습니다. 다시 말씀해 주세요.",
            confidence=0.1,
            summary="",
        )
        parser_metrics = {"engine": "empty-speech"}
    else:
        try:
            result, parser_metrics = interpret_order(transcript, prosody=prosody)
        except OllamaError as exc:
            result, parser_metrics = interpret_order_rules(transcript)
            result.standard_order = parser_metrics["corrected_text"]
            parser_metrics = {
                **parser_metrics,
                "fallback": True,
                "fallback_reason": str(exc),
            }

    average_log_probability = asr_metrics.get("average_log_probability")
    if (
        average_log_probability is not None
        and average_log_probability < -1.0
        and result.intent == "order"
    ):
        result.intent = "clarification"
        result.needs_clarification = True
        result.clarification_question = (
            "제가 정확히 들었는지 확인이 필요합니다. 주문을 한 번 더 말씀해 주세요."
        )
        result.confidence = min(result.confidence, 0.45)

    if (
        prosody.get("type") == "uncertain"
        and result.intent == "order"
        and float(prosody.get("confidence") or 0.0) < 0.4
    ):
        result.confidence = min(result.confidence, 0.65)

    if (
        prosody.get("type") == "question"
        and float(prosody.get("confidence") or 0.0) >= 0.72
        and result.intent == "order"
    ):
        result.intent = "clarification"
        result.needs_clarification = True
        result.clarification_question = (
            "질문이나 확인으로 들렸습니다. 이 내용을 주문으로 확정할까요?"
        )
        result.confidence = min(result.confidence, 0.6)

    return VoiceOrderResponse(
        transcript=transcript,
        corrected_text=result.standard_order,
        result=result,
        prosody=prosody,
        metrics={"asr": asr_metrics, "prosody": prosody, "parser": parser_metrics},
    )


@app.post(
    "/api/v1/prosody",
    dependencies=[Depends(require_token)],
)
def prosody_only(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    content = audio.file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="음성 파일은 20MB 이하여야 합니다.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        return analyze_prosody(temp_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"억양 분석에 실패했습니다: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post(
    "/api/v1/interpret",
    response_model=InterpretResponse,
    dependencies=[Depends(require_token)],
)
def interpret(request: InterpretRequest):
    try:
        result, metrics = interpret_order(
            request.utterance,
            context=request.context,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return InterpretResponse(result=result, metrics=metrics)
