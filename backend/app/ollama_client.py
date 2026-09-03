import json
import os
import re
import threading
from pathlib import Path
from time import perf_counter

import httpx

from .menu_store import MenuSearchResult, menu_store
from .menu_advisor import advise_menu
from .schemas import ORDER_JSON_SCHEMA, OrderInterpretation


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _load_json(name: str):
    with (DATA_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


DIALECT_EXAMPLES = _load_json("dialect_examples.json")


SYSTEM_PROMPT = f"""
당신은 경상도·부산·울산·경남 지역 어르신의 말을 의미가 보존된 자연스러운
표준 주문문으로 바꾸는 주문 언어 변환기입니다.

가장 중요한 결과는 standard_order입니다. JSON은 프로그램 내부 전달 형식일 뿐이며,
standard_order에는 직원과 사용자가 바로 이해할 수 있는 자연스러운 존댓말 주문문을
작성하십시오.

판단 규칙:
1. 사투리 어미와 표현을 표준어로 바꾸되 메뉴, 수량, 옵션, 부정 표현과 사용자의
   자기수정은 임의로 빼거나 반대로 해석하지 않습니다.
2. "하나, 아니 두 개"처럼 사용자가 정정하면 마지막 의도를 반영합니다.
3. "파는 내가 영 못 묵는다"와 "파는 빼 주이소"처럼 주문 의도가 분명하면
   "파는 빼 주세요"로 정리합니다.
4. "달달한 거"처럼 정확한 메뉴가 아닌 취향 표현은 제공된 검색 후보만 사용하여
   후보를 안내하고 어느 메뉴인지 한 번 질문합니다. 사용자가 말하지 않은
   "음료", "식사" 등의 분류를 standard_order에 임의로 추가하지 않습니다.
5. 지시어나 수량의 대상이 불명확하면 추측하지 말고 clarification을 반환합니다.
6. 알레르기나 안전 문제가 있어 보일 때에는 텍스트만 보고 확정하지 않습니다.
7. items에는 아래의 '검색된 실제 판매 메뉴'에 포함된 메뉴만 넣습니다.
8. 검색 결과가 없으면 메뉴를 만들어 내지 말고 사용자가 메뉴를 다시 말하도록 질문합니다.
9. 주문과 관계없는 말은 non_order로 분류합니다.
10. summary에는 standard_order와 같은 의미의 간결한 문장을 작성합니다.

변환 예시:
- "돼지국밥 하나 아니 두 개 주이소"
  -> standard_order: "돼지국밥 2개 주세요."
  -> intent: order, quantity: 2
- "팥은 싫고 달달한 거 주이소"
  -> standard_order: "팥이 들어가지 않은 달콤한 메뉴를 원합니다."
  -> 여러 후보가 있으면 clarification으로 메뉴를 다시 질문합니다.

방언 예시:
{json.dumps(DIALECT_EXAMPLES, ensure_ascii=False)}
""".strip()


class OllamaError(RuntimeError):
    pass


_KANANA_MODEL = None
_KANANA_TOKENIZER = None
_KANANA_ADAPTER_PATH = None
_KANANA_LOCK = threading.Lock()


def _load_kanana():
    global _KANANA_MODEL, _KANANA_TOKENIZER, _KANANA_ADAPTER_PATH
    if _KANANA_MODEL is not None and _KANANA_TOKENIZER is not None:
        return _KANANA_MODEL, _KANANA_TOKENIZER

    with _KANANA_LOCK:
        if _KANANA_MODEL is not None and _KANANA_TOKENIZER is not None:
            return _KANANA_MODEL, _KANANA_TOKENIZER
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            model_path = os.getenv(
                "KANANA_MODEL_PATH",
                r"D:\AIModels\kanana-2-3b-instruct",
            )
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            _KANANA_TOKENIZER = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                fix_mistral_regex=True,
            )
            _KANANA_MODEL = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                quantization_config=quantization,
                device_map="auto",
                max_memory={0: "4GiB", "cpu": "24GiB"},
                low_cpu_mem_usage=True,
            )
            adapter_path = os.getenv("KANANA_ADAPTER_PATH", "").strip()
            if adapter_path:
                from peft import PeftModel

                _KANANA_MODEL = PeftModel.from_pretrained(
                    _KANANA_MODEL,
                    adapter_path,
                    local_files_only=True,
                )
                _KANANA_ADAPTER_PATH = adapter_path
            _KANANA_MODEL.eval()
        except Exception as exc:
            _KANANA_MODEL = None
            _KANANA_TOKENIZER = None
            _KANANA_ADAPTER_PATH = None
            raise OllamaError(f"Kanana 모델을 불러오지 못했습니다: {exc}") from exc
    return _KANANA_MODEL, _KANANA_TOKENIZER


def _extract_json(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```json").removeprefix("```").strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    start = candidate.find("{")
    if start < 0:
        raise OllamaError("Kanana가 JSON 주문 결과를 반환하지 않았습니다.")
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Kanana 주문 JSON을 읽지 못했습니다: {exc}") from exc
    return json.dumps(value, ensure_ascii=False)


def _call_kanana(
    messages: list[dict],
    *,
    max_new_tokens: int = 96,
    assistant_prefill: str = "",
) -> tuple[str, dict]:
    try:
        import torch

        model, tokenizer = _load_kanana()
        template_options = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_options,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, **template_options)
        prompt += assistant_prefill

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        with _KANANA_LOCK, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0, inputs["input_ids"].shape[1] :]
        content = assistant_prefill + tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )
        return content, {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(output_ids.shape[0]),
        }
    except OllamaError:
        raise
    except Exception as exc:
        raise OllamaError(f"Kanana 추론에 실패했습니다: {exc}") from exc


def _validate_against_menu(
    result: OrderInterpretation,
    *,
    utterance: str,
    retrieved_ids: set[str],
) -> OrderInterpretation:
    validation_errors: list[str] = []
    valid_items = []

    for item in result.items:
        menu = menu_store.get_menu(item.menu_id)
        if (
            menu is None
            or not menu["available"]
            or item.menu_id not in retrieved_ids
        ):
            validation_errors.append(item.menu_name or item.menu_id)
            continue

        item.menu_name = menu["name"]
        invalid_options = [
            option for option in item.options if option not in menu["allowed_options"]
        ]
        if invalid_options:
            validation_errors.extend(invalid_options)
            item.options = [
                option
                for option in item.options
                if option in menu["allowed_options"]
            ]
        valid_items.append(item)

    result.items = valid_items

    if validation_errors:
        result.unknown_terms = list(
            dict.fromkeys([*result.unknown_terms, *validation_errors])
        )

    if result.intent == "non_order":
        result.items = []
        result.needs_clarification = False
        result.clarification_question = None
        result.summary = ""
        return result

    if result.intent == "clarification":
        result.needs_clarification = True

    if result.needs_clarification:
        result.intent = "clarification"
        result.clarification_question = (
            result.clarification_question
            or "어떤 메뉴를 원하시는지 다시 말씀해 주시겠습니까?"
        )

    result.standard_order = result.standard_order.strip()
    if not result.standard_order and result.summary:
        result.standard_order = result.summary.strip()
    if not result.standard_order and result.intent == "order":
        result.standard_order = utterance.strip()

    if result.standard_order:
        result.summary = result.standard_order
    elif result.intent != "non_order":
        result.summary = ""

    return result


def _retrieval_payload(results: list[MenuSearchResult]) -> list[dict]:
    return [result.as_prompt_dict() for result in results]


def _kanana_retrieval_payload(results: list[MenuSearchResult]) -> list[dict]:
    return [
        {
            "menu_id": result.menu["id"],
            "menu_name": result.menu["name"],
            "allowed_options": result.menu["allowed_options"],
        }
        for result in results
    ]


KANANA_OUTPUT_GUIDE = """
입력 문장의 의미를 보존해 표준어 주문문으로 바꾸세요.
답은 반드시 아래 셋 중 한 줄만 출력하세요.
ORDER|확정된 자연스러운 표준 주문문
CLARIFY|사용자에게 다시 물을 한 문장
NON_ORDER|
메뉴, 수량, 빼 달라는 재료와 사용자의 자기수정을 절대 생략하지 마세요.
후보가 여러 개이고 메뉴를 특정할 수 없으면 CLARIFY, 주문과 무관하면 NON_ORDER입니다.
""".strip()


KANANA_SYSTEM_PROMPT = """
당신은 경상도·부산·울산·경남 지역 어르신의 말을 표준 주문문으로 바꾸는 변환기입니다.
사용자가 말한 메뉴, 수량, 제외·추가 옵션과 자기수정을 빠뜨리거나 만들어 내지 마세요.
제공된 실제 메뉴 후보만 사용하고, 특정할 수 없으면 한 번만 다시 질문하세요.
설명이나 마크다운 없이 요청한 한 줄 형식만 출력하세요.
""".strip()


_QUANTITY_PATTERNS = (
    (
        re.compile(
            r"(?:(?<![가-힣])(?:하나|1)\s*(?:개|그릇|잔|병)?"
            r"(?=\s|$|하고|랑|와|과|로|라|만|씩)|한\s*(?:개|그릇|잔|병))"
        ),
        1,
    ),
    (
        re.compile(
            r"(?:(?<![가-힣])(?:둘|2)\s*(?:개|그릇|잔|병)?"
            r"(?=\s|$|하고|랑|와|과|로|라|만|씩)|두\s*(?:개|그릇|잔|병))"
        ),
        2,
    ),
    (
        re.compile(
            r"(?:(?<![가-힣])(?:셋|3)\s*(?:개|그릇|잔|병)?"
            r"(?=\s|$|하고|랑|와|과|로|라|만|씩)|세\s*(?:개|그릇|잔|병))"
        ),
        3,
    ),
    (
        re.compile(
            r"(?:(?<![가-힣])(?:넷|4)\s*(?:개|그릇|잔|병)?"
            r"(?=\s|$|하고|랑|와|과|로|라|만|씩)|네\s*(?:개|그릇|잔|병))"
        ),
        4,
    ),
    (
        re.compile(
            r"(?:(?<![가-힣])(?:다섯|5)\s*(?:개|그릇|잔|병)?"
            r"(?=\s|$|하고|랑|와|과|로|라|만|씩))"
        ),
        5,
    ),
)


def _quantity_from_text(text: str) -> int:
    matches: list[tuple[int, int]] = []
    for pattern, value in _QUANTITY_PATTERNS:
        matches.extend((match.start(), value) for match in pattern.finditer(text))
    return sorted(matches)[-1][1] if matches else 1


def _explicit_menu_results(
    utterance: str,
    retrieved: list[MenuSearchResult],
) -> list[MenuSearchResult]:
    normalized = re.sub(r"[^0-9a-z가-힣]+", "", utterance.lower())
    positioned: list[tuple[int, MenuSearchResult]] = []
    for result in retrieved:
        names = [result.menu["name"], *result.menu["aliases"]]
        positions = [
            normalized.find(re.sub(r"[^0-9a-z가-힣]+", "", name.lower()))
            for name in names
        ]
        positions = [position for position in positions if position >= 0]
        if positions:
            positioned.append((min(positions), result))
    return [result for _, result in sorted(positioned, key=lambda item: item[0])]


def _options_from_text(menu: dict, text: str) -> list[str]:
    options = []
    for option in menu["allowed_options"]:
        ingredient = option.removesuffix(" 빼기").removesuffix(" 따로")
        if option.endswith(" 빼기") and re.search(
            rf"{re.escape(ingredient)}(?:은|는|이|가|을|를)?[^.!?\n]{{0,14}}"
            r"(?:없이|없는\s*(?:걸|것)?(?:로)?|빼|제외|싫|"
            r"못\s*(?:먹|묵)|넣지\s*마|안\s*넣|필요\s*없)",
            text,
        ):
            options.append(option)
        elif option.endswith(" 따로") and re.search(
            rf"{re.escape(ingredient)}(?:은|는|이|가|을|를)?[^.!?\n]{{0,8}}따로",
            text,
        ):
            options.append(option)
        elif option == "연하게" and "연하게" in text:
            options.append(option)
        elif option == "샷 추가" and re.search(r"샷\s*(?:하나\s*)?(?:추가|더)", text):
            options.append(option)
    return options


def _build_order_items(
    utterance: str,
    retrieved: list[MenuSearchResult],
) -> list[dict]:
    explicit = _explicit_menu_results(utterance, retrieved)
    items = []
    for index, result in enumerate(explicit):
        menu = result.menu
        names = [menu["name"], *menu["aliases"]]
        positions = [
            (utterance.find(name), name)
            for name in names
            if utterance.find(name) >= 0
        ]
        start, matched_name = min(positions) if positions else (0, "")
        end = len(utterance)
        for other in explicit[index + 1 :]:
            other_positions = [
                utterance.find(name, start + len(matched_name))
                for name in [other.menu["name"], *other.menu["aliases"]]
            ]
            other_positions = [position for position in other_positions if position >= 0]
            if other_positions:
                end = min(end, min(other_positions))
        segment = utterance[start:end]
        quantity = _quantity_from_text(segment)
        options = _options_from_text(menu, utterance)
        partial = re.search(
            r"((?:한|하나|두|둘|세|셋|네|넷|\d+)\s*(?:개|잔|그릇|병))"
            r"\s*만\s*연하게",
            segment,
        )
        if partial and "연하게" in menu["allowed_options"]:
            total_quantity = _quantity_from_text(segment[: partial.start()])
            partial_quantity = _quantity_from_text(partial.group(1))
            if 0 < partial_quantity < total_quantity:
                items.append(
                    {
                        "menu_id": menu["id"],
                        "menu_name": menu["name"],
                        "quantity": total_quantity - partial_quantity,
                        "options": [],
                    }
                )
                items.append(
                    {
                        "menu_id": menu["id"],
                        "menu_name": menu["name"],
                        "quantity": partial_quantity,
                        "options": ["연하게"],
                    }
                )
                continue
        items.append(
            {
                "menu_id": menu["id"],
                "menu_name": menu["name"],
                "quantity": quantity,
                "options": options,
            }
        )
    return items


def _canonical_order(items: list[dict]) -> str:
    parts = []
    for item in items:
        menu = menu_store.get_menu(item["menu_id"])
        unit = "잔" if menu and menu["category"] in {"음료", "커피", "차"} else "개"
        text = f"{item['menu_name']} {item['quantity']}{unit}"
        exclusions = [
            option.removesuffix(" 빼기")
            for option in item["options"]
            if option.endswith(" 빼기")
        ]
        separate = [
            option.removesuffix(" 따로")
            for option in item["options"]
            if option.endswith(" 따로")
        ]
        additions = [
            option
            for option in item["options"]
            if not option.endswith(" 빼기") and not option.endswith(" 따로")
        ]
        if exclusions:
            text += f", {', '.join(exclusions)}는 빼고"
        if separate:
            text += f", {', '.join(separate)}는 따로"
        if additions:
            text += f", {', '.join(additions)}"
        parts.append(text)
    return ", ".join(parts) + " 주세요."


def _interpret_kanana_text(
    content: str,
    *,
    utterance: str,
    retrieved: list[MenuSearchResult],
) -> OrderInterpretation:
    line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    marker, _, model_text = line.partition("|")
    marker = marker.strip().upper()
    model_text = model_text.strip()
    items = _build_order_items(utterance, retrieved)

    has_order_cue = bool(
        re.search(
            r"(?:주이소|주세요|주라|달라|먹|묵|마시|하나|두\s*개|한\s*잔)",
            utterance,
        )
    )
    if marker == "NON_ORDER" or (not retrieved and not has_order_cue):
        return OrderInterpretation(
            intent="non_order",
            standard_order="",
            items=[],
            needs_clarification=False,
            confidence=0.9,
            summary="",
        )
    if marker == "CLARIFY" or not items:
        question = model_text or "어떤 메뉴를 원하시는지 다시 말씀해 주시겠어요?"
        if re.search(r"(?:달달|달콤|단\s*거|단것)", utterance):
            standard_order = "달달한 메뉴를 원합니다."
        else:
            standard_order = model_text
        return OrderInterpretation(
            intent="clarification",
            standard_order=standard_order,
            items=[],
            missing_fields=["menu"],
            needs_clarification=True,
            clarification_question=question,
            confidence=0.7,
            summary=standard_order,
        )

    standard_order = _canonical_order(items)
    return OrderInterpretation(
        intent="order",
        standard_order=standard_order,
        items=items,
        needs_clarification=False,
        confidence=0.9,
        summary=standard_order,
    )


def _topic_form(text: str) -> str:
    last = text[-1]
    has_batchim = "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0
    return f"{text}{'은' if has_batchim else '는'}"


def _has_explicit_menu(results: list[MenuSearchResult]) -> bool:
    for result in results:
        names = {result.menu["name"], *result.menu["aliases"]}
        if names.intersection(result.matched_terms):
            return True
    return False


def _guard_ambiguous_preference(
    result: OrderInterpretation,
    retrieved: list[MenuSearchResult],
    *,
    utterance: str,
) -> OrderInterpretation:
    if not retrieved or _has_explicit_menu(retrieved):
        return result

    candidate_names = [item.menu["name"] for item in retrieved[:3]]
    joined = ", ".join(candidate_names)
    categories = {item.menu["category"] for item in retrieved}
    if len(categories) > 1:
        for category_word in ("음료", "식사", "차", "죽", "커피"):
            result.standard_order = result.standard_order.replace(category_word, "메뉴")
    negated_ingredients = menu_store.negated_ingredients(utterance)
    missing_exclusions = [
        ingredient
        for ingredient in negated_ingredients
        if ingredient not in result.standard_order
    ]
    if missing_exclusions:
        exclusions = "·".join(missing_exclusions)
        result.standard_order = (
            f"{_topic_form(exclusions)} 제외하고, {result.standard_order}"
        )
    result.summary = result.standard_order
    result.intent = "clarification"
    result.items = []
    result.needs_clarification = True
    result.missing_fields = list(dict.fromkeys([*result.missing_fields, "menu"]))
    result.clarification_question = (
        f"조건에 맞는 메뉴로 {joined} 등이 있습니다. 어떤 메뉴로 드릴까요?"
    )
    result.confidence = min(result.confidence, 0.75)
    return result


def interpret_order(
    utterance: str,
    *,
    context: str | None = None,
    prosody: dict | None = None,
    timeout_seconds: float = 90,
) -> tuple[OrderInterpretation, dict]:
    advisory_result = advise_menu(utterance, menu_store)
    if advisory_result is not None:
        return advisory_result, {
            "engine": "closed-menu-advisor",
            "model": None,
            "adapter": None,
            "elapsed_seconds": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "retrieval": {
                "source": "sqlite",
                "catalog": "json",
                "candidate_count": len(advisory_result.recommendations),
                "candidates": [
                    {
                        "menu_id": item.menu_id,
                        "menu_name": item.menu_name,
                        "score": 1.0,
                        "matched_terms": [item.reason],
                    }
                    for item in advisory_result.recommendations
                ],
            },
        }

    backend = os.getenv("ORDER_MODEL_BACKEND", "kanana").strip().lower()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    kanana_model = os.getenv(
        "KANANA_MODEL_ID",
        "kakaocorp/kanana-2-3b-instruct",
    )

    retrieval_query = f"{context or ''} {utterance}".strip()
    retrieved = menu_store.search(retrieval_query, limit=5)
    retrieved_payload = _retrieval_payload(retrieved)
    retrieved_ids = {item.menu["id"] for item in retrieved}

    user_text = (
        f"주문 발화: {utterance}\n"
        "검색된 실제 판매 메뉴:\n"
        f"{json.dumps(retrieved_payload, ensure_ascii=False)}"
    )
    if context:
        user_text += f"\n직전 대화 문맥: {context}"
    if not retrieved:
        user_text += (
            "\n검색 결과가 없습니다. 매장에 없는 메뉴를 생성하지 말고 "
            "사용자에게 메뉴를 다시 확인하십시오."
        )
    user_text += (
        "\n반드시 설명이나 마크다운 없이 다음 JSON 스키마를 만족하는 JSON 객체 하나만 "
        "반환하십시오:\n"
        f"{json.dumps(ORDER_JSON_SCHEMA, ensure_ascii=False)}"
    )

    ollama_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    kanana_user_text = (
        f"{KANANA_OUTPUT_GUIDE}\n\n"
        f"주문 발화: {utterance}\n"
        "사용 가능한 메뉴 후보:\n"
        f"{json.dumps(_kanana_retrieval_payload(retrieved), ensure_ascii=False)}"
    )
    if context:
        kanana_user_text += f"\n직전 대화 문맥: {context}"
    if prosody:
        kanana_user_text += (
            "\n음성 억양 분석(보조 신호이며 문장 내용보다 우선하지 않음): "
            f"{json.dumps(prosody, ensure_ascii=False)}"
        )
        if prosody.get("type") == "question":
            kanana_user_text += (
                "\n억양이 질문·확인형이면 주문으로 단정하지 말고, 문장 내용까지 질문으로 "
                "해석될 때 clarification으로 재확인하세요."
            )
        elif prosody.get("type") == "request":
            kanana_user_text += (
                "\n억양이 명령·요청형이면 메뉴 후보와 문장 내용이 일치할 때 주문 의도를 "
                "판단하는 보조 근거로 사용하세요."
            )
    if not retrieved:
        kanana_user_text += (
            "\n사용 가능한 후보가 없습니다. 메뉴를 만들지 말고 clarification으로 "
            "메뉴를 다시 질문하세요."
        )
    kanana_messages = [
        {"role": "system", "content": KANANA_SYSTEM_PROMPT},
        {"role": "user", "content": kanana_user_text},
    ]
    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
        "think": False,
        "format": ORDER_JSON_SCHEMA,
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 300,
        },
    }

    started = perf_counter()
    if backend == "kanana":
        content, token_metrics = _call_kanana(kanana_messages)
        raw = {
            "message": {"content": content},
            "prompt_eval_count": token_metrics["prompt_tokens"],
            "eval_count": token_metrics["output_tokens"],
        }
    else:
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(f"{base_url}/api/chat", json=payload)
                response.raise_for_status()
                raw = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(f"Ollama 호출에 실패했습니다: {exc}") from exc
        content = raw.get("message", {}).get("content", "")
        if not content:
            raise OllamaError("Ollama가 빈 응답을 반환했습니다.")

    if backend == "kanana":
        parsed = _interpret_kanana_text(
            content,
            utterance=utterance,
            retrieved=retrieved,
        )
    else:
        try:
            parsed = OrderInterpretation.model_validate_json(content)
        except ValueError as exc:
            raise OllamaError(f"주문 JSON 검증에 실패했습니다: {exc}") from exc

    parsed = _validate_against_menu(
        parsed,
        utterance=utterance,
        retrieved_ids=retrieved_ids,
    )
    parsed = _guard_ambiguous_preference(
        parsed,
        retrieved,
        utterance=utterance,
    )
    adapter_path = os.getenv("KANANA_ADAPTER_PATH", "").strip()
    metrics = {
        "engine": (
            "kanana-qlora-closed-menu-rag"
            if backend == "kanana" and adapter_path
            else f"{backend}-closed-menu-rag"
        ),
        "model": kanana_model if backend == "kanana" else model,
        "adapter": adapter_path or None,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "prompt_tokens": raw.get("prompt_eval_count"),
        "output_tokens": raw.get("eval_count"),
        "retrieval": {
            "source": "sqlite",
            "catalog": "json",
            "candidate_count": len(retrieved),
            "candidates": [
                {
                    "menu_id": item.menu["id"],
                    "menu_name": item.menu["name"],
                    "score": round(item.score, 3),
                    "matched_terms": list(item.matched_terms),
                }
                for item in retrieved
            ],
        },
    }
    return parsed, metrics
