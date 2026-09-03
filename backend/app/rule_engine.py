from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import OrderInterpretation, OrderItem


ROOT = Path(__file__).resolve().parents[1]
MENU = json.loads((ROOT / "data" / "menu_cafe.json").read_text(encoding="utf-8"))
MENU_BY_ID = {item["id"]: item for item in MENU["menus"]}

DIALECT_REPLACEMENTS = (
    ("주이소", "주세요"),
    ("주서요", "주세요"),
    ("주소", "주세요"),
    ("하이소", "하세요"),
    ("묵어", "먹어"),
    ("묵고", "먹고"),
    ("묵을", "먹을"),
    ("마이", "많이"),
    ("어데", "어디"),
    ("그라모", "그러면"),
    ("아이가", "아니야"),
)

NON_ORDER_MARKERS = (
    "날씨",
    "배 안 고",
    "구경만",
    "주문 안",
    "괜찮습니다",
    "안 먹",
    "다음에",
    "몇 시까지",
    "화장실",
    "사장님",
    "사람이 참 많",
    "메뉴판만",
    "기다리는 중",
    "포장은 안",
    "계산했습니다",
    "자리 있",
    "쉬는 날",
    "사진 찍",
    "카드 결제",
    "전화 받고",
)

GENERIC_MENU_MARKERS = ("국밥", "커피", "음료", "그거", "저거", "아무거나")

NUMBER_PATTERNS = (
    (
        re.compile(r"(?:(?<![가-힣])(?:하나|1)\s*(?:개|그릇|잔)?(?=\s|$|하고|랑|와|과|로|라|만|씩)|한\s*(?:개|그릇|잔))"),
        1,
    ),
    (
        re.compile(r"(?:(?<![가-힣])(?:둘|2)\s*(?:개|그릇|잔)?(?=\s|$|하고|랑|와|과|로|라|만|씩)|두\s*(?:개|그릇|잔))"),
        2,
    ),
    (
        re.compile(r"(?:(?<![가-힣])(?:셋|3)\s*(?:개|그릇|잔)?(?=\s|$|하고|랑|와|과|로|라|만|씩)|세\s*(?:개|그릇|잔))"),
        3,
    ),
    (
        re.compile(r"(?:(?<![가-힣])(?:넷|4)\s*(?:개|그릇|잔)?(?=\s|$|하고|랑|와|과|로|라|만|씩)|네\s*(?:개|그릇|잔))"),
        4,
    ),
    (
        re.compile(r"(?:(?<![가-힣])(?:다섯|5)\s*(?:개|그릇|잔)?(?=\s|$|하고|랑|와|과|로|라|만|씩))"),
        5,
    ),
)

OPTION_PATTERNS = (
    (re.compile(r"밥\s*(?:은\s*)?따로"), "밥 따로"),
    (re.compile(r"다대기\s*(?:는\s*)?따로"), "다대기 따로"),
    (
        re.compile(r"다대기(?:는|를|랑|하고)?\s*(?:없이|빼|빼고|넣지|안\s*넣)"),
        "다대기 빼기",
    ),
    (re.compile(r"파(?:는|를|랑|하고)?\s*(?:없이|빼|빼고|넣지|안\s*넣)"), "파 빼기"),
    (re.compile(r"시럽(?:은|을)?\s*(?:없이|빼|빼고|넣지|안\s*넣)"), "시럽 빼기"),
    (re.compile(r"연하게"), "연하게"),
    (re.compile(r"샷\s*(?:하나\s*)?(?:추가|더)"), "샷 추가"),
)


def correct_dialect(text: str) -> str:
    corrected = re.sub(r"\s+", " ", text).strip()
    for dialect, standard in DIALECT_REPLACEMENTS:
        corrected = corrected.replace(dialect, standard)
    return corrected


def _menu_occurrences(text: str) -> list[tuple[int, int, dict]]:
    candidates: list[tuple[int, int, dict]] = []
    for menu in MENU["menus"]:
        aliases = sorted({menu["name"], *menu["aliases"]}, key=len, reverse=True)
        for alias in aliases:
            for match in re.finditer(re.escape(alias), text):
                candidates.append((match.start(), match.end(), menu))

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, dict]] = []
    for candidate in candidates:
        start, end, _ = candidate
        if any(not (end <= old_start or start >= old_end) for old_start, old_end, _ in selected):
            continue
        selected.append(candidate)
    return sorted(selected)


def _last_quantity(text: str) -> int:
    matches: list[tuple[int, int]] = []
    for pattern, value in NUMBER_PATTERNS:
        matches.extend((match.start(), value) for match in pattern.finditer(text))
    return max(matches, default=(-1, 1), key=lambda item: item[0])[1]


def _options(text: str, allowed: set[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for pattern, option in OPTION_PATTERNS:
        if option not in allowed:
            continue
        for match in pattern.finditer(text):
            found.append((match.start(), option))
    return list(dict.fromkeys(option for _, option in sorted(found)))


def _clarification(question: str, *, unknown: list[str] | None = None) -> OrderInterpretation:
    return OrderInterpretation(
        intent="clarification",
        items=[],
        unknown_terms=unknown or [],
        missing_fields=["menu"],
        needs_clarification=True,
        clarification_question=question,
        confidence=0.35,
        summary="",
    )


def interpret_order_rules(utterance: str) -> tuple[OrderInterpretation, dict]:
    original = re.sub(r"\s+", " ", utterance).strip()
    corrected = correct_dialect(original)
    occurrences = _menu_occurrences(original)

    if not occurrences:
        if any(marker in original for marker in NON_ORDER_MARKERS):
            result = OrderInterpretation(
                intent="non_order",
                items=[],
                needs_clarification=False,
                confidence=0.98,
                summary="",
            )
        elif any(marker in original for marker in GENERIC_MENU_MARKERS):
            result = _clarification("어떤 메뉴를 원하시는지 다시 말씀해 주세요.")
        else:
            result = _clarification(
                "메뉴 이름을 다시 말씀해 주세요.",
                unknown=[original],
            )
        return result, {"corrected_text": corrected, "engine": "rules"}

    items: list[OrderItem] = []
    global_all = bool(re.search(r"(?:전부|모두|둘\s*다|세\s*개\s*다)", original))
    for index, (start, _, menu) in enumerate(occurrences):
        next_start = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(original)
        previous_end = occurrences[index - 1][1] if index else 0
        local_text = original[previous_end:next_start]
        if index == len(occurrences) - 1:
            local_text = original[previous_end:]

        quantity = _last_quantity(local_text)
        allowed = set(menu["allowed_options"])
        options = _options(local_text, allowed)
        if global_all:
            options = list(dict.fromkeys([*options, *_options(original, allowed)]))

        items.append(
            OrderItem(
                menu_id=menu["id"],
                menu_name=menu["name"],
                quantity=quantity,
                options=options,
            )
        )

    merged: dict[tuple[str, tuple[str, ...]], OrderItem] = {}
    for item in items:
        key = (item.menu_id, tuple(sorted(item.options)))
        if key in merged:
            merged[key].quantity += item.quantity
        else:
            merged[key] = item
    items = list(merged.values())

    summaries = []
    for item in items:
        option_text = f" ({', '.join(item.options)})" if item.options else ""
        summaries.append(f"{item.menu_name} {item.quantity}개{option_text}")

    result = OrderInterpretation(
        intent="order",
        items=items,
        unknown_terms=[],
        missing_fields=[],
        needs_clarification=False,
        clarification_question=None,
        confidence=0.96,
        summary=", ".join(summaries) + "를 주문합니다.",
    )
    return result, {"corrected_text": corrected, "engine": "rules"}
