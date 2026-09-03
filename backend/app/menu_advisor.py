from __future__ import annotations

import re

from .menu_store import MenuStore
from .schemas import MenuRecommendation, OrderInterpretation


INQUIRY_PATTERN = re.compile(
    r"(뭐\s*(?:뭐)?\s*(?:있|파)|어떤\s*(?:거|것|메뉴)|메뉴\s*(?:알려|보여)|"
    r"메뉴\s*(?:뭐꼬|먹고)|"
    r"뭐가\s*(?:좋|괜찮|맛있)|추천|골라|먹을\s*만한|마실\s*만한|맛있|뭔데|뭐고|무슨\s*뜻|"
    r"잘\s*모르|설명해|있나|있어요|있습니까)"
)
RECOMMENDATION_PATTERN = re.compile(
    r"(추천|골라|뭐가\s*(?:좋|괜찮)|먹을\s*만한|마실\s*만한|"
    r"잠을?\s*못|잠이\s*안|민감|안\s*센|부담\s*(?:없|적)|잘\s*모르)"
)
ORDER_CUE_PATTERN = re.compile(
    r"(?:\d+|한|두|세|네|다섯|하나|둘|셋)\s*(?:개|잔|그릇|병)?"
    r".{0,18}(?:주이소|주세요|주라|주소|주서요|시키|주문)"
)
CAFFEINE_FREE_PATTERN = re.compile(
    r"(카페인\s*(?:없는|안\s*든|0|제로)|무카페인|카페인\s*아예\s*없는)"
)
DECAF_PATTERN = re.compile(
    r"(디카페인|디카페|디카\s*페인|뒤\s*카페인|카페인\s*(?:적은|덜한|낮은)|"
    r"카페인에?\s*민감|커피.*잠을?\s*못|잠이\s*안\s*오)"
)
HOT_PATTERN = re.compile(r"(따뜻|뜨끈|뜨거|차갑지\s*않)")
COLD_PATTERN = re.compile(r"(차갑|시원|아이스|찬\s)")
SWEET_PATTERN = re.compile(r"(달달|달콤|단\s*(?:거|것|음료))")
NOT_SWEET_PATTERN = re.compile(r"(안\s*단|달지\s*않|덜\s*달|단\s*거\s*말고)")
COFFEE_PATTERN = re.compile(r"(커피|아메리카노|라떼|아아|뜨아|카페)")
POPULARITY_PATTERN = re.compile(
    r"(뭐가?\s*잘\s*나가|잘\s*나가는\s*(?:거|것|메뉴)|"
    r"가장\s*많이\s*(?:팔|나가)|제일\s*많이\s*(?:팔|나가)|인기\s*메뉴)"
)
OWNER_RECOMMENDATION_PATTERN = re.compile(
    r"(사장님|사장|주인장|점장님|매장)\s*(?:의\s*)?(?:추천|픽)"
)
TASTE_REQUEST_PATTERN = re.compile(
    r"(뭐가?\s*맛있|맛있는\s*(?:거|것|메뉴)|입맛에\s*맞|맛으로\s*추천)"
)
TASTE_PREFERENCE_PATTERN = re.compile(
    r"((?:먹|묵|마시)(?:고|을|을라)?\s*싶|(?:먹|묵|마시)고\s*싶|땡기)"
)
TASTE_TAG_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("달콤함", SWEET_PATTERN),
    ("고소함", re.compile(r"(고소|꼬소)")),
    ("담백함", re.compile(r"(담백|슴슴|자극적이지\s*않|덜\s*자극)")),
    ("진한 맛", re.compile(r"(진한\s*맛|진하게|얼큰|깊은\s*맛)")),
    ("깔끔함", re.compile(r"(깔끔|개운|산뜻)")),
    ("새콤함", re.compile(r"(새콤|상큼|시큼)")),
)
SUGGESTED_TASTE_TAGS = [tag for tag, _ in TASTE_TAG_RULES]


def _selected_taste_tags(utterance: str) -> list[str]:
    return [
        tag
        for tag, pattern in TASTE_TAG_RULES
        if pattern.search(utterance)
    ]


def _recommendation(menu: dict, reason: str) -> MenuRecommendation:
    return MenuRecommendation(
        menu_id=menu["id"],
        menu_name=menu["name"],
        price=menu["price"],
        description=menu["description"],
        reason=reason,
        caffeine_level=menu.get("caffeine_level", "unknown"),
        caffeine_note=menu.get("caffeine_note", ""),
        temperature_tags=menu.get("temperature_tags", []),
        taste_tags=menu.get("taste_tags", []),
        allowed_options=menu.get("allowed_options", []),
    )


def _reason(menu: dict, utterance: str) -> str:
    level = menu.get("caffeine_level")
    reasons: list[str] = []
    if DECAF_PATTERN.search(utterance) and level == "decaf":
        reasons.append("디카페인 원두를 사용하는 메뉴")
    elif CAFFEINE_FREE_PATTERN.search(utterance) and level == "none":
        reasons.append("카페인이 들어가지 않는 메뉴")
    if HOT_PATTERN.search(utterance) and "따뜻함" in menu["temperature_tags"]:
        reasons.append("따뜻하게 마실 수 있음")
    if COLD_PATTERN.search(utterance) and "차가움" in menu["temperature_tags"]:
        reasons.append("차갑게 마실 수 있음")
    if SWEET_PATTERN.search(utterance) and "달콤함" in menu["taste_tags"]:
        reasons.append("달콤한 맛")
    if NOT_SWEET_PATTERN.search(utterance) and "달콤함" not in menu["taste_tags"]:
        reasons.append("기본적으로 달지 않은 편")
    if not reasons:
        reasons.append(menu["description"])
    return " · ".join(reasons)


def _filter_candidates(
    menus: list[dict],
    utterance: str,
    *,
    excluded_ingredients: set[str] | None = None,
    required_ingredients: set[str] | None = None,
) -> list[dict]:
    excluded_ingredients = excluded_ingredients or set()
    required_ingredients = required_ingredients or set()
    candidates = [
        menu
        for menu in menus
        if not excluded_ingredients.intersection(menu.get("ingredients", []))
        and required_ingredients.issubset(set(menu.get("ingredients", [])))
    ]
    wants_zero = bool(CAFFEINE_FREE_PATTERN.search(utterance))
    wants_decaf = bool(DECAF_PATTERN.search(utterance))
    wants_coffee = bool(COFFEE_PATTERN.search(utterance))

    if wants_zero:
        candidates = [
            menu
            for menu in candidates
            if menu.get("caffeine_level") == "none"
            and menu["category"] in {"차", "음료"}
        ]
    elif wants_decaf:
        candidates = [
            menu for menu in candidates if menu.get("caffeine_level") == "decaf"
        ]
    elif wants_coffee:
        candidates = [menu for menu in candidates if menu["category"] == "커피"]

    if HOT_PATTERN.search(utterance):
        candidates = [
            menu for menu in candidates if "따뜻함" in menu["temperature_tags"]
        ]
    elif COLD_PATTERN.search(utterance):
        candidates = [
            menu for menu in candidates if "차가움" in menu["temperature_tags"]
        ]

    if NOT_SWEET_PATTERN.search(utterance):
        candidates = [
            menu for menu in candidates if "달콤함" not in menu["taste_tags"]
        ]
    elif SWEET_PATTERN.search(utterance):
        candidates = [
            menu for menu in candidates if "달콤함" in menu["taste_tags"]
        ]

    selected_tastes = _selected_taste_tags(utterance)
    if selected_tastes:
        candidates = [
            menu
            for menu in candidates
            if any(tag in menu.get("taste_tags", []) for tag in selected_tastes)
        ]

    def score(menu: dict) -> tuple[int, int, str]:
        value = 0
        if menu.get("caffeine_level") == "decaf" and wants_decaf:
            value += 20
        if menu.get("caffeine_level") == "none" and wants_zero:
            value += 20
        if "어르신" in menu.get("audience_tags", []):
            value += 2
        return (-value, menu["price"], menu["name"])

    return sorted(candidates, key=score)[:3]


def _explicit_menu_explanation(
    store: MenuStore,
    utterance: str,
) -> OrderInterpretation | None:
    results = store.search(utterance, limit=3)
    explicit = []
    for result in results:
        names = {result.menu["name"], *result.menu["aliases"]}
        if names.intersection(result.matched_terms):
            explicit.append(result.menu)
    if len(explicit) != 1:
        return None

    menu = explicit[0]
    caffeine = menu.get("caffeine_note", "")
    answer = f"{menu['name']}은(는) {menu['description']}입니다."
    if caffeine:
        answer += f" {caffeine}"
    recommendation = _recommendation(menu, "질문하신 메뉴")
    return OrderInterpretation(
        intent="menu_inquiry",
        standard_order=answer,
        items=[],
        recommendations=[recommendation],
        needs_clarification=True,
        missing_fields=["order_confirmation"],
        clarification_question=f"{menu['name']}으로 주문할까요?",
        confidence=0.98,
        summary=answer,
    )


def advise_menu(
    utterance: str,
    store: MenuStore,
) -> OrderInterpretation | None:
    normalized = utterance.strip()
    if not normalized:
        return None

    if POPULARITY_PATTERN.search(normalized):
        answer = (
            "아직 판매량 정보가 연결되지 않아 가장 많이 팔린 메뉴를 "
            "정확히 말씀드리기는 어렵습니다."
        )
        return OrderInterpretation(
            intent="menu_inquiry",
            standard_order=answer,
            items=[],
            recommendations=[],
            needs_clarification=False,
            confidence=1.0,
            summary=answer,
        )

    if OWNER_RECOMMENDATION_PATTERN.search(normalized):
        answer = "사장님 추천 메뉴는 아직 등록되지 않았습니다."
        return OrderInterpretation(
            intent="menu_inquiry",
            standard_order=answer,
            items=[],
            recommendations=[],
            needs_clarification=False,
            confidence=1.0,
            summary=answer,
        )

    if (
        TASTE_REQUEST_PATTERN.search(normalized)
        and not _selected_taste_tags(normalized)
    ):
        answer = "맛은 취향에 따라 달라서 좋아하시는 맛을 먼저 알려주시면 좋습니다."
        return OrderInterpretation(
            intent="recommendation",
            standard_order=answer,
            items=[],
            recommendations=[],
            suggested_tags=SUGGESTED_TASTE_TAGS,
            needs_clarification=True,
            missing_fields=["taste_preference"],
            clarification_question="어떤 맛을 좋아하세요?",
            confidence=1.0,
            summary=answer,
        )

    has_taste_preference = bool(
        _selected_taste_tags(normalized)
        and TASTE_PREFERENCE_PATTERN.search(normalized)
    )

    if not INQUIRY_PATTERN.search(normalized) and not has_taste_preference:
        return None

    # “디카페인 아메리카노 한 잔 주세요”처럼 메뉴와 수량이 확정된 문장은
    # 문의가 아니라 기존 주문 해석기로 넘긴다.
    if ORDER_CUE_PATTERN.search(normalized) and not RECOMMENDATION_PATTERN.search(
        normalized
    ):
        return None

    explanation = _explicit_menu_explanation(store, normalized)
    if explanation is not None and re.search(r"(뭔데|뭐고|무슨\s*뜻|설명)", normalized):
        return explanation

    menus = store.list_menus(available_only=True)
    candidates = _filter_candidates(
        menus,
        normalized,
        excluded_ingredients=set(store.negated_ingredients(normalized)),
        required_ingredients=set(store.required_ingredients(normalized)),
    )
    intent = (
        "recommendation"
        if RECOMMENDATION_PATTERN.search(normalized) or has_taste_preference
        else "menu_inquiry"
    )

    if not candidates:
        answer = "말씀하신 조건에 맞는 판매 메뉴를 찾지 못했습니다."
        if CAFFEINE_FREE_PATTERN.search(normalized):
            answer += (
                " 디카페인도 미량의 카페인이 남을 수 있어 "
                "카페인이 전혀 없는 음료와는 구분해서 확인해야 합니다."
            )
        return OrderInterpretation(
            intent=intent,
            standard_order=answer,
            items=[],
            recommendations=[],
            needs_clarification=True,
            missing_fields=["menu"],
            clarification_question="원하시는 온도나 맛을 조금 더 말씀해 주시겠어요?",
            confidence=0.92,
            summary=answer,
        )

    recommendations = [
        _recommendation(menu, _reason(menu, normalized)) for menu in candidates
    ]
    if DECAF_PATTERN.search(normalized):
        answer = "카페인을 줄인 메뉴를 찾았습니다."
        answer += " 디카페인에도 미량의 카페인이 남을 수 있습니다."
    elif CAFFEINE_FREE_PATTERN.search(normalized):
        answer = "카페인이 들어가지 않는 메뉴를 찾았습니다."
    else:
        answer = "말씀하신 조건에 맞는 메뉴를 찾았습니다."

    return OrderInterpretation(
        intent=intent,
        standard_order=answer,
        items=[],
        recommendations=recommendations,
        needs_clarification=True,
        missing_fields=["menu_selection"],
        clarification_question="어느 메뉴로 주문할까요?",
        confidence=0.96,
        summary=answer,
    )
