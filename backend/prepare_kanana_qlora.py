from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from app.menu_store import MenuStore
from app.ollama_client import (
    KANANA_OUTPUT_GUIDE,
    KANANA_SYSTEM_PROMPT,
    _canonical_order,
    _kanana_retrieval_payload,
)


ROOT = Path(__file__).resolve().parent


def _bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _order_answer(case: dict, store: MenuStore) -> str:
    expected = case["expected"]
    intent = expected["intent"]
    if intent == "non_order":
        return "NON_ORDER|"
    if intent == "clarification":
        return "CLARIFY|어떤 메뉴를 원하시는지 다시 말씀해 주시겠어요?"

    items = []
    for expected_item in expected.get("items", []):
        menu = store.get_menu(expected_item["menu_id"])
        if menu is None:
            raise ValueError(f"알 수 없는 메뉴: {expected_item['menu_id']}")
        items.append(
            {
                "menu_id": menu["id"],
                "menu_name": menu["name"],
                "quantity": expected_item["quantity"],
                "options": expected_item.get("options", []),
            }
        )
    return f"ORDER|{_canonical_order(items)}"


def _order_example(case: dict, store: MenuStore) -> dict:
    utterance = case["utterance"]
    candidates = _kanana_retrieval_payload(store.search(utterance, limit=5))
    user = (
        f"{KANANA_OUTPUT_GUIDE}\n\n"
        f"주문 발화: {utterance}\n"
        "사용 가능한 메뉴 후보:\n"
        f"{json.dumps(candidates, ensure_ascii=False)}"
    )
    return {
        "id": case["id"],
        "source": "synthetic_order",
        "category": case["category"],
        "messages": [
            {"role": "system", "content": KANANA_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _order_answer(case, store)},
        ],
    }


def _dialect_example(row: dict, source: str) -> dict:
    dialect = row["dialect_text"].strip()
    standard = row["standard_text"].strip()
    return {
        "id": row["utterance_id"],
        "source": source,
        "category": "dialect_normalization",
        "messages": [
            {
                "role": "system",
                "content": (
                    "경상도 방언 문장의 의미를 보존해 자연스러운 표준어로 바꾸세요. "
                    "설명 없이 변환한 문장만 답하세요."
                ),
            },
            {"role": "user", "content": dialect},
            {"role": "assistant", "content": standard},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\AIData\malhaeye\kanana-2-3b-qlora-cafe-v2"),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--order-cases",
        type=Path,
        default=ROOT / "data" / "order_cases_cafe_200.json",
    )
    parser.add_argument(
        "--menu-path",
        type=Path,
        default=ROOT / "data" / "menu_cafe.json",
    )
    args = parser.parse_args()

    store = MenuStore(args.menu_path, args.output_dir / "menu.sqlite3")

    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = defaultdict(list)

    order_cases = json.loads(
        args.order_cases.read_text(encoding="utf-8")
    )
    by_category: dict[str, list[dict]] = defaultdict(list)
    for case in order_cases:
        by_category[case["category"]].append(case)
    ordered_categories = sorted(by_category)
    validation_counts = {
        category: int(len(by_category[category]) * 0.1)
        for category in ordered_categories
    }
    validation_target = round(len(order_cases) * 0.1)
    remaining_validation = validation_target - sum(validation_counts.values())
    remainders = sorted(
        ordered_categories,
        key=lambda category: (
            -(len(by_category[category]) * 0.1 % 1), category
        ),
    )
    for category in remainders[:remaining_validation]:
        validation_counts[category] += 1

    for category in ordered_categories:
        cases = by_category[category]
        rng.shuffle(cases)
        train_end = int(len(cases) * 0.8)
        val_end = train_end + validation_counts[category]
        for split, subset in (
            ("train", cases[:train_end]),
            ("validation", cases[train_end:val_end]),
            ("test", cases[val_end:]),
        ):
            splits[split].extend(_order_example(case, store) for case in subset)

    for source_name in ("aihub_119", "aihub_71517"):
        path = ROOT / "data" / source_name / "utterances.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            dialect = row.get("dialect_text", "").strip()
            standard = row.get("standard_text", "").strip()
            if (
                not dialect
                or not standard
                or dialect == standard
                or len(dialect) > 180
                or len(standard) > 180
            ):
                continue
            value = _bucket(f"{source_name}:{row['conversation_id']}")
            split = "train" if value < 8 else "validation" if value == 8 else "test"
            splits[split].append(_dialect_example(row, source_name))

    summary = {"seed": args.seed, "splits": {}, "sources": {}}
    for split in ("train", "validation", "test"):
        rng.shuffle(splits[split])
        _write_jsonl(args.output_dir / f"{split}.jsonl", splits[split])
        summary["splits"][split] = len(splits[split])
        counts: dict[str, int] = defaultdict(int)
        for row in splits[split]:
            counts[row["source"]] += 1
        summary["sources"][split] = dict(sorted(counts.items()))

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
