from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT / "data" / "menu_cafe.json"
DEFAULT_DB_PATH = ROOT / "data" / "menu.sqlite3"

TAG_FIELDS = (
    "taste_tags",
    "temperature_tags",
    "texture_tags",
    "audience_tags",
    "ingredients",
    "keywords",
)

CONCEPT_PATTERNS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("달달", "달콤", "단 거", "단것"), ("달콤함",)),
    (("뜨끈", "따뜻", "뜨거", "온기"), ("따뜻함", "뜨거움")),
    (("시원", "차갑", "찬 거", "찬것"), ("차가움",)),
    (("안 맵", "맵지", "애들", "아이", "어린이"), ("맵지 않음", "어린이")),
    (("부드럽", "이가 안", "이가 영", "씹기 힘", "연한 거"), ("부드러움",)),
    (("속 편", "부담 없", "소화 잘"), ("속 편한 식사", "부드러움", "담백함")),
    (("든든", "배부", "한 끼", "한끼"), ("든든한 식사",)),
    (("국물", "해장"), ("뜨끈한 국물", "해장")),
    (("쌉싸", "쓴 거", "커피"), ("쓴맛", "커피")),
)

def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    return parsed if isinstance(parsed, list) else []


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", text.lower())


def _catalog_hash(catalog: dict[str, Any]) -> str:
    payload = json.dumps(catalog, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _negated_ingredients(query: str, ingredients: list[str]) -> set[str]:
    negated: set[str] = set()
    negative = (
        r"(?:싫|(?:못|몬)\s*(?:먹|묵)|빼|제외|알레르기|"
        r"없|안\s*(?:들|든|들어))"
    )
    for ingredient in ingredients:
        escaped = re.escape(ingredient)
        after = rf"{escaped}(?:은|는|이|가|을|를|도)?\s*{negative}"
        before_allergy = rf"알레르기(?:가|는)?\s*(?:있|있어|있어서)?\s*{escaped}"
        if re.search(after, query) or re.search(before_allergy, query):
            negated.add(ingredient)
    return negated


def _required_ingredients(query: str, ingredients: list[str]) -> set[str]:
    required: set[str] = set()
    positive = r"(?:있|들어|든|넣|함유)"
    for ingredient in ingredients:
        escaped = re.escape(ingredient)
        pattern = (
            rf"{escaped}(?:은|는|이|가|을|를|도)?\s*"
            rf"(?:이|가)?\s*{positive}"
        )
        if re.search(pattern, query):
            required.add(ingredient)
    return required


@dataclass(frozen=True)
class MenuSearchResult:
    menu: dict[str, Any]
    score: float
    matched_terms: tuple[str, ...]

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.menu["id"],
            "name": self.menu["name"],
            "price": self.menu["price"],
            "description": self.menu["description"],
            "caffeine_level": self.menu["caffeine_level"],
            "caffeine_note": self.menu["caffeine_note"],
            "allowed_options": self.menu["allowed_options"],
            "matched_terms": list(self.matched_terms),
            "retrieval_score": round(self.score, 3),
        }

    def as_api_dict(self) -> dict[str, Any]:
        return {
            **self.as_prompt_dict(),
            "category": self.menu["category"],
            "taste_tags": self.menu["taste_tags"],
            "temperature_tags": self.menu["temperature_tags"],
            "texture_tags": self.menu["texture_tags"],
            "available": self.menu["available"],
        }


class MenuStore:
    """JSON seed catalog backed by a small operational SQLite database."""

    def __init__(
        self,
        catalog_path: Path | str = DEFAULT_CATALOG_PATH,
        db_path: Path | str | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        configured_db = os.getenv("MENU_DB_PATH")
        self.db_path = Path(db_path or configured_db or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS menus (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    price INTEGER NOT NULL CHECK(price >= 0),
                    description TEXT NOT NULL,
                    available INTEGER NOT NULL DEFAULT 1,
                    aliases_json TEXT NOT NULL,
                    allowed_options_json TEXT NOT NULL,
                    taste_tags_json TEXT NOT NULL,
                    temperature_tags_json TEXT NOT NULL,
                    texture_tags_json TEXT NOT NULL,
                    audience_tags_json TEXT NOT NULL,
                    ingredients_json TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    caffeine_level TEXT NOT NULL DEFAULT 'unknown',
                    caffeine_note TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_menus_available
                ON menus(available);

                CREATE INDEX IF NOT EXISTS idx_menus_category
                ON menus(category);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(menus)").fetchall()
            }
            if "caffeine_level" not in columns:
                connection.execute(
                    "ALTER TABLE menus ADD COLUMN caffeine_level TEXT NOT NULL DEFAULT 'unknown'"
                )
            if "caffeine_note" not in columns:
                connection.execute(
                    "ALTER TABLE menus ADD COLUMN caffeine_note TEXT NOT NULL DEFAULT ''"
                )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'catalog_hash'"
            ).fetchone()
            expected_ids = {
                str(menu["id"])
                for menu in catalog.get("menus", [])
            }
            stored_ids = {
                str(item["id"])
                for item in connection.execute("SELECT id FROM menus").fetchall()
            }
            if (
                row is None
                or row["value"] != _catalog_hash(catalog)
                or stored_ids != expected_ids
            ):
                self._sync_catalog(connection, catalog)
            connection.commit()

    def _sync_catalog(
        self,
        connection: sqlite3.Connection,
        catalog: dict[str, Any],
    ) -> None:
        # The JSON catalog is the source of truth. Remove rows from an older
        # catalog so changing the service domain does not leave stale menus
        # searchable in SQLite.
        connection.execute("DELETE FROM menus")
        for menu in catalog.get("menus", []):
            connection.execute(
                """
                INSERT INTO menus (
                    id, name, category, price, description, available,
                    aliases_json, allowed_options_json, taste_tags_json,
                    temperature_tags_json, texture_tags_json,
                    audience_tags_json, ingredients_json, keywords_json,
                    caffeine_level, caffeine_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    price = excluded.price,
                    description = excluded.description,
                    aliases_json = excluded.aliases_json,
                    allowed_options_json = excluded.allowed_options_json,
                    taste_tags_json = excluded.taste_tags_json,
                    temperature_tags_json = excluded.temperature_tags_json,
                    texture_tags_json = excluded.texture_tags_json,
                    audience_tags_json = excluded.audience_tags_json,
                    ingredients_json = excluded.ingredients_json,
                    keywords_json = excluded.keywords_json,
                    caffeine_level = excluded.caffeine_level,
                    caffeine_note = excluded.caffeine_note
                """,
                (
                    menu["id"],
                    menu["name"],
                    menu.get("category", "기타"),
                    int(menu["price"]),
                    menu.get("description", ""),
                    int(bool(menu.get("available", True))),
                    _json_dumps(menu.get("aliases", [])),
                    _json_dumps(menu.get("allowed_options", [])),
                    _json_dumps(menu.get("taste_tags", [])),
                    _json_dumps(menu.get("temperature_tags", [])),
                    _json_dumps(menu.get("texture_tags", [])),
                    _json_dumps(menu.get("audience_tags", [])),
                    _json_dumps(menu.get("ingredients", [])),
                    _json_dumps(menu.get("keywords", [])),
                    menu.get("caffeine_level", "unknown"),
                    menu.get("caffeine_note", ""),
                ),
            )
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES('catalog_hash', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (_catalog_hash(catalog),),
        )
        connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES('store_name', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (catalog.get("store_name", ""),),
        )

    @staticmethod
    def _row_to_menu(row: sqlite3.Row) -> dict[str, Any]:
        menu = {
            "id": row["id"],
            "name": row["name"],
            "category": row["category"],
            "price": row["price"],
            "description": row["description"],
            "available": bool(row["available"]),
            "caffeine_level": row["caffeine_level"],
            "caffeine_note": row["caffeine_note"],
            "aliases": _json_loads(row["aliases_json"]),
            "allowed_options": _json_loads(row["allowed_options_json"]),
        }
        for field in TAG_FIELDS:
            menu[field] = _json_loads(row[f"{field}_json"])
        return menu

    def list_menus(self, *, available_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM menus"
        if available_only:
            query += " WHERE available = 1"
        query += " ORDER BY category, name"
        with closing(self._connect()) as connection:
            rows = connection.execute(query).fetchall()
        return [self._row_to_menu(row) for row in rows]

    def get_menu(self, menu_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM menus WHERE id = ?",
                (menu_id,),
            ).fetchone()
        return self._row_to_menu(row) if row else None

    def set_availability(self, menu_id: str, available: bool) -> bool:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE menus SET available = ? WHERE id = ?",
                    (int(available), menu_id),
                )
                changed = cursor.rowcount > 0
        return changed

    @staticmethod
    def _query_concepts(query: str) -> list[set[str]]:
        normalized = _normalize(query)
        concepts: list[set[str]] = []
        for patterns, tags in CONCEPT_PATTERNS:
            if any(_normalize(pattern) in normalized for pattern in patterns):
                concepts.append(set(tags))
        return concepts

    def search(self, query: str, *, limit: int = 5) -> list[MenuSearchResult]:
        normalized_query = _normalize(query)
        if not normalized_query:
            return []

        concepts = self._query_concepts(query)
        has_allergy_marker = "알레르기" in query
        results: list[MenuSearchResult] = []

        for menu in self.list_menus(available_only=True):
            if re.search(r"(?:디카페인|디카페|카페인\s*(?:적은|덜한))", query):
                if menu.get("caffeine_level") == "regular":
                    continue
            score = 0.0
            matches: set[str] = set()

            excluded_ingredients = _negated_ingredients(query, menu["ingredients"])
            removable_options = {
                option.removesuffix(" 빼기")
                for option in menu["allowed_options"]
                if option.endswith(" 빼기")
            }
            unsafe_exclusions = {
                ingredient
                for ingredient in excluded_ingredients
                if has_allergy_marker or ingredient not in removable_options
            }
            if unsafe_exclusions:
                continue

            names = [menu["name"], *menu["aliases"]]
            for name in names:
                normalized_name = _normalize(name)
                if normalized_name and normalized_name in normalized_query:
                    score += 20.0 if name == menu["name"] else 16.0
                    matches.add(name)

            searchable_tags = {
                value
                for field in TAG_FIELDS
                for value in menu.get(field, [])
            }
            for concept_group in concepts:
                matched_concepts = concept_group.intersection(searchable_tags)
                if matched_concepts:
                    score += 5.0
                    matches.update(matched_concepts)

            free_text_fields = [
                menu["category"],
                menu["description"],
                *menu["keywords"],
                *menu["ingredients"],
            ]
            for text in free_text_fields:
                normalized_text = _normalize(text)
                if len(normalized_text) >= 2 and normalized_text in normalized_query:
                    score += 2.0
                    matches.add(text)

            if score > 0:
                results.append(
                    MenuSearchResult(
                        menu=menu,
                        score=score,
                        matched_terms=tuple(sorted(matches)),
                    )
                )

        results.sort(key=lambda item: (-item.score, item.menu["price"], item.menu["name"]))
        return results[: max(1, min(limit, 10))]

    def negated_ingredients(self, query: str) -> list[str]:
        ingredients = sorted(
            {
                ingredient
                for menu in self.list_menus(available_only=True)
                for ingredient in menu["ingredients"]
            }
        )
        return sorted(_negated_ingredients(query, ingredients))

    def required_ingredients(self, query: str) -> list[str]:
        ingredients = sorted(
            {
                ingredient
                for menu in self.list_menus(available_only=True)
                for ingredient in menu["ingredients"]
            }
        )
        return sorted(_required_ingredients(query, ingredients))

    def status(self) -> dict[str, Any]:
        menus = self.list_menus(available_only=False)
        return {
            "database": str(self.db_path),
            "catalog": str(self.catalog_path),
            "menu_count": len(menus),
            "available_count": sum(1 for menu in menus if menu["available"]),
            "retrieval": "closed-menu-hybrid",
        }


menu_store = MenuStore()
