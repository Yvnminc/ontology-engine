"""Domain Schema loader and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ontology_engine.core.errors import ConfigError
from ontology_engine.core.schema_format import (
    DomainSchemaModel, EntityTypeDefinition, ExtractionConfig,
    LinkTypeDefinition, SeedEntityDefinition, ValidationRule,
)


def _simplified_chinese(text: str) -> str:
    """Best-effort Traditional → Simplified Chinese conversion.

    Uses a small built-in table covering common characters found in
    business / meeting contexts.  No external dependency required.
    """
    _T2S: dict[str, str] = {
        "總": "总", "陳": "陈", "標": "标", "長": "长", "東": "东",
        "國": "国", "開": "开", "門": "门", "學": "学", "電": "电",
        "車": "车", "機": "机", "問": "问", "題": "题", "員": "员",
        "報": "报", "設": "设", "備": "备", "計": "计", "劃": "划",
        "業": "业", "務": "务", "產": "产", "質": "质", "廠": "厂",
        "處": "处", "運": "运", "輸": "输", "關": "关", "係": "系",
        "聯": "联", "絡": "络", "醫": "医", "師": "师", "場": "场",
        "據": "据", "數": "数", "會": "会", "議": "议", "決": "决",
        "訂": "订", "單": "单", "價": "价", "銷": "销", "購": "购",
        "買": "买", "賣": "卖", "經": "经", "濟": "济", "發": "发",
        "達": "达", "歐": "欧", "彥": "彦", "寶": "宝", "樣": "样",
        "種": "种", "類": "类", "體": "体", "個": "个", "條": "条",
        "點": "点", "時": "时", "間": "间", "動": "动", "進": "进",
        "連": "连", "構": "构", "結": "结", "節": "节", "線": "线",
        "圖": "图", "網": "网", "戶": "户", "訊": "讯", "導": "导",
        "級": "级", "項": "项", "實": "实", "驗": "验", "層": "层",
        "戰": "战", "術": "术", "軟": "软", "對": "对", "雙": "双",
        "面": "面", "資": "资",
    }
    return "".join(_T2S.get(ch, ch) for ch in text)


class DomainSchema:
    """Loaded domain schema with convenience accessors."""

    def __init__(self, model: DomainSchemaModel):
        self._model = model
        self._entity_map = {et.name: et for et in model.entity_types}
        self._link_map = {lt.name: lt for lt in model.link_types}

        # Build seed-entity lookup tables  (alias → canonical name)
        self._seed_alias_map: dict[str, str] = {}
        self._seed_entity_map: dict[str, SeedEntityDefinition] = {}
        for se in model.seed_entities:
            key = _simplified_chinese(se.name).strip().lower()
            self._seed_alias_map[key] = se.name
            self._seed_entity_map[se.name] = se
            for alias in se.aliases:
                akey = _simplified_chinese(alias).strip().lower()
                self._seed_alias_map[akey] = se.name

    @classmethod
    def from_yaml(cls, path: str | Path) -> DomainSchema:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Schema file not found: {p}")
        if p.suffix not in (".yaml", ".yml"):
            raise ConfigError(f"Expected YAML file, got: {p.suffix}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"Schema YAML must be a mapping, got {type(raw).__name__}")
        try:
            model = DomainSchemaModel.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"Schema validation failed for {p}: {exc}") from exc
        return cls(model)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainSchema:
        return cls(DomainSchemaModel.model_validate(data))

    @property
    def domain(self) -> str: return self._model.domain
    @property
    def version(self) -> str: return self._model.version
    @property
    def description(self) -> str: return self._model.description
    @property
    def model(self) -> DomainSchemaModel: return self._model
    @property
    def entity_types(self) -> list[EntityTypeDefinition]: return self._model.entity_types
    @property
    def link_types(self) -> list[LinkTypeDefinition]: return self._model.link_types
    @property
    def extraction(self) -> ExtractionConfig: return self._model.extraction
    @property
    def validation_rules(self) -> list[ValidationRule]: return self._model.validation_rules

    def get_entity_type(self, name: str) -> EntityTypeDefinition | None:
        return self._entity_map.get(name)

    def get_link_type(self, name: str) -> LinkTypeDefinition | None:
        return self._link_map.get(name)

    def entity_type_names(self) -> list[str]:
        return [et.name for et in self._model.entity_types]

    def link_type_names(self) -> list[str]:
        return [lt.name for lt in self._model.link_types]

    def has_entity_type(self, name: str) -> bool: return name in self._entity_map
    def has_link_type(self, name: str) -> bool: return name in self._link_map

    # --- Seed Entities API ------------------------------------------------

    def get_seed_entities(self) -> list[SeedEntityDefinition]:
        """Return all seed entities defined in the schema."""
        return list(self._model.seed_entities)

    def resolve_name(self, raw_name: str) -> str | None:
        """Resolve *raw_name* to its canonical name via seed-entity aliases.

        Matching is **case-insensitive** and **Traditional → Simplified
        Chinese normalised**.  Returns ``None`` when no match is found.
        """
        key = _simplified_chinese(raw_name).strip().lower()
        return self._seed_alias_map.get(key)

    def get_seed_entity(self, canonical_name: str) -> SeedEntityDefinition | None:
        """Look up a seed entity by its canonical name."""
        return self._seed_entity_map.get(canonical_name)

    def __repr__(self) -> str:
        return (f"DomainSchema(domain='{self.domain}', version='{self.version}', "
                f"entity_types={len(self.entity_types)}, link_types={len(self.link_types)})")


class SchemaRegistry:
    """Singleton registry for domain schemas."""
    _instance: SchemaRegistry | None = None
    _initialized: bool = False

    def __new__(cls) -> SchemaRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not SchemaRegistry._initialized:
            self._schemas: dict[str, DomainSchema] = {}
            self._active: str | None = None
            SchemaRegistry._initialized = True

    def register(self, schema: DomainSchema) -> None:
        self._schemas[schema.domain] = schema
        if self._active is None:
            self._active = schema.domain

    def register_from_yaml(self, path: str | Path) -> DomainSchema:
        s = DomainSchema.from_yaml(path)
        self.register(s)
        return s

    def get(self, domain: str) -> DomainSchema | None: return self._schemas.get(domain)

    def get_active(self) -> DomainSchema | None:
        return self._schemas.get(self._active) if self._active else None

    def set_active(self, domain: str) -> None:
        if domain not in self._schemas:
            raise ConfigError(f"Domain '{domain}' not registered")
        self._active = domain

    def list_domains(self) -> list[dict[str, Any]]:
        return [{"domain": n, "version": s.version, "description": s.description,
                 "entity_types": len(s.entity_types), "link_types": len(s.link_types),
                 "active": n == self._active} for n, s in self._schemas.items()]

    def has_domain(self, d: str) -> bool: return d in self._schemas

    def unregister(self, domain: str) -> None:
        if domain in self._schemas:
            del self._schemas[domain]
            if self._active == domain:
                self._active = next(iter(self._schemas), None)

    def clear(self) -> None:
        self._schemas.clear(); self._active = None

    @classmethod
    def reset(cls) -> None:
        cls._instance = None; cls._initialized = False

    def __len__(self) -> int: return len(self._schemas)
