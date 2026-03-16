"""Stage 2: 4-Pass structured extraction.

Pass 1: Entity extraction (NER + LLM)
Pass 2: Relation extraction
Pass 3: Decision / Event extraction
Pass 4: Action item extraction
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from ontology_engine.core.config import OntologyConfig
from ontology_engine.core.types import (
    DecisionType,
    EntityType,
    ExtractionResult,
    ExtractedActionItem,
    ExtractedDecision,
    ExtractedEntity,
    ExtractedLink,
    LinkType,
)
from ontology_engine.llm.base import LLMProvider
from ontology_engine.pipeline.preprocessor import ProcessedMeeting, Segment


class StructuredExtractor:
    """4-Pass structured extraction pipeline."""

    def __init__(self, llm: LLMProvider, config: OntologyConfig):
        self.llm = llm
        self.config = config

    async def extract(self, meeting: ProcessedMeeting) -> ExtractionResult:
        """Run all 4 passes and aggregate results."""
        result = ExtractionResult(
            source_file=meeting.metadata.get("source_file", ""),
            meeting_date=meeting.meeting_date,
            participants=meeting.participants,
            extraction_model=self.llm.config.model,
        )

        # Prepare text blocks (group segments by topic for coherence)
        text_blocks = self._prepare_blocks(meeting.segments)

        # Pass 1: Entity extraction
        if self.config.pipeline.enable_pass1_entities:
            result.entities = await self._pass1_entities(text_blocks, meeting)

        # Pass 2: Relation extraction (needs Pass 1 results)
        if self.config.pipeline.enable_pass2_relations:
            result.links = await self._pass2_relations(text_blocks, result.entities)

        # Pass 3: Decision extraction
        if self.config.pipeline.enable_pass3_decisions:
            result.decisions = await self._pass3_decisions(text_blocks, meeting)

        # Pass 4: Action item extraction
        if self.config.pipeline.enable_pass4_actions:
            result.action_items = await self._pass4_actions(
                text_blocks, result.entities, result.decisions
            )

        return result

    def _prepare_blocks(self, segments: list[Segment]) -> list[str]:
        """Group segments into manageable text blocks for LLM processing."""
        blocks: list[str] = []
        current_block: list[str] = []
        current_len = 0
        max_block_len = 3000  # ~3K chars per block to stay within context

        for seg in segments:
            line = f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text
            if current_len + len(line) > max_block_len and current_block:
                blocks.append("\n".join(current_block))
                current_block = []
                current_len = 0
            current_block.append(line)
            current_len += len(line) + 1

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    # =========================================================================
    # Pass 1: Entity Extraction
    # =========================================================================

    async def _pass1_entities(
        self, blocks: list[str], meeting: ProcessedMeeting
    ) -> list[ExtractedEntity]:
        """Extract entities (Person, Project, Organization, Technology, etc.)."""
        known_aliases = self.config.known_entities.aliases
        known_list = ", ".join(known_aliases.keys()) if known_aliases else "无"

        all_entities: list[ExtractedEntity] = []

        for block in blocks:
            prompt = f"""从以下会议内容中抽取所有实体（人物、项目、组织、技术/工具）。

已知实体库：{known_list}

会议内容：
{block}

输出 JSON：
{{
  "entities": [
    {{
      "name": "标准化名称",
      "type": "Person|Project|Risk|Deadline",
      "aliases": ["会议中出现的变体名称"],
      "confidence": 0.95,
      "is_new": false,
      "context": "原文出处（简短引用）",
      "properties": {{}}
    }}
  ]
}}

规则：
1. 人名优先匹配已知实体库
2. 新实体标注 is_new: true
3. 只抽取明确提到的实体，不推测
4. type 限定为: Person, Project, Risk, Deadline"""

            try:
                data = await self.llm.generate_json(prompt)
                for ent_data in data.get("entities", []):
                    try:
                        entity_type = EntityType(ent_data.get("type", "Person"))
                    except ValueError:
                        continue  # Skip unknown types
                    ent = ExtractedEntity(
                        name=ent_data.get("name", ""),
                        entity_type=entity_type,
                        aliases=ent_data.get("aliases", []),
                        confidence=ent_data.get("confidence", 0.8),
                        is_new=ent_data.get("is_new", False),
                        context=ent_data.get("context", ""),
                        properties=ent_data.get("properties", {}),
                    )
                    if ent.name and ent.confidence >= self.config.pipeline.min_confidence:
                        all_entities.append(ent)
            except Exception:
                continue

        return self._deduplicate_entities(all_entities)

    # =========================================================================
    # Pass 2: Relation Extraction
    # =========================================================================

    async def _pass2_relations(
        self, blocks: list[str], entities: list[ExtractedEntity]
    ) -> list[ExtractedLink]:
        """Extract relationships between entities."""
        entity_names = [e.name for e in entities]
        entity_list = ", ".join(entity_names) if entity_names else "无"

        all_links: list[ExtractedLink] = []

        # Process blocks in pairs for context continuity
        for block in blocks:
            prompt = f"""从以下会议内容中抽取实体之间的关系。

已知实体：{entity_list}

会议内容：
{block}

可用关系类型：
- participates_in: 人物 → 项目
- makes: 人物 → 决策
- assigned_to: 行动项 → 人物
- relates_to: 决策 → 项目
- reports_to: 人物 → 人物
- collaborates_with: 人物 → 人物
- owns: 人物 → 项目

输出 JSON：
{{
  "relations": [
    {{
      "type": "participates_in",
      "source": "源实体名称",
      "target": "目标实体名称",
      "confidence": 0.85,
      "context": "原文引用"
    }}
  ]
}}

规则：
1. 只抽取明确提到或强烈暗示的关系
2. source 和 target 必须是已知实体列表中的名称
3. 不推测不确定的关系"""

            try:
                data = await self.llm.generate_json(prompt)
                for rel_data in data.get("relations", []):
                    try:
                        link_type = LinkType(rel_data.get("type", ""))
                    except ValueError:
                        continue
                    link = ExtractedLink(
                        link_type=link_type,
                        source_name=rel_data.get("source", ""),
                        target_name=rel_data.get("target", ""),
                        confidence=rel_data.get("confidence", 0.8),
                        context=rel_data.get("context", ""),
                    )
                    if (
                        link.source_name
                        and link.target_name
                        and link.confidence >= self.config.pipeline.min_confidence
                    ):
                        all_links.append(link)
            except Exception:
                continue

        return all_links

    # =========================================================================
    # Pass 3: Decision Extraction
    # =========================================================================

    async def _pass3_decisions(
        self, blocks: list[str], meeting: ProcessedMeeting
    ) -> list[ExtractedDecision]:
        """Extract decisions, risks, and milestones."""
        all_decisions: list[ExtractedDecision] = []

        for block in blocks:
            prompt = f"""从以下会议内容中抽取所有决策。

会议日期：{meeting.meeting_date or '未知'}
参会人：{', '.join(meeting.participants) or '未知'}

会议内容：
{block}

决策识别标志：
- 明确决策: "我们决定…", "就这么定了", "OK那就…", "方向确定了"
- 隐性决策: CEO/负责人说 "我觉得应该…" → 高概率决策
- 否定决策: "这个不做", "先不考虑", "砍掉"
- 条件决策: "如果…就…", "等…再…"

输出 JSON：
{{
  "decisions": [
    {{
      "summary": "决策摘要（一句话）",
      "detail": "详细描述",
      "decision_type": "strategic|tactical|operational",
      "made_by": "决策人姓名",
      "participants": ["参与讨论者"],
      "rationale": "决策依据",
      "conditions": "生效条件（如有）",
      "confidence": 0.85,
      "source_segment": "原文引用"
    }}
  ]
}}

规则：
1. 只抽取实际做出的决策，不包括建议或讨论
2. 每个决策必须有明确的决策人
3. confidence 反映决策的确定程度"""

            try:
                data = await self.llm.generate_json(prompt)
                for dec_data in data.get("decisions", []):
                    try:
                        dt = DecisionType(dec_data.get("decision_type", "operational"))
                    except ValueError:
                        dt = DecisionType.OPERATIONAL
                    dec = ExtractedDecision(
                        summary=dec_data.get("summary", ""),
                        detail=dec_data.get("detail", ""),
                        decision_type=dt,
                        made_by=dec_data.get("made_by", ""),
                        participants=dec_data.get("participants", []),
                        rationale=dec_data.get("rationale", ""),
                        conditions=dec_data.get("conditions", ""),
                        confidence=dec_data.get("confidence", 0.8),
                        source_segment=dec_data.get("source_segment", ""),
                    )
                    if dec.summary and dec.confidence >= self.config.pipeline.min_confidence:
                        all_decisions.append(dec)
            except Exception:
                continue

        return all_decisions

    # =========================================================================
    # Pass 4: Action Item Extraction
    # =========================================================================

    async def _pass4_actions(
        self,
        blocks: list[str],
        entities: list[ExtractedEntity],
        decisions: list[ExtractedDecision],
    ) -> list[ExtractedActionItem]:
        """Extract action items (who does what by when)."""
        person_names = [e.name for e in entities if e.entity_type == EntityType.PERSON]
        decision_summaries = [d.summary for d in decisions[:10]]

        all_actions: list[ExtractedActionItem] = []

        for block in blocks:
            prompt = f"""从以下会议内容中抽取所有行动项（TODO/任务）。

已知人物：{', '.join(person_names) or '未知'}
已知决策：{json.dumps(decision_summaries, ensure_ascii=False) if decision_summaries else '无'}

会议内容：
{block}

行动项识别标志：
- 命令式: "你去做…", "帮我…", "负责…"
- 承诺式: "我来处理", "我今天搞定", "我跟进一下"
- 分配式: "这个交给XX", "XX你来负责"
- 截止式: "周五之前", "下周一要…"

输出 JSON：
{{
  "action_items": [
    {{
      "task": "任务描述",
      "owner": "负责人姓名",
      "assignees": ["执行人"],
      "due_date": "YYYY-MM-DD 或 null",
      "priority": "high|medium|low",
      "related_decision": "关联的决策摘要（如有）",
      "completion_criteria": "完成标准",
      "confidence": 0.85,
      "source_segment": "原文引用"
    }}
  ]
}}

规则：
1. 必须有明确的负责人
2. 模糊的讨论不算行动项
3. 如果提到时间，转换为具体日期"""

            try:
                data = await self.llm.generate_json(prompt)
                for act_data in data.get("action_items", []):
                    due = act_data.get("due_date")
                    due_date = None
                    if due and due != "null":
                        try:
                            due_date = date.fromisoformat(due)
                        except ValueError:
                            pass

                    act = ExtractedActionItem(
                        task=act_data.get("task", ""),
                        owner=act_data.get("owner", ""),
                        assignees=act_data.get("assignees", []),
                        due_date=due_date,
                        priority=act_data.get("priority", "medium"),
                        related_decision=act_data.get("related_decision"),
                        completion_criteria=act_data.get("completion_criteria", ""),
                        confidence=act_data.get("confidence", 0.8),
                        source_segment=act_data.get("source_segment", ""),
                    )
                    if act.task and act.confidence >= self.config.pipeline.min_confidence:
                        all_actions.append(act)
            except Exception:
                continue

        return all_actions

    # =========================================================================
    # Deduplication
    # =========================================================================

    def _deduplicate_entities(
        self, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Merge duplicate entities by name / alias overlap."""
        seen: dict[str, ExtractedEntity] = {}

        for ent in entities:
            key = ent.name.lower().strip()

            # Check if this is a known alias of an existing entity
            merged = False
            for existing_key, existing in seen.items():
                if key == existing_key:
                    # Same name — merge aliases and keep higher confidence
                    existing.aliases = list(set(existing.aliases + ent.aliases))
                    existing.confidence = max(existing.confidence, ent.confidence)
                    merged = True
                    break
                # Check alias overlap
                existing_aliases = {a.lower() for a in existing.aliases}
                new_aliases = {a.lower() for a in ent.aliases} | {key}
                if existing_aliases & new_aliases:
                    existing.aliases = list(
                        set(existing.aliases + ent.aliases + [ent.name])
                    )
                    existing.confidence = max(existing.confidence, ent.confidence)
                    merged = True
                    break

            if not merged:
                seen[key] = ent

        return list(seen.values())
