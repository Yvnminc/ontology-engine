"""Stage 1: Meeting transcript preprocessing.

Handles: text cleaning, filler word removal, topic segmentation,
coreference resolution, temporal normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ontology_engine.core.config import OntologyConfig
from ontology_engine.llm.base import LLMProvider


@dataclass
class Segment:
    """A topically coherent segment of a meeting transcript."""

    text: str
    speaker: str = ""
    topic: str = ""
    start_offset: int = 0
    end_offset: int = 0


@dataclass
class ProcessedMeeting:
    """Output of the preprocessing stage."""

    segments: list[Segment] = field(default_factory=list)
    raw_text: str = ""
    cleaned_text: str = ""
    meeting_date: date | None = None
    participants: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# Chinese filler words and verbal tics
FILLER_PATTERNS = [
    r"[，。]?\s*(?:嗯|啊|呃|那个|就是说|怎么说呢|然后就是)\s*[，。]?",
    r"[，。]?\s*(?:对对对|是是是|好好好|ok\s*ok|嗯嗯嗯?)\s*[，。]?",
    r"[，。]?\s*(?:其实就是|所以说|反正就是)\s*[，。]?",
]
FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)

# Speaker label patterns (common in transcripts)
SPEAKER_RE = re.compile(
    r"^(?:【(.+?)】|(\w{1,10})\s*[:：])\s*",
    re.MULTILINE,
)


class MeetingPreprocessor:
    """Stage 1 of the pipeline: clean and segment meeting transcripts."""

    def __init__(self, llm: LLMProvider, config: OntologyConfig):
        self.llm = llm
        self.config = config

    async def process(
        self,
        raw_text: str,
        meeting_date: date | None = None,
        participants: list[str] | None = None,
    ) -> ProcessedMeeting:
        """Run the full preprocessing pipeline."""
        # Step 1: Basic text cleaning
        cleaned = self._clean_text(raw_text)

        # Step 2: Speaker diarization (from transcript labels)
        speaker_segments = self._extract_speakers(cleaned)

        # Step 3: Topic segmentation via LLM
        if self.config.pipeline.segment_topics and len(cleaned) > 500:
            segments = await self._segment_topics(speaker_segments, cleaned)
        else:
            segments = speaker_segments

        # Step 4: Coreference resolution via LLM
        if self.config.pipeline.resolve_coreferences:
            segments = await self._resolve_coreferences(segments, participants or [])

        return ProcessedMeeting(
            segments=segments,
            raw_text=raw_text,
            cleaned_text=cleaned,
            meeting_date=meeting_date,
            participants=participants or self._detect_participants(speaker_segments),
        )

    def _clean_text(self, text: str) -> str:
        """Remove filler words and normalize whitespace."""
        if not self.config.pipeline.remove_filler_words:
            return text

        text = FILLER_RE.sub("", text)
        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _extract_speakers(self, text: str) -> list[Segment]:
        """Split text into segments by speaker labels."""
        segments: list[Segment] = []
        last_end = 0
        last_speaker = ""

        for m in SPEAKER_RE.finditer(text):
            # Save previous segment
            if last_end < m.start() and last_speaker:
                seg_text = text[last_end : m.start()].strip()
                if seg_text:
                    segments.append(
                        Segment(
                            text=seg_text,
                            speaker=last_speaker,
                            start_offset=last_end,
                            end_offset=m.start(),
                        )
                    )

            last_speaker = m.group(1) or m.group(2) or ""
            last_end = m.end()

        # Last segment
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append(
                    Segment(
                        text=remaining,
                        speaker=last_speaker,
                        start_offset=last_end,
                        end_offset=len(text),
                    )
                )

        # If no speaker labels found, return whole text as one segment
        if not segments:
            segments = [Segment(text=text, start_offset=0, end_offset=len(text))]

        return segments

    async def _segment_topics(
        self, segments: list[Segment], full_text: str
    ) -> list[Segment]:
        """Use LLM to identify topic boundaries and label segments."""
        # Prepare condensed text for LLM
        condensed = "\n".join(
            f"[{i}] {s.speaker}: {s.text[:200]}" for i, s in enumerate(segments)
        )

        prompt = f"""你是一个会议记录分析器。请分析以下会议内容，将其划分为不同的讨论话题。

会议内容（每行格式：[编号] 说话人: 内容摘要）：
{condensed}

请输出 JSON 格式：
{{
  "topics": [
    {{
      "topic": "话题名称（简短）",
      "segment_ids": [0, 1, 2]
    }}
  ]
}}

要求：
- 每个话题用一个简短的中文标签描述
- segment_ids 是上面内容的编号列表
- 一个 segment 只属于一个话题"""

        try:
            result = await self.llm.generate_json(
                prompt, model=self.llm.config.fast_model
            )
            topics = result.get("topics", [])
            # Apply topic labels to segments
            for topic in topics:
                for sid in topic.get("segment_ids", []):
                    if 0 <= sid < len(segments):
                        segments[sid].topic = topic.get("topic", "")
        except Exception:
            # If LLM fails, keep segments without topic labels
            pass

        return segments

    async def _resolve_coreferences(
        self, segments: list[Segment], known_participants: list[str]
    ) -> list[Segment]:
        """Use LLM to resolve pronouns to actual names."""
        # Only process segments with pronouns
        pronoun_pattern = re.compile(r"[他她它们这那][们的]?|[这那]个")
        segments_with_pronouns = [
            (i, s) for i, s in enumerate(segments) if pronoun_pattern.search(s.text)
        ]

        if not segments_with_pronouns:
            return segments

        # Build context window (include surrounding segments)
        texts = "\n".join(
            f"[{i}] {s.speaker}: {s.text}" for i, s in segments_with_pronouns[:20]
        )

        known = ", ".join(known_participants) if known_participants else "未知"

        prompt = f"""请对以下会议内容进行代词消解（将"他/她/这个/那个"替换为实际指代的名词）。

已知参会人：{known}

原文：
{texts}

请输出 JSON 格式：
{{
  "resolved": [
    {{
      "id": 0,
      "text": "消解后的文本"
    }}
  ]
}}

规则：
- 只替换能确定指代对象的代词
- 不确定的保持原样
- 保持原文风格"""

        try:
            result = await self.llm.generate_json(
                prompt, model=self.llm.config.fast_model
            )
            for item in result.get("resolved", []):
                idx = item.get("id")
                resolved_text = item.get("text", "")
                if idx is not None and resolved_text:
                    # Map back to original segment index
                    orig_idx = segments_with_pronouns[idx][0] if idx < len(segments_with_pronouns) else None
                    if orig_idx is not None:
                        segments[orig_idx].text = resolved_text
        except Exception:
            pass

        return segments

    def _detect_participants(self, segments: list[Segment]) -> list[str]:
        """Extract unique speaker names from segments."""
        speakers = set()
        for s in segments:
            if s.speaker:
                speakers.add(s.speaker)
        return sorted(speakers)
