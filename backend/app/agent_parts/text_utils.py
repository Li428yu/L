from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.agent_parts.state import (
    DocumentProfile,
    PaperAgentState,
    ParsedTask,
    ReadingTaskContract,
)
from backend.app.models import EvidenceItem, RetrievalDebugItem, RuntimeStep


PDF_TEXT_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u00ad": "",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}


class AgentTextUtilityMixin:
    def _extract_field(self, text: str, field_name: str) -> str:
        pattern = rf"{re.escape(field_name)}\s*[:：]\s*([^。；;\n]+?)(?=\s*(实验类型|指导教师|专业班级|姓名|学号|电子邮件|实\s*验\s*日\s*期|一、|二、|$))"
        match = re.search(pattern, text)
        if not match:
            return ""
        return match.group(1).strip()

    def _guess_title_from_text(self, text: str) -> str:
        if "端口扫描" in text:
            return "端口扫描实验"
        if "TCP通信" in text or "TCP 通信" in text or "回显" in text:
            return "TCP 通信实验"
        return "当前文档主题"

    def _extract_after_heading(self, text: str, heading: str) -> str:
        pattern = rf"{re.escape(heading)}\s*([^一二三四五六七八九十、]+)"
        match = re.search(pattern, text)
        if not match:
            return ""
        return match.group(1).strip()

    def _sanitize_evidence_text(self, text: str) -> str:
        sanitized = str(text or "")
        for source, target in PDF_TEXT_REPLACEMENTS.items():
            sanitized = sanitized.replace(source, target)
        sanitized = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", sanitized)
        sanitized = re.sub(r"[\w.\-+]+@[\w.\-]+\.\w+", "[邮箱已隐藏]", sanitized)
        sanitized = re.sub(r"(姓\s*名|姓名)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        sanitized = re.sub(r"(学\s*号|学号)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        sanitized = re.sub(r"(电子邮件|邮箱)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        return sanitized

    def _looks_like_visual_question_text(self, question: str) -> bool:
        normalized = " ".join(question.lower().split())
        if not normalized:
            return False

        strong_visual_markers = [
            "图片",
            "截图",
            "运行截图",
            "图表",
            "图中",
            "图里",
            "图上",
            "图示",
            "架构图",
            "示意图",
            "视觉证据",
            "图片证据",
            "看图",
            "figure",
            "fig.",
            "chart",
            "diagram",
            "screenshot",
            "visual evidence",
            "image evidence",
            "what does the image",
            "what does the figure",
            "shown in the image",
            "shown in the figure",
            "in the image",
            "in the figure",
            "from the image",
            "from the figure",
        ]
        if any(marker in normalized for marker in strong_visual_markers):
            return True

        modality_pair_markers = [
            "图像-文本",
            "图像文本",
            "图像/文本",
            "图像与文本",
            "图像和文本",
            "图文",
            "image-text",
            "image text",
            "image/text",
            "image and text",
        ]
        data_context_markers = [
            "数据",
            "数据集",
            "规模",
            "训练",
            "预训练",
            "对",
            "dataset",
            "datasets",
            "data",
            "training",
            "pretraining",
            "pre-training",
            "pairs",
            "pair",
        ]
        if any(marker in normalized for marker in modality_pair_markers) and any(
            marker in normalized for marker in data_context_markers
        ):
            return False

        return any(marker in normalized for marker in ["图像", "视觉", "image", "visual"])

    def _evidence_page_label(self, item: EvidenceItem) -> str:
        page_start = item.page_start or item.page
        page_end = item.page_end or page_start
        if page_start and page_end and page_end != page_start:
            return f"{page_start}-{page_end}"
        return str(page_start or item.page or 0)

    def _first_informative_overview_row(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        for row in rows[:12]:
            text = str(row.get("text", ""))
            if self._looks_like_front_matter_noise(text):
                continue
            if self._overview_structure_score(text) > 0 or self._readable_text_score(text) >= 0.55:
                return row
        for row in rows:
            text = str(row.get("text", ""))
            if not self._looks_like_front_matter_noise(text):
                return row
        return None

    def _overview_structure_score(self, text: str) -> float:
        normalized = " ".join(self._sanitize_evidence_text(text).lower().split())
        if not normalized:
            return 0.0
        score = 0.0
        strong_markers = [
            "abstract",
            "introduction",
            "conclusion",
            "in this paper",
            "in this work",
            "we propose",
            "we present",
            "we introduce",
            "we show",
            "本文围绕",
            "摘要",
            "结论",
            "研究认为",
        ]
        medium_markers = [
            "state-of-the-art",
            "architecture",
            "model",
            "training",
            "method",
            "experiment",
            "evaluation",
            "dataset",
            "benchmark",
            "result",
            "实验",
            "模型",
            "方法",
            "评测",
            "数据",
            "结果",
        ]
        score += sum(0.22 for marker in strong_markers if marker in normalized)
        score += sum(0.08 for marker in medium_markers if marker in normalized)
        return min(score, 0.8)

    def _looks_like_front_matter_noise(self, text: str) -> bool:
        normalized = " ".join(self._sanitize_evidence_text(text).lower().split())
        if not normalized:
            return True
        if any(marker in normalized for marker in ["abstract", "introduction", "we propose", "in this work"]):
            return False
        noise_markers = [
            "provided proper attribution",
            "hereby grants permission",
            "journalistic or scholarly works",
            "work performed while at",
            "conference on neural information processing systems",
            "arxiv:",
        ]
        if any(marker in normalized for marker in noise_markers):
            return True
        front_labels = [
            "论文题目",
            "题目",
            "作者",
            "单位",
            "学院",
            "学校",
            "专业",
            "班级",
            "姓名",
            "学号",
            "指导教师",
            "完成日期",
            "日期",
            "关键词",
            "关键字",
        ]
        front_label_hits = sum(1 for marker in front_labels if marker in normalized)
        body_markers = ["本文围绕", "本文旨在", "本文提出", "本研究", "研究认为", "in this paper", "in this work"]
        if front_label_hits >= 2 and len(normalized) <= 360 and not any(marker in normalized for marker in body_markers):
            return True
        if re.fullmatch(r"(摘\s*要|摘要|abstract)[:：]?", normalized, flags=re.IGNORECASE):
            return True
        if (
            any(marker in normalized for marker in ["作者", "单位", "日期"])
            and any(marker in normalized for marker in ["摘 要", "摘要"])
            and not any(marker in normalized for marker in ["本文围绕", "研究认为", "in this paper", "in this work"])
            and len(normalized) <= 260
        ):
            return True
        hidden_email_count = normalized.count("[邮箱已隐藏]")
        if hidden_email_count >= 3 and not any(marker in normalized for marker in ["abstract", "introduction"]):
            return True
        return False

    def _trim_front_matter_prefix(self, text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        cleaned = re.sub(
            r"^\d+(?:\.\d+)*\s+(?:abstract|introduction|background|conclusion|results?|model|experiments?)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        markers = [
            "Abstract ",
            "摘要",
            "1 Introduction ",
            "Introduction ",
            "In this work ",
            "In this paper ",
        ]
        for marker in markers:
            index = cleaned.find(marker)
            if 0 < index <= 260:
                if marker in {"Abstract ", "摘要", "1 Introduction ", "Introduction "}:
                    return cleaned[index + len(marker) :].strip()
                return cleaned[index:].strip()
        return cleaned

    def _truncate_readable_text(self, text: str, limit: int = 220) -> str:
        cleaned = " ".join(text.split()).strip()
        if len(cleaned) <= limit:
            return cleaned
        boundary = max(
            cleaned.rfind("。", 0, limit),
            cleaned.rfind("！", 0, limit),
            cleaned.rfind("？", 0, limit),
            cleaned.rfind(".", 0, limit),
            cleaned.rfind(";", 0, limit),
            cleaned.rfind(" ", 0, limit),
        )
        if boundary < int(limit * 0.55):
            boundary = limit
        return cleaned[:boundary].rstrip(" ,，;；.") + "..."

    def _question_relevance_score(self, question: str, text: str) -> float:
        normalized_text = " ".join(self._sanitize_evidence_text(text).lower().split())
        tokens = self._question_keywords(question)
        if not normalized_text:
            return 0.0

        hits = sum(1 for token in tokens if token.lower() in normalized_text)
        exact_score = hits / max(len(tokens), 1) if tokens else 0.0

        phrases = self._question_keyphrases(question)
        phrase_text = normalized_text.replace("-", " ")
        phrase_hits = sum(1 for phrase in phrases if phrase in phrase_text)
        phrase_score = phrase_hits / max(min(len(phrases), 6), 1) if phrases else 0.0

        q_chars = re.findall(r"[\u4e00-\u9fff]", question)
        t_chars = set(re.findall(r"[\u4e00-\u9fff]", normalized_text))
        char_score = 0.0
        if q_chars:
            meaningful_chars = [
                char
                for char in q_chars
                if char not in set("这篇份个的了呢吗啊和与及或是有在中里上下一些哪些什么怎么如何请问")
            ]
            if meaningful_chars:
                char_hits = sum(1 for char in meaningful_chars if char in t_chars)
                char_score = char_hits / max(len(meaningful_chars), 1)

        return max(0.0, min(1.0, exact_score * 0.55 + phrase_score * 0.3 + char_score * 0.15))

    def _question_keywords(self, question: str) -> list[str]:
        normalized = question.lower()
        stop_phrases = [
            "这篇论文",
            "这份文档",
            "这个文档",
            "这篇文档",
            "这份报告",
            "请你",
            "请问",
            "给我",
            "一下",
            "什么",
            "哪些",
            "为什么",
            "怎么样",
            "如何",
            "是否",
            "能否",
            "可以",
        ]
        for phrase in stop_phrases:
            normalized = normalized.replace(phrase, " ")

        domain_terms = [
            "参考文献",
            "引用文献",
            "结论",
            "结果",
            "方法",
            "研究方法",
            "实验方法",
            "局限",
            "不足",
            "风险",
            "贡献",
            "创新",
            "数据",
            "样本",
            "问卷",
            "访谈",
            "实验",
            "模型",
            "算法",
            "公式",
            "表格",
            "作者",
            "标题",
            "主题",
            "目的",
        ]
        tokens = [term for term in domain_terms if term in question]
        tokens.extend(re.findall(r"[a-z0-9]{2,}", normalized))

        cjk_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
        for sequence in cjk_sequences:
            if len(sequence) <= 4:
                tokens.append(sequence)
                continue
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
            tokens.extend(sequence[index : index + 3] for index in range(len(sequence) - 2))

        blocked = {"论文", "文档", "报告", "内容", "主要", "一个", "这个", "那个"}
        unique: list[str] = []
        for token in tokens:
            cleaned = token.strip()
            if len(cleaned) < 2 or cleaned in blocked:
                continue
            if cleaned not in unique:
                unique.append(cleaned)
        return unique[:40]

    def _question_keyphrases(self, question: str) -> list[str]:
        normalized = " ".join(self._sanitize_evidence_text(question).lower().split())
        words = re.findall(r"[a-z0-9][a-z0-9\-]*", normalized)
        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "how",
            "does",
            "did",
            "is",
            "are",
            "was",
            "were",
            "the",
            "this",
            "that",
            "paper",
            "document",
            "report",
            "reported",
            "describe",
            "describes",
            "claim",
            "claims",
            "about",
            "with",
            "from",
            "compared",
            "previous",
            "use",
            "uses",
            "using",
        }
        content = [word for word in words if len(word) >= 3 and word not in stopwords]
        phrases: list[str] = []

        def add(value: str) -> None:
            cleaned = " ".join(value.replace("-", " ").split())
            if len(cleaned) < 5 or cleaned in phrases:
                return
            phrases.append(cleaned)

        for size in range(4, 1, -1):
            for start in range(0, max(len(content) - size + 1, 0)):
                phrase_words = content[start : start + size]
                if not any(len(word) >= 5 for word in phrase_words):
                    continue
                add(" ".join(phrase_words))
                if len(phrases) >= 18:
                    return phrases
        return phrases

    def _looks_like_reference_section_text(self, text: str) -> bool:
        normalized = " ".join(text.split())
        if re.search(r"\breferences\b", normalized, flags=re.IGNORECASE):
            return True
        return "参考文献" in normalized and self._reference_marker_count(normalized) > 0

    def _looks_like_reference_continuation(self, text: str) -> bool:
        normalized = " ".join(text.split())
        if self._reference_marker_count(normalized) > 0:
            return True
        return bool(re.search(r"(教育|研究|Journal|Higher Education|University).{0,40}\d{4}", normalized))

    def _reference_marker_count(self, text: str) -> int:
        return len(re.findall(r"\[\d{1,3}\]", text))

    def _extract_references_from_text(self, text: str) -> list[tuple[str, str]]:
        sanitized = self._sanitize_evidence_text(text)
        normalized = re.sub(r"\s+", " ", sanitized).strip()
        section_match = re.search(r"(参考文献|References)\s*", normalized, flags=re.IGNORECASE)
        section_text = normalized[section_match.start() :] if section_match else normalized
        matches = list(
            re.finditer(
                r"\[(\d{1,3})\]\s*(.*?)(?=\s*\[\d{1,3}\]\s*|$)",
                section_text,
                flags=re.DOTALL,
            )
        )
        references_by_number: dict[int, str] = {}
        for match in matches:
            number = int(match.group(1))
            content = match.group(2).strip(" ，,。；;")
            content = re.sub(r"^(参考文献|References)\s*", "", content, flags=re.IGNORECASE).strip()
            content = re.split(
                r"\s*(?:（说明：|说明：|附录[:：]|设计维度\s*\||Appendix\b)",
                content,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            content = re.sub(r"\s+", " ", content)
            if len(content) < 6:
                continue
            references_by_number[number] = content[:500]
        return [
            (str(number), content)
            for number, content in sorted(references_by_number.items())
        ]

    def _pick_readable_sentences(self, text: str, limit: int) -> list[str]:
        parts = re.split(r"(?<=[。！？.!?])\s+|(?<=。)|(?<=！)|(?<=？)", text)
        sentences: list[str] = []
        blocked = ["姓名", "学号", "电子邮件", "邮箱", "实验评分"]
        for part in parts:
            cleaned = self._trim_front_matter_prefix(part.strip())
            if len(cleaned) < 18:
                continue
            if self._looks_like_front_matter_noise(cleaned):
                continue
            if any(word in cleaned for word in blocked):
                continue
            if self._is_table_like_text(cleaned):
                continue
            sentences.append(self._truncate_readable_text(cleaned, limit=220))
            if len(sentences) >= limit:
                break
        return sentences

    def _looks_like_table_question(self, question: str) -> bool:
        presentation_only = any(phrase in question for phrase in ["适合表格", "可以用表格", "用表格", "表格呈现"])
        evidence_keywords = [
            "评分表",
            "实验评分",
            "分数构成",
            "分数",
            "总分",
            "表格中",
            "表中",
            "结构化数据",
            "数据是多少",
            "指标",
            "数值",
            "公式",
            "计算",
        ]
        if presentation_only and not any(keyword in question for keyword in evidence_keywords):
            return False
        keywords = evidence_keywords
        return any(keyword in question for keyword in keywords)

    def _reliability_relevance_score(self, text: str) -> int:
        weighted_keywords = [
            ("课程报告", 8),
            ("实验报告", 7),
            ("随机生成", 8),
            ("论文样稿", 8),
            ("毕业论文", 6),
            ("学位论文", 6),
            ("摘要", 3),
            ("参考文献", 4),
            ("数据来源", 4),
            ("实证数据", 4),
            ("未来研究", 4),
            ("文献分析", 3),
            ("情境推演", 3),
            ("机制建构", 3),
            ("一致性检验", 3),
            ("后评价", 3),
            ("结论", 2),
            ("结果", 2),
            ("计算", 2),
            ("公式", 2),
            ("分析", 2),
            ("评价", 2),
            ("完成日期", 2),
            ("测试", 2),
            ("局限", 2),
            ("不足", 2),
        ]
        return sum(weight for keyword, weight in weighted_keywords if keyword in text)

    def _document_kind_from_evidence(self, evidence: list[EvidenceItem]) -> str:
        text = " ".join([item.paper_name for item in evidence] + [item.text for item in evidence[:6]])
        if "课程报告" in text:
            return "课程报告"
        if "实验报告" in text:
            return "实验报告"
        if "毕业论文" in text or "学位论文" in text:
            return "论文"
        if ("摘要" in text and "参考文献" in text) or "Abstract" in text:
            return "论文"
        return "普通文档"

    def _first_citation_with(
        self,
        evidence: list[EvidenceItem],
        keywords: list[str],
        *,
        fallback: bool = True,
    ) -> str:
        for item in evidence:
            text = f"{item.paper_name}\n{item.quote}\n{item.text}"
            if any(keyword in text for keyword in keywords):
                return item.citation_id
        return evidence[0].citation_id if fallback and evidence else ""

    def _first_citation_with_all(self, evidence: list[EvidenceItem], keywords: list[str]) -> str:
        for item in evidence:
            text = f"{item.paper_name}\n{item.quote}\n{item.text}"
            if all(keyword in text for keyword in keywords):
                return item.citation_id
        return ""

    def _join_citations(self, citation_ids: list[str]) -> str:
        unique_ids: list[str] = []
        for citation_id in citation_ids:
            if citation_id and citation_id not in unique_ids:
                unique_ids.append(citation_id)
        return f" {' '.join(f'[{citation_id}]' for citation_id in unique_ids)}" if unique_ids else ""

    def _best_readable_quote(self, text: str, limit: int = 220) -> str:
        sanitized = self._sanitize_evidence_text(text)
        normalized = " ".join(sanitized.split())
        if not normalized:
            return ""

        blocked = ["姓名", "学号", "电子邮件", "邮箱", "实验评分"]
        candidates = re.split(r"\n+|(?<=[。！？；;.!?])\s+", sanitized)
        for candidate in candidates:
            cleaned = " ".join(candidate.split()).strip()
            if len(cleaned) < 12:
                continue
            has_document_type = any(word in cleaned for word in ["课程报告", "实验报告", "毕业论文", "学位论文"])
            if any(word in cleaned for word in blocked) and not has_document_type:
                continue
            if self._is_table_like_text(cleaned):
                continue
            return cleaned[:limit] + ("..." if len(cleaned) > limit else "")

        if self._is_table_like_text(normalized):
            return self._best_table_quote_for_question("", sanitized, limit=min(limit, 180))
        return normalized[:limit] + ("..." if len(normalized) > limit else "")

    def _best_quote_for_question(self, question: str, text: str, limit: int = 240) -> str:
        sanitized = self._sanitize_evidence_text(text)
        if not sanitized.strip():
            return ""

        if self._looks_like_reference_question(question):
            return self._best_reference_quote(sanitized, limit=limit)
        if self._is_table_like_text(sanitized):
            return self._best_table_quote_for_question(question, sanitized, limit=limit)
        if self._looks_like_document_wide_question(question):
            preferred_keywords = self._overview_focus_keywords(question)
        else:
            preferred_keywords = self._question_keywords(question)

        quote = self._focused_sentence_quote(
            question=question,
            text=sanitized,
            preferred_keywords=preferred_keywords,
            limit=limit,
        )
        return quote or self._best_readable_quote(sanitized, limit=limit)

    def _focused_sentence_quote(
        self,
        *,
        question: str,
        text: str,
        preferred_keywords: list[str],
        limit: int,
        extra_terms: list[str] | None = None,
        bonus_phrases: list[str] | None = None,
    ) -> str:
        sentences = self._split_quote_sentences(text)
        if not sentences:
            return ""
        question_terms = self._quote_focus_terms(question, preferred_keywords, extra_terms or [])
        normalized_bonus = [term.lower() for term in bonus_phrases or [] if term]
        scored: list[tuple[float, int, str]] = []
        for position, sentence in enumerate(sentences):
            normalized = sentence.lower()
            term_hits = sum(1 for term in question_terms if self._quote_term_present(term, normalized))
            bonus_hits = sum(1 for term in normalized_bonus if term in normalized)
            number_bonus = 0.35 if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|m|b|k|million|billion)?\b", normalized) else 0.0
            method_bonus = 0.2 if any(term in normalized for term in ["we propose", "we present", "we show", "result", "dataset", "method"]) else 0.0
            score = term_hits + bonus_hits * 1.8 + number_bonus + method_bonus
            if score > 0:
                scored.append((score, position, sentence))
        if not scored:
            return ""
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        picked = sorted(scored[:2], key=lambda row: row[1])
        quote = " ".join(sentence for _, _, sentence in picked)
        return self._truncate_readable_text(quote, limit=limit)

    def _split_quote_sentences(self, text: str) -> list[str]:
        sanitized = self._sanitize_evidence_text(text)
        parts = [
            " ".join(part.split()).strip()
            for part in re.split(r"\n+|(?<=[.!?。！？；;])\s+", sanitized)
            if part.strip()
        ]
        result: list[str] = []
        for part in parts:
            if len(part) > 420 and re.search(r"[,，;；]", part):
                result.extend(
                    segment.strip()
                    for segment in re.split(r"(?<=[,，;；])\s+", part)
                    if len(segment.strip()) >= 20
                )
            else:
                result.append(part)
        return [part for part in result if len(part) >= 8 and not self._is_table_like_text(part)]

    def _quote_focus_terms(
        self,
        question: str,
        preferred_keywords: list[str],
        extra_terms: list[str],
    ) -> list[str]:
        terms: list[str] = []

        def add(term: str) -> None:
            lowered = term.strip().lower()
            stopwords = {
                "what",
                "which",
                "where",
                "when",
                "how",
                "does",
                "did",
                "is",
                "are",
                "was",
                "were",
                "the",
                "this",
                "that",
                "for",
                "and",
                "with",
                "from",
                "reported",
                "report",
            }
            if len(lowered) >= 2 and lowered not in stopwords and lowered not in terms:
                terms.append(lowered)

        for term in [*preferred_keywords, *extra_terms]:
            add(str(term))
        for phrase in self._question_keyphrases(question)[:8]:
            add(phrase)
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{1,}|[\u4e00-\u9fff]{2,}", question.lower()):
            add(token)
        role_detector = getattr(self, "_paper_structure_roles_for_question", None)
        term_getter = getattr(self, "_paper_structure_role_terms", None)
        if callable(role_detector) and callable(term_getter):
            for role in role_detector(question):
                for term in term_getter(role, kind="evidence")[:8]:
                    add(term)
        return terms[:32]

    def _best_table_quote_for_question(self, question: str, text: str, limit: int = 320) -> str:
        rows = self._split_table_rows(text)
        if not rows:
            normalized = " ".join(self._sanitize_evidence_text(text).split())
            return self._truncate_readable_text(normalized, limit=min(limit, 180))
        header = self._table_header_row(rows)
        focus_terms = self._quote_focus_terms(question, self._question_keywords(question), [])
        scored: list[tuple[float, int, str]] = []
        for position, row in enumerate(rows):
            if header and row == header:
                continue
            normalized = row.lower()
            term_hits = sum(1 for term in focus_terms if self._quote_term_present(term, normalized))
            number_hits = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|m|b|k|million|billion)?\b", normalized))
            metric_hits = sum(
                1
                for term in ["accuracy", "error", "score", "metric", "result", "auc", "f1", "%", "结果", "指标"]
                if term in normalized
            )
            score = term_hits + min(number_hits, 3) * 0.35 + metric_hits * 0.45
            if score > 0:
                scored.append((score, position, row))
        if not scored:
            scored = [
                (
                    len(re.findall(r"\d", row)) * 0.05 + (0.2 if "|" in row or "\t" in row else 0.0),
                    position,
                    row,
                )
                for position, row in enumerate(rows)
                if not header or row != header
            ]
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        picked_scored = scored[:1]
        if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.2:
            picked_scored.append(scored[1])
        picked_rows = [row for _, _, row in sorted(picked_scored, key=lambda item: item[1])]
        output_rows: list[str] = []
        if header:
            output_rows.append(header)
        for row in picked_rows:
            if row not in output_rows:
                output_rows.append(row)
        return self._truncate_readable_text(" | ".join(output_rows), limit=limit)

    def _quote_term_present(self, term: str, normalized_text: str) -> bool:
        if re.fullmatch(r"[a-z0-9][a-z0-9\-]*", term):
            return bool(re.search(rf"\b{re.escape(term)}\b", normalized_text))
        return term in normalized_text

    def _split_table_rows(self, text: str) -> list[str]:
        rows: list[str] = []
        for raw_line in self._sanitize_evidence_text(text).splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            if re.fullmatch(r"[:|\-\s]+", line):
                continue
            if "|" in line:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(" | ".join(cells))
                continue
            if "\t" in line:
                cells = [cell.strip() for cell in line.split("\t") if cell.strip()]
                if cells:
                    rows.append(" | ".join(cells))
                continue
            rows.append(line)
        if len(rows) <= 1:
            flattened = " ".join(self._sanitize_evidence_text(text).split())
            rows = [
                part.strip()
                for part in re.split(r"\s{2,}|(?<=\d)\s+(?=[A-Z][A-Za-z0-9 /-]{2,}\s+\d)", flattened)
                if part.strip()
            ]
        return rows

    def _table_header_row(self, rows: list[str]) -> str:
        for row in rows[:3]:
            normalized = row.lower()
            has_metric_word = any(term in normalized for term in ["metric", "score", "accuracy", "error", "model", "method", "dataset", "指标", "结果"])
            has_many_numbers = len(re.findall(r"\d", row)) >= 4
            if has_metric_word and not has_many_numbers:
                return row
        return ""

    def _best_reference_quote(self, text: str, limit: int = 240) -> str:
        references = self._extract_references_from_text(text)
        if references:
            snippet = "；".join(
                f"[{number}] {content}"
                for number, content in references[:2]
            )
            return snippet[:limit] + ("..." if len(snippet) > limit else "")

        normalized = " ".join(self._sanitize_evidence_text(text).split())
        marker_match = re.search(r"(参考文献|References).{0,220}", normalized, flags=re.IGNORECASE)
        if marker_match:
            quote = marker_match.group(0)
            return quote[:limit] + ("..." if len(quote) > limit else "")
        return self._best_readable_quote(normalized, limit=limit)

    def _readable_text_score(self, text: str) -> float:
        normalized = " ".join(text.split())
        if not normalized:
            return 0.0

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
        alpha_count = len(re.findall(r"[A-Za-z]", normalized))
        score = 0.2
        if cjk_count >= 30 or alpha_count >= 30:
            score += 0.25
        if any(char in normalized for char in "。！？.!?；;"):
            score += 0.2
        if 80 <= len(normalized) <= 1200:
            score += 0.15
        if self._is_table_like_text(normalized):
            score -= 0.35
        if any(word in normalized for word in ["姓名", "学号", "电子邮件", "邮箱"]):
            score -= 0.15
        return max(0.0, min(score, 1.0))

    def _is_table_like_text(self, text: str) -> bool:
        normalized = " ".join(text.split())
        if not normalized:
            return False

        pipe_count = normalized.count("|")
        lines = [line for line in text.splitlines() if line.strip()]
        table_line_ratio = (
            sum(1 for line in lines if "|" in line or "\t" in line) / max(len(lines), 1)
        )
        digit_ratio = len(re.findall(r"\d", normalized)) / max(len(normalized), 1)
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", normalized))
        separator_ratio = len(re.findall(r"[|,:：/\\]", normalized)) / max(len(normalized), 1)

        return (
            pipe_count >= 6
            or table_line_ratio >= 0.35
            or (digit_ratio > 0.28 and cjk_count < 80 and len(normalized) > 100)
            or (separator_ratio > 0.22 and len(normalized) > 120)
        )

