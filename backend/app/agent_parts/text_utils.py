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
        sanitized = re.sub(r"[\w.\-+]+@[\w.\-]+\.\w+", "[邮箱已隐藏]", text)
        sanitized = re.sub(r"(姓\s*名|姓名)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        sanitized = re.sub(r"(学\s*号|学号)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        sanitized = re.sub(r"(电子邮件|邮箱)\s*[:：]\s*\S+", r"\1：[已隐藏]", sanitized)
        return sanitized

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
            "transformer",
            "attention",
            "sequence transduction",
            "machine translation",
            "state-of-the-art",
            "architecture",
            "model",
            "training",
            "实验",
            "模型",
            "方法",
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
        if not tokens or not normalized_text:
            return 0.0

        hits = sum(1 for token in tokens if token.lower() in normalized_text)
        exact_score = hits / max(len(tokens), 1)

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

        return max(0.0, min(1.0, exact_score * 0.75 + char_score * 0.25))

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
        keywords = ["表格", "表", "公式", "计算", "数值", "数据是多少", "参数", "指标"]
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
        candidates = re.split(r"\n+|(?<=[。！？.!?；;])\s*", sanitized)
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
            return normalized[: min(limit, 120)] + ("..." if len(normalized) > min(limit, 120) else "")
        return normalized[:limit] + ("..." if len(normalized) > limit else "")

    def _best_quote_for_question(self, question: str, text: str, limit: int = 240) -> str:
        sanitized = self._sanitize_evidence_text(text)
        if not sanitized.strip():
            return ""

        if self._looks_like_compound_request(question):
            preferred_keywords = self._compound_focus_keywords_for_question(question)
        elif self._looks_like_reference_question(question):
            return self._best_reference_quote(sanitized, limit=limit)
        elif self._looks_like_structured_review_request(question):
            preferred_keywords = [
                "摘要",
                "本文围绕",
                "采用",
                "文献分析",
                "认知支架",
                "资源重组",
                "课程学习场景",
                "风险",
                "人机协同",
                "未来研究",
                "实证数据",
                "参考文献",
            ]
        elif self._looks_like_title_alignment_question(question):
            preferred_keywords = [
                "机制、风险与治理路径",
                "未来研究",
                "实证数据",
                "认知支架",
                "资源重组",
                "过程陪伴",
                "反馈生成",
                "组织协同",
                "学习依赖",
                "信息准确性",
                "数据隐私",
                "算法偏差",
                "学术诚信",
                "责任边界",
                "人机协同",
                "价值对齐",
                "过程可控",
                "数据最小化",
                "多主体治理",
                "未来研究",
                "实证数据",
                "文献分析",
                "情境推演",
                "机制建构",
            ]
        elif self._looks_like_reliability_question(question):
            preferred_keywords = [
                "随机生成",
                "论文样稿",
                "采用",
                "文献分析",
                "情境推演",
                "机制建构",
                "未来研究",
                "实证数据",
                "参考文献",
                "风险",
                "局限",
            ]
        elif self._looks_like_research_limitation_question(question):
            preferred_keywords = [
                "局限性",
                "研究局限",
                "研究不足",
                "结论与展望",
                "未来研究",
                "实证数据",
                "检验",
                "验证",
                "不同应用场景",
                "不同学生群体",
                "文献分析",
                "情境推演",
                "机制建构",
            ]
        elif self._looks_like_document_wide_question(question):
            preferred_keywords = self._overview_focus_keywords(question)
        else:
            preferred_keywords = []

        candidates = [
            " ".join(part.split()).strip()
            for part in re.split(r"\n+|(?<=[。！？.!?；;])\s*", sanitized)
            if part.strip()
        ]
        for keyword in preferred_keywords:
            for candidate in candidates:
                if keyword in candidate and not self._is_table_like_text(candidate):
                    return candidate[:limit] + ("..." if len(candidate) > limit else "")
        return self._best_readable_quote(sanitized, limit=limit)

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

