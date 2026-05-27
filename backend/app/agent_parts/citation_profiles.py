from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitationStabilityProfile:
    name: str
    question_groups: tuple[tuple[str, ...], ...]
    evidence_groups: tuple[tuple[str, ...], ...]
    quote_terms: tuple[str, ...]
    quote_bonus: tuple[str, ...] = ()
    avoid_terms: tuple[str, ...] = ()
    quote_limit: int = 620


def citation_stability_profiles() -> tuple[CitationStabilityProfile, ...]:
    return (
        CitationStabilityProfile(
            name="attention_complexity_table_1",
            question_groups=(
                ("table 1",),
                ("self-attention", "self attention"),
                ("recurrent",),
                ("convolutional",),
            ),
            evidence_groups=(
                ("table 1",),
                ("sequential operations", "sequentially executed operations", "minimum number of sequential operations"),
                ("maximum path length", "maximum path lengths", "path length"),
            ),
            quote_terms=(
                "table 1",
                "complexity per layer",
                "sequential operations",
                "maximum path length",
                "self-attention",
                "recurrent",
                "convolutional",
            ),
            quote_bonus=("maximum path lengths", "sequential operations", "complexity per layer"),
            avoid_terms=("table 2", "bleu", "28.4", "41.8"),
            quote_limit=760,
        ),
        CitationStabilityProfile(
            name="attention_bleu_table_2",
            question_groups=(("bleu",),),
            evidence_groups=(("table 2",), ("28.4",), ("41.8",)),
            quote_terms=(
                "table 2",
                "bleu",
                "wmt 2014",
                "english-to-german",
                "english-to-french",
                "en-de",
                "en-fr",
                "28.4",
                "41.8",
            ),
            quote_bonus=("table 2", "28.4", "41.8", "state-of-the-art"),
            avoid_terms=("table 1", "complexity per layer", "maximum path length"),
            quote_limit=620,
        ),
        CitationStabilityProfile(
            name="gpt2_bpe_context",
            question_groups=(("gpt-2", "gpt2"), ("vocabulary", "bpe"), ("context", "context length")),
            evidence_groups=(("byte pair encoding", "bpe"), ("50,257", "50257"), ("1024",)),
            quote_terms=("byte pair encoding", "bpe", "50,257", "50257", "context size", "context length", "1024"),
            quote_bonus=("50,257", "1024", "byte pair encoding"),
            quote_limit=520,
        ),
        CitationStabilityProfile(
            name="clip_contrastive_batch",
            question_groups=(("clip",), ("contrastive",), ("batch",)),
            evidence_groups=(
                ("image encoder",),
                ("text encoder",),
                ("cosine similarity",),
                ("n real pairs", "real pairs"),
            ),
            quote_terms=("image encoder", "text encoder", "cosine similarity", "n real pairs", "real pairs"),
            quote_bonus=("n real pairs", "cosine similarity"),
            quote_limit=620,
        ),
        CitationStabilityProfile(
            name="attention_wmt_training_data",
            question_groups=(
                ("attention", "transformer"),
                ("wmt 2014", "wmt"),
                ("training data", "data sizes", "sentence pairs"),
            ),
            evidence_groups=(
                ("english-german", "english-to-german"),
                ("4.5 million",),
                ("english-french", "english-to-french"),
                ("36m", "36 million"),
            ),
            quote_terms=("wmt 2014", "english-german", "english-french", "4.5 million", "36m", "36 million"),
            quote_bonus=("4.5 million", "36m"),
            avoid_terms=("table 2", "bleu", "28.4", "41.8"),
            quote_limit=620,
        ),
        CitationStabilityProfile(
            name="sam_automatic_masks",
            question_groups=(("sam", "segment anything"), ("automatic", "automatically"), ("mask", "masks")),
            evidence_groups=(("1.1b", "1.1 billion"), ("99.1%", "99.1 percent"), ("fully automatically",)),
            quote_terms=("1.1b", "1.1 billion", "99.1%", "99.1 percent", "fully automatically"),
            quote_bonus=("1.1b", "99.1%", "fully automatically"),
            quote_limit=520,
        ),
    )
