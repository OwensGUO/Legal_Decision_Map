"""Clause-level realization: deterministic edits, one quoted rewrite, and splicing.

The generator no longer rewrites a whole case. A plan picks the exact comma clause
that carries the target defendant's fact, deterministic fields (amount, restitution)
are edited in place, and only a behaviour flip asks the model for one replacement
sentence. Attribution is therefore a property of the span, not of re-extraction from
free text. Clauses rather than sentences because a joint narrative names the target
and a co-defendant in neighbouring clauses of the same sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from legal_landscape.counterfactual.validators import amounts_match
from legal_landscape.factors.extract import CONDUCT_CUES, RESTITUTION_CUES
from legal_landscape.factors.schema import InterventionSpec

TERMINATORS = "。；！？\n"
_SPLIT = re.compile(rf"[^{re.escape(TERMINATORS)}]+[{re.escape(TERMINATORS)}]?")
_CLAUSE = re.compile(r"[^，。；！？\n]+")
# Cue-aware amount literal, mirroring the extractor's reading of 涉案金额/价值/获利.
_CUED_AMOUNT = re.compile(
    r"(?:涉案(?:金额)?|价值|获利|金额|数额|共计)[^0-9，。；]{0,8}?"
    r"([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万)?元"
)
_ANY_AMOUNT = re.compile(r"([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万)?元")
_MASKS = ("××", "[已屏蔽]")
# Judgments write conduct with inflected verbs the canonical cue table does not carry
# (盗得, 窃得, 骗得). These lists are for consistency checking, never for extraction.
_TAKING_VERBS = (
    "盗得", "窃得", "骗得", "抢得", "夺取", "夺走", "劫取", "劫走", "强行劫走",
    "盗走", "偷走", "窃取", "盗取", "扒窃", "骗取", "秘密窃取",
    "趁人不备", "乘人不备", "持刀威胁", "以暴力相威胁", "胁迫交付",
)
_HARM_VERBS = (
    "殴打", "打伤", "打倒在", "致其轻伤", "致人轻伤", "致其重伤", "致人重伤",
    "轻伤一级", "轻伤二级", "重伤一级", "重伤二级", "持棍棒", "持械",
)
# Verbs that describe only a role in a joint offence, never the taking itself.
_PARTICIPATION_VERBS = ("放哨", "望风", "驾车", "开车接应", "接应", "负责开车")
# Summary lines restate the count rather than describing one taking. Keep these phrases
# specific: 共计 appears in ordinary counts (价值共计2304元) and 起 in 一起/起诉.
_SUMMARY_MARKERS = ("共参与", "共实施", "共同作案", "作案", "总计", "总价值", "共计作案")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Realization:
    """How one request is satisfied.

    ``deterministic`` splices ``text`` with no model call, ``span`` asks the model for a
    replacement for ``quote`` inside ``base_text``, and ``skip`` is a structural refusal
    that no retry can fix.
    """

    kind: Literal["deterministic", "span", "skip"]
    base_text: str
    span: Span | None = None
    text: str | None = None
    quote: str | None = None
    reason: str | None = None


def sentence_spans(text: str) -> tuple[Span, ...]:
    """Non-empty sentence spans with their terminator included."""
    return tuple(Span(m.start(), m.end(), m.group()) for m in _SPLIT.finditer(text))


def clause_spans(text: str) -> tuple[Span, ...]:
    return tuple(Span(m.start(), m.end(), m.group()) for m in _CLAUSE.finditer(text))


def other_defendants(spec: InterventionSpec) -> tuple[str, ...]:
    return tuple(
        name for name in spec.co_defendants if name and name != spec.target_defendant
    )


def mask_counts(text: str) -> tuple[int, int]:
    return tuple(text.count(mask) for mask in _MASKS)  # type: ignore[return-value]


def mentions(text: str, name: str, others: tuple[str, ...]) -> bool:
    """True when ``name`` names this defendant rather than part of a longer co-defendant.

    Anonymized names share prefixes (李某 and 李某A), so a bare substring test would read
    every 李某A as 李某. Deleting the longer names first keeps the two apart.
    """
    stripped = text
    for other in others:
        if other != name and name in other:
            stripped = stripped.replace(other, "")
    return name in stripped


def scope_spans(parent_text: str, spec: InterventionSpec) -> tuple[Span, ...]:
    """Clauses that may carry the target defendant's fact.

    Clause-level rather than sentence-level: a multi-defendant narrative puts the
    target's fact in one comma-separated clause and another defendant's in the next,
    so a sentence-level rule discards most of the evidence. A clause belongs to the
    target when it names the target and no other defendant, or when it names nobody
    (a pronoun or impersonal continuation). Clauses naming only another defendant
    never belong to the target.
    """
    others = other_defendants(spec)
    if not others:
        return clause_spans(parent_text)
    kept: list[Span] = []
    for sentence in sentence_spans(parent_text):
        names_target = mentions(sentence.text, spec.target_defendant, others)
        names_other = any(name in sentence.text for name in others)
        for clause in clause_spans(sentence.text):
            span = Span(sentence.start + clause.start, sentence.start + clause.end, clause.text)
            clause_target = mentions(clause.text, spec.target_defendant, others)
            clause_other = any(name in clause.text for name in others)
            if clause_target and not clause_other:
                kept.append(span)
            elif not clause_target and not clause_other and names_target and not names_other:
                # Inherit the sentence: an unnamed clause belongs to its sentence's subject.
                kept.append(span)
    return tuple(kept)


def _format_amount(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _splice(text: str, span: Span, replacement: str) -> str:
    return text[: span.start] + replacement + text[span.end :]


def _clause(value: bool) -> str:
    return "案发后已退赃退赔" if value else "案发后未退赃退赔"


def _restationalize(text: str, spec: InterventionSpec) -> str | None:
    """Replace every restitution clause inside the target scope with one canonical clause."""
    value = spec.target_factors.restitution
    if value is None:
        return None
    clause = _clause(bool(value))
    spans = scope_spans(text, spec)
    if not spans:
        return None
    others = other_defendants(spec)
    # A multi-defendant edit must name the target; a single-defendant one keeps the
    # sentence's existing subject.
    prefix = spec.target_defendant if others else ""
    edits: list[tuple[int, int, str]] = [
        (span.start, span.end, prefix + clause)
        for span in spans
        if any(cue in span.text for cue in RESTITUTION_CUES)
    ]
    if not edits:
        # No explicit restitution clause: state the fact at the end of a target clause.
        anchor = spans[0]
        edits.append((anchor.start, anchor.end, anchor.text + "，" + prefix + clause))
    result = text
    for start, end, new in reversed(edits):
        result = result[:start] + new + result[end:]
    return result


def _literal_is_source(match: re.Match[str], source: float | None) -> bool:
    """The matched literal is the amount the intervention starts from.

    A joint total or a compensation figure in the same clause is a different fact;
    editing it would leave the intervened amount unchanged.
    """
    if source is None:
        return True
    try:
        value = float(re.sub("[,，]", "", match.group(1)))
    except ValueError:
        return False
    if match.group(2):
        value *= 10000.0
    return amounts_match(value, float(source))


def _amountalize(text: str, spec: InterventionSpec) -> str | None:
    """Replace the target defendant's amount literal; 万元 sources are written back in 元."""
    target = spec.target_factors.amount
    if target is None:
        return None
    source = spec.source_factors.amount
    spans = scope_spans(text, spec)
    if not spans:
        return None
    literal = _format_amount(float(target)) + "元"
    for span in spans:
        match = _CUED_AMOUNT.search(span.text)
        if match is not None and _literal_is_source(match, source):
            start = span.start + match.start(1)
            end = span.start + match.end()
            return text[:start] + literal + text[end:]
    for span in spans:
        if not mentions(span.text, spec.target_defendant, other_defendants(spec)):
            continue
        match = _ANY_AMOUNT.search(span.text)
        if match is not None and _literal_is_source(match, source):
            start = span.start + match.start(1)
            end = span.start + match.end()
            return text[:start] + literal + text[end:]
    return None


_BARE_INITIAL_VERBS = "破进将趁因持驾开负把抓在到殴打盗窃夺抢骗敲望放接"


def _behaviour_verbs(conduct: str | None) -> tuple[str, ...]:
    """Verb set used to find a case's own description of the conduct being flipped."""
    if conduct in ("秘密窃取", "趁人不备公开夺取", "暴力胁迫取财"):
        return _TAKING_VERBS
    if conduct == "虚构事实使被害人自愿交付":
        return (*_TAKING_VERBS, "虚构事实", "隐瞒真相", "谎称", "冒充")
    if conduct in ("无事生非随意殴打", "因个人纠纷定向伤害"):
        return _HARM_VERBS
    return tuple(dict(CONDUCT_CUES).get(conduct or "", ()))


def _bare_initial_actors(text: str, spec: InterventionSpec) -> set[str]:
    """Actors a judgment refers to by a bare surname initial, as in 王破坏防护栏.

    Such a reference cannot be attributed to one defendant: 王 is a prefix of both 王某A
    and an unrelated 王某, so any conduct in that clause belongs to an unknown actor.
    """
    actors = (spec.target_defendant, *other_defendants(spec))
    initials = {actor[:1] for actor in actors if actor}
    if not initials:
        return set()
    pattern = re.compile(
        rf"(?:由|被告人|，|。|；|、|和|与|及|^)(?:{'|'.join(map(re.escape, sorted(initials)))})"
        rf"(?=[{_BARE_INITIAL_VERBS}])"
    )
    return {match.group(0)[-1] for match in pattern.finditer(text)}


def _behaviour_spans(text: str, spec: InterventionSpec, verbs: tuple[str, ...]) -> tuple[Span, ...]:
    """Every clause describing the conduct, whichever defendant it names.

    Deliberately wider than scope_spans: a case that describes taking property in its
    factual counts must be rewritten in all of them, and a count usually names only the
    victim, not the defendant.
    """
    others = other_defendants(spec)
    kept: list[Span] = []
    for sentence in sentence_spans(text):
        for clause in clause_spans(sentence.text):
            body = clause.text
            if any(marker in body for marker in _SUMMARY_MARKERS):
                continue
            names_other = any(name in body for name in others)
            names_target = mentions(body, spec.target_defendant, others)
            if names_other and not names_target:
                continue
            if any(verb in body for verb in verbs):
                kept.append(
                    Span(sentence.start + clause.start, sentence.start + clause.end, body)
                )
    return tuple(kept)


def _flip_span(text: str, spec: InterventionSpec) -> tuple[Span | None, str | None]:
    """Pick the clause a behaviour flip rewrites, or the structural reason it cannot.

    A flip is only sound when the case describes the conduct exactly once and names the
    actor unambiguously. Anything else would either leave contradicting counts behind or
    edit a defendant the program cannot identify.
    """
    if _bare_initial_actors(text, spec):
        return None, "abbreviated_ambiguous"
    verbs = _behaviour_verbs(spec.source_factors.conduct)
    if not verbs:
        return None, "no_target_span"
    spans = _behaviour_spans(text, spec, verbs)
    if not spans:
        return None, "no_target_span"
    if len(spans) > 1:
        # Several counts describe the same conduct; one quoted rewrite cannot reconcile them.
        return None, "incomplete_flip"
    span = spans[0]
    if any(verb in span.text for verb in _PARTICIPATION_VERBS) and not any(
        verb in span.text for verb in verbs
    ):
        return None, "no_target_span"
    return span, None


def plan_realization(parent_text: str, spec: InterventionSpec) -> Realization:
    """Choose the realization path for one request.

    Deterministic fields are spliced first; a behaviour flip is planned on the result so
    the quoted span carries the final text.
    """
    text = parent_text
    if "restitution" in spec.changed_fields:
        edited = _restationalize(text, spec)
        if edited is None:
            return Realization("skip", parent_text, reason="no_restitution_span")
        text = edited
    if "amount" in spec.changed_fields:
        edited = _amountalize(text, spec)
        if edited is None:
            return Realization("skip", parent_text, reason="no_amount_span")
        text = edited
    if "conduct" in spec.changed_fields:
        span, reason = _flip_span(text, spec)
        if span is None:
            return Realization("skip", parent_text, reason=reason)
        return Realization("span", text, span=span, quote=span.text)
    if text == parent_text:
        return Realization("skip", parent_text, reason="no_supported_field")
    return Realization("deterministic", parent_text, text=text)


def splice(base_text: str, span: Span, replacement: str) -> tuple[str, Span]:
    """Insert one replacement sentence at the planned span and report its new offsets."""
    body = replacement.strip()
    if span.text and span.text[-1] in TERMINATORS:
        # A whole sentence: keep exactly one terminator.
        if body and body[-1] in TERMINATORS:
            terminator = body[-1]
            body = body[:-1]
        else:
            terminator = span.text[-1]
        full = body + terminator
    else:
        # A clause inside a sentence: never terminate it, or the sentence splits.
        full = body.rstrip(TERMINATORS)
    return _splice(base_text, span, full), Span(span.start, span.start + len(full), full)


def replacement_problem(
    spec: InterventionSpec, original: str, replacement: str
) -> str | None:
    """Program-side checks on a model replacement, before it is spliced.

    These are the properties the old flow re-derived from the finished text; checking
    the sentence itself makes them cheap and exact.
    """
    others = other_defendants(spec)
    body = replacement.strip()
    if not body:
        return "empty_replacement"
    if "\n" in body:
        return "replacement_not_single_sentence"
    if not mentions(body, spec.target_defendant, others):
        return "replacement_missing_target"
    if any(name in body for name in others):
        return "replacement_mentions_other_defendant"
    source = spec.source_factors.conduct
    target = spec.target_factors.conduct
    source_cues = dict(CONDUCT_CUES).get(source or "", ())
    if source and source != target and any(cue in body for cue in source_cues):
        return "replacement_keeps_source_conduct"
    target_cues = dict(CONDUCT_CUES).get(target or "", ())
    if target_cues and not any(cue in body for cue in target_cues):
        return "replacement_missing_target_conduct"
    before, after = mask_counts(original), mask_counts(body)
    if any(after[index] < before[index] for index in range(len(_MASKS))):
        return "replacement_restores_mask"
    return None
