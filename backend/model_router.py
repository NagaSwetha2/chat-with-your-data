"""
Model Router — single source of truth for LLM selection.

Tiers:
  extract      gpt-4o-mini / temp=0   Structured extraction, format-following
  classify     gpt-4o-mini / temp=0   Intent detection, classification
  followup     gpt-4o-mini / temp=0   Follow-up question generation
  narrate      gpt-4o-mini / temp=0.1 Verbalise pre-computed pandas results
  insights     gpt-4o-mini / temp=0.3 EDA insight cards
  reason       gpt-4o      / temp=0.3 Complex business reasoning
  hypothesize  gpt-4o      / temp=0.3 Hypothesis formation & recommendations

Auto-escalation: "narrate" upgrades to "reason" when the question is complex
(multi-clause, contains why/should/recommend/strategy/prioritize).

Deterministic tier (no LLM at all):
  _detect_intent() — regex/keyword
  _compute_analytics() — pandas
  All charts — matplotlib
  Anomaly / trend / prediction — numpy + sklearn
"""
import os
import re
from langchain_openai import ChatOpenAI

_TASK_CONFIG: dict[str, tuple[str, float, int]] = {
    #            model           temp  timeout
    "extract":   ("gpt-4o-mini", 0,    25),
    "classify":  ("gpt-4o-mini", 0,    20),
    "followup":  ("gpt-4o-mini", 0,    20),
    "narrate":   ("gpt-4o-mini", 0.1,  20),
    "insights":  ("gpt-4o-mini", 0.3,  25),
    "reason":    ("gpt-4o",      0.3,  40),
    "hypothesize":("gpt-4o",     0.3,  40),
}

# Patterns that signal a question needs genuine reasoning, not just narration
_COMPLEX_RE = re.compile(
    r'\bwhy\b'
    r'|\bshould\b'
    r'|\brecommend\b'
    r'|\bstrateg(?:y|ies)\b'
    r'|\bprioritize?\b'
    r'|\bhow\s+(?:do|can|should|to)\b'
    r'|\bwhat\s+(?:should|must|can\s+i)\b'
    r'|\bexplain\s+why\b'
    r'|\broot\s+cause\b'
    r'|\brisk\b.{0,30}\b(?:focus|prior|strateg)\b'
    r'|\bcompare\b.{0,40}\band\b',
    re.IGNORECASE,
)


def is_complex(question: str) -> bool:
    """Return True if the question warrants the strong reasoning model."""
    return len(question) > 90 or bool(_COMPLEX_RE.search(question))


def get_model(task: str, question: str = "") -> ChatOpenAI:
    """
    Return the right ChatOpenAI instance for the given task.

    Pass `question` to enable auto-escalation of 'narrate' → 'reason'.
    """
    effective_task = task
    if task == "narrate" and question and is_complex(question):
        effective_task = "reason"

    model, temp, timeout = _TASK_CONFIG.get(effective_task, ("gpt-4o-mini", 0.2, 20))

    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=temp,
        timeout=timeout,
    )
