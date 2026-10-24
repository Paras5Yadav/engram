"""The controller's question catalog.

Every memory-management decision is expressed as typed questions:

* ``ProbQuestion`` — a yes/no proposition; the backend returns P(true) in [0, 1]
* ``PickQuestion`` — a categorical choice; the backend returns one option key

Each question's ``state`` fields are supplied by the caller at ask-time; the
controller never relies on hidden context.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProbQuestion:
    instructions: str
    yes: str
    no: str


@dataclass
class PickQuestion:
    instructions: str
    options: dict[str, str] = field(default_factory=dict)


QUESTIONS = "questions"


def admission_questions() -> dict[str, ProbQuestion]:
    return {
        "keep": ProbQuestion(
            "Should `observation` be retained for future recall?",
            "Contains a concrete fact, plan, preference, event or need worth recalling later.",
            "Greeting, acknowledgement, or filler with no recallable detail."),
        "fresh": ProbQuestion(
            "Does `observation` add detail not already present in `recent`?",
            "At least one new, corrected or time-updated detail versus every recent item.",
            "Paraphrase or exact repeat of something already in `recent`."),
        "duplicate": ProbQuestion(
            "Is `observation` a repeat of an item in `recent`?",
            "Same information, no new detail.",
            "New or updated information."),
    }


def typing_questions() -> dict[str, ProbQuestion]:
    return {
        "type_event": ProbQuestion(
            "Does `observation` describe a particular experience or event?",
            "A specific past, current or planned happening.",
            "No particular happening is described."),
        "type_fact": ProbQuestion(
            "Does `observation` state a durable fact about a person, thing or the world?",
            "An attributable fact or relationship that outlives the turn.",
            "Nothing asserted that stays true beyond the turn."),
        "type_how": ProbQuestion(
            "Does `observation` explain the steps for performing a task?",
            "Steps, method or actionable rule.",
            "No procedure described."),
        "type_like": ProbQuestion(
            "Does `observation` express a preference, taste or habit of a participant?",
            "A like, dislike, allergy, habit or chosen option.",
            "No preference expressed."),
    }


def relation_questions() -> dict[str, ProbQuestion]:
    return {
        "topic_link": ProbQuestion(
            "Do `observation` and `candidate` share a specific topic worth linking?",
            "A shared subject, fact or event makes the link useful.",
            "Only generic vocabulary overlap."),
        "explains": ProbQuestion(
            "Does `candidate` cause, enable or explain what happened in `observation`?",
            "The candidate account supports this direction of explanation.",
            "Similarity or chronology alone is not explanation."),
        "explained_by": ProbQuestion(
            "Does `observation` cause, enable or explain what happened in `candidate`?",
            "The new account supports this direction of explanation.",
            "Similarity or chronology alone is not explanation."),
        "same_entity": ProbQuestion(
            "Do `observation` and `candidate` refer to the same real-world person, place or thing?",
            "Names or aliases resolve to one entity.",
            "Distinct entities, or not enough evidence."),
    }


def consolidation_questions() -> dict[str, object]:
    return {
        "repeats": ProbQuestion(
            "Do `observation` and `candidate` state the same fact with no new detail?",
            "Duplicate or paraphrase with nothing added.",
            "Different details or distinct occurrences."),
        "conflicts": ProbQuestion(
            "Do `observation` and `candidate` make incompatible claims about the same subject?",
            "Both cannot hold for the same time and subject.",
            "Compatible, or a change over time explains the difference."),
        "supersedes": ProbQuestion(
            "Does `observation` explicitly replace the earlier fact in `candidate`?",
            "An explicit update makes the candidate outdated.",
            "No explicit replacement."),
        "representation": PickQuestion(
            "Pick the safest representation for this pair.",
            {
                "keep_apart": "Distinct facts or contradictions; merging loses detail.",
                "combine": "Same fact; nothing is lost by one representation.",
                "generalize": "Repeated occurrences support a stable pattern.",
                "unsure": "Not enough evidence to combine or separate.",
            }),
    }


def routing_questions() -> dict[str, ProbQuestion]:
    return {
        "need_topic": ProbQuestion(
            "Would topically related memories help answer `query`?",
            "Related facts are useful.", "Not needed."),
        "need_time": ProbQuestion(
            "Does `query` need dates, ordering, durations or change over time?",
            "A time relation is required.", "Time is incidental."),
        "need_cause": ProbQuestion(
            "Does `query` ask for a cause, motivation or explanation?",
            "Explanatory evidence is required.", "Only facts are requested."),
        "need_entity": ProbQuestion(
            "Would connecting mentions of the same person, place or thing help answer `query`?",
            "Entity-specific facts are useful.", "Entity identity is irrelevant."),
        "need_multi_hop": ProbQuestion(
            "Does `query` require combining two or more separate memories?",
            "Comparison, aggregation or chained inference.", "One fact suffices."),
        "need_latest": ProbQuestion(
            "Does `query` need the most recent applicable fact?",
            "Current state or latest update.", "Historical fact is fine."),
    }


def traversal_questions() -> dict[str, ProbQuestion]:
    return {
        "relevant": ProbQuestion(
            "Does `candidate` contain information needed to answer `query`?",
            "Direct answer material or a necessary link.", "Only topic overlap."),
        "adds_detail": ProbQuestion(
            "Does `candidate` add an answer-relevant detail missing from `evidence`?",
            "A distinct relevant detail.", "Already covered or irrelevant."),
        "corroborates": ProbQuestion(
            "Does `candidate` independently support a claim already in `evidence`?",
            "Compatible support for a specific claim.", "No specific support."),
    }


def stopping_questions() -> dict[str, ProbQuestion]:
    return {
        "enough": ProbQuestion(
            "Does `evidence` support every factual part of an answer to `query`?",
            "A grounded answer is possible.", "Something required is missing."),
        "worth_continuing": ProbQuestion(
            "Would another retrieval round likely fill a specific gap for `query`?",
            "An identifiable gap could be filled.", "No useful gap remains."),
        "missing_piece": ProbQuestion(
            "Is at least one fact required by `query` absent from `evidence`?",
            "A required detail is unsupported.", "Everything is supported."),
        "conflicts_present": ProbQuestion(
            "Does `evidence` contain unreconciled conflicting claims relevant to `query`?",
            "A conflict affects the answer.", "No relevant conflict."),
    }
