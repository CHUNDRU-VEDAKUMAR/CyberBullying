# src/ontology.py

# A Knowledge Graph / Rules Engine for Interventions
CYBERBULLYING_ONTOLOGY = {
    "severe_toxic": {
        "severity": "CRITICAL",
        "explanation": "Extreme toxicity detected. Contains highly offensive language intended to cause severe harm.",
        "intervention": "BLOCK_ACCOUNT_IMMEDIATELY + REPORT_TO_CYBER_CELL"
    },
    "threat": {
        "severity": "CRITICAL",
        "explanation": "Physical threat detected. The text implies intent to kill, injure, or physically harm.",
        "intervention": "POLICE_ALERT + ACCOUNT_SUSPENSION"
    },
    "identity_hate": {
        "severity": "HIGH",
        "explanation": "Hate speech detected. Attacks a protected group (race, religion, gender, etc.).",
        "intervention": "PERMANENT_BAN + HIDE_CONTENT"
    },
    "toxic": {
        "severity": "MEDIUM",
        "explanation": "General toxicity. The content is rude, disrespectful, or unreasonable.",
        "intervention": "HIDE_COMMENT + ISSUE_WARNING_STRIKE_1"
    },
    "insult": {
        "severity": "LOW",
        "explanation": "Personal insult. Uses disparaging language towards an individual.",
        "intervention": "FLAG_FOR_REVIEW + USER_TIMEOUT(24H)"
    },
    "obscene": {
        "severity": "LOW",
        "explanation": "Obscene language. Uses vulgarity or profanity.",
        "intervention": "AUTO_FILTER_WORDS + WARN_USER"
    },
    "clean": {
        "severity": "NONE",
        "explanation": "No cyberbullying detected.",
        "intervention": "NO_ACTION"
    }
}

from src.config import DEFAULTS


def get_intervention_plan(predicted_labels_or_scores, min_score=None):
    """
    Input: either a list of labels OR a dict of {label: score}
    Output: Dictionary containing the Severity, Explanation, Action, and confidence.
    """
    # Normalize input to scores dict
    if not predicted_labels_or_scores:
        return CYBERBULLYING_ONTOLOGY["clean"]

    if isinstance(predicted_labels_or_scores, dict):
        scores = predicted_labels_or_scores
    else:
        # list of labels -> convert to dict with default confidence 1.0
        scores = {label: 1.0 for label in predicted_labels_or_scores}

    # Filter out low-confidence scores below `min_score` so that tiny residual
    # probabilities don't trigger high-severity interventions for safe text.
    if min_score is None:
        min_score = float(DEFAULTS.get('min_score', 0.5))

    if isinstance(scores, dict):
        scores = {k: float(v) for k, v in scores.items() if float(v) >= float(min_score)}

    # Priority Rank: Critical > High > Medium > Low
    severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

    best_label = None
    best_priority = -1
    best_score = 0.0

    for label, score in scores.items():
        key = label.lower()
        info = CYBERBULLYING_ONTOLOGY.get(key)
        if not info:
            continue
        priority = severity_rank.get(info['severity'], 0)
        # Choose higher severity first, then higher confidence
        if priority > best_priority or (priority == best_priority and score > best_score):
            best_label = key
            best_priority = priority
            best_score = score

    if not best_label:
        return CYBERBULLYING_ONTOLOGY["clean"]

    # Build the plan and compute a calibrated confidence score
    plan = CYBERBULLYING_ONTOLOGY.get(best_label, CYBERBULLYING_ONTOLOGY["clean"]).copy()

    # Confidence calibration: normalize best_score against the total mass
    total_mass = sum(float(v) for v in scores.values()) if scores else 0.0
    if total_mass <= 0:
        calibrated_conf = float(best_score)
    else:
        calibrated_conf = float(best_score) / (total_mass + 1e-8)

    # Bound confidence to [0,1] and provide a readable score
    calibrated_conf = max(0.0, min(1.0, calibrated_conf))
    plan['confidence'] = round(calibrated_conf, 4)
    plan['detected_label'] = best_label

    # Aggregate severity note: if there are multiple positive labels, include that context
    positive_labels = [k for k, v in scores.items() if v > 0.1]
    if len(positive_labels) > 1:
        plan['note'] = f"Multiple signals detected: {', '.join(positive_labels)}. Action prioritized by severity."

    return plan


def aggregate_severity(scores):
    """Return an aggregated severity level given raw label scores.

    Rules:
    - If any CRITICAL label score > 0.3 -> CRITICAL
    - Else if sum(HIGH) > 0.4 -> HIGH
    - Else if sum(MEDIUM) > 0.4 -> MEDIUM
    - Else if any LOW > 0.2 -> LOW
    - Otherwise NONE
    """
    severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    score_by_sev = {k: 0.0 for k in severity_rank.keys()}
    for label, val in scores.items():
        info = CYBERBULLYING_ONTOLOGY.get(label.lower())
        if not info:
            continue
        score_by_sev[info['severity']] += float(val)

    if score_by_sev['CRITICAL'] > 0.3:
        return 'CRITICAL'
    if score_by_sev['HIGH'] > 0.4:
        return 'HIGH'
    if score_by_sev['MEDIUM'] > 0.4:
        return 'MEDIUM'
    if score_by_sev['LOW'] > 0.2:
        return 'LOW'
    return 'NONE'


def recommend_intervention(plan):
    """Enhance intervention recommendation based on confidence and aggregated severity."""
    sev = plan.get('severity')
    conf = plan.get('confidence', 0.0)

    # conservative mapping by confidence
    if sev == 'CRITICAL':
        if conf >= 0.5:
            action = 'BLOCK_ACCOUNT_IMMEDIATELY + REPORT_TO_CYBER_CELL'
        else:
            action = 'SUSPEND_ACCOUNT_TEMP + FLAG_FOR_HUMAN_REVIEW'
    elif sev == 'HIGH':
        if conf >= 0.5:
            action = 'PERMANENT_BAN + HIDE_CONTENT'
        else:
            action = 'HIDE_COMMENT + FLAG_FOR_REVIEW'
    elif sev == 'MEDIUM':
        if conf >= 0.5:
            action = 'HIDE_COMMENT + ISSUE_WARNING_STRIKE_1'
        else:
            action = 'FLAG_FOR_REVIEW + SUGGESTED_WARNING'
    elif sev == 'LOW':
        action = 'AUTO_FILTER_WORDS + WARN_USER'
    else:
        action = 'NO_ACTION'

    plan['recommended_action'] = action
    return plan