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

def get_intervention_plan(predicted_labels):
    """
    Input: List of detected labels ['toxic', 'threat']
    Output: Dictionary containing the Severity, Explanation, and Action.
    """
    if not predicted_labels:
        return CYBERBULLYING_ONTOLOGY["clean"]

    # Priority Rank: Critical > High > Medium > Low
    severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    
    highest_severity_label = predicted_labels[0]
    max_score = -1

    # Find the label with the highest severity score
    for label in predicted_labels:
        # Standardize label if model gives different casing
        label_key = label.lower()
        info = CYBERBULLYING_ONTOLOGY.get(label_key)
        
        if info:
            current_rank = severity_rank.get(info['severity'], 0)
            if current_rank > max_score:
                max_score = current_rank
                highest_severity_label = label_key
    
    return CYBERBULLYING_ONTOLOGY.get(highest_severity_label, CYBERBULLYING_ONTOLOGY["clean"])