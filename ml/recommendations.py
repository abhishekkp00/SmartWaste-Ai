"""
ml/recommendations.py
─────────────────────────────────────────────────────────────────────────────
SmartWaste AI — Recycling recommendation engine.

Maps waste class names to disposal guidance and uncertainty messages.
"""

from typing import Dict

from ml.config import cfg


RECOMMENDATIONS: Dict[str, Dict[str, str]] = {
    "cardboard": {
        "action": "Recycle",
        "guidance": "Flatten and keep dry. Remove contamination and place in cardboard/paper recycling.",
        "bin": "Blue or paper/cardboard recycling bin",
    },
    "glass": {
        "action": "Recycle",
        "guidance": "Empty and rinse containers. Keep lids separate if local rules require it.",
        "bin": "Glass recycling bin",
    },
    "metal": {
        "action": "Recycle",
        "guidance": "Empty and rinse cans. Avoid compressing sharp metal items.",
        "bin": "Metal recycling bin",
    },
    "paper": {
        "action": "Recycle",
        "guidance": "Keep clean and dry. Remove plastic films, food residue, and wax coatings.",
        "bin": "Blue or paper recycling bin",
    },
    "plastic": {
        "action": "Recycle conditionally",
        "guidance": "Check the local recycling code. Empty and rinse accepted plastic containers.",
        "bin": "Plastic recycling bin",
    },
    "trash": {
        "action": "General waste",
        "guidance": "Place in general waste unless local rules identify a special disposal or recycling stream.",
        "bin": "General waste bin",
    },
}


def get_recommendation(class_name: str, confidence: float) -> Dict[str, object]:
    """Return action and guidance, with an uncertainty warning below the threshold."""
    if class_name not in RECOMMENDATIONS:
        raise ValueError(f"Unknown waste class: {class_name}")

    result = dict(RECOMMENDATIONS[class_name])
    result["class_name"] = class_name
    result["confidence"] = confidence
    result["is_uncertain"] = confidence < cfg.inference.CONFIDENCE_THRESHOLD

    if result["is_uncertain"]:
        result["warning"] = (
            "Low confidence prediction. Verify the item manually before disposing of it."
        )
    else:
        result["warning"] = None

    return result
