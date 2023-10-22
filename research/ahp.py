from typing import Dict, Tuple, Optional
import ahpy


def ahp_score(criteria_comparisons: Dict[Tuple[str, str], float],
              criteria_alternative_scores: Dict[str, Dict[str, float]]) -> \
        Dict[str, float]:
    criteria = ahpy.Compare('Criteria', criteria_comparisons)

    for criterion_name, alternative_scores in criteria_alternative_scores.items():
        criteria.add_children([ahpy.Compare(criterion_name, alternative_scores)])

    return criteria.target_weights
