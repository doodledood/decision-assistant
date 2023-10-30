from typing import Dict, Tuple, Optional
import ahpy


def ahp_score(criteria_comparisons: Dict[Tuple[str, str], float],
              criteria_alternative_scores: Dict[str, Dict[str, float]],
              ideal_solution: Dict[str, float],
              non_ideal_solution: Dict[str, float]) -> \
        Dict[str, float]:
    criteria_alternative_scores = criteria_alternative_scores.copy()

    criteria = ahpy.Compare('Criteria', criteria_comparisons)

    for criterion_name, alternative_scores in criteria_alternative_scores.items():
        alternative_scores['#Ideal Solution#'] = ideal_solution[criterion_name]
        alternative_scores['#Non-Ideal Solution#'] = non_ideal_solution[criterion_name]

        criteria.add_children([ahpy.Compare(criterion_name, alternative_scores)])

    scores = criteria.target_weights
    ideal_score = scores['#Ideal Solution#']
    non_ideal_score = scores['#Non-Ideal Solution#']
    scores = {criterion_name: (score - non_ideal_score) / (ideal_score - non_ideal_score)
              for criterion_name, score in scores.items()
              if criterion_name not in ['#Ideal Solution#', '#Non-Ideal Solution#']}

    return scores
