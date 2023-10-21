from typing import Callable, List, TypeVar, Tuple, Dict, Optional

import numpy as np
from research.topsis import topsis

T = TypeVar('T')


def topsis_score(items: List[T], weights: Dict[str, float], value_mapper: Optional[Callable[[T, str], float]] = None,
                 best_and_worst_solutions: Optional[Tuple[T, T]] = None) \
        -> List[float]:
    assert sum(weights.values()) != 0.0, 'Sum of weights cannot be zero.'
    assert all([0.0 <= weight for weight in weights.values()]), 'Weights must be non-negative.'

    if value_mapper is None:
        value_mapper = lambda item, criterion: item[criterion]

    criteria = list(weights.keys())

    values = np.array([[value_mapper(item, criterion) for criterion in criteria] for item in items])
    weights = np.array([weights[criterion] for criterion in criteria])
    benefit_or_cost = np.array([1 for _ in range(len(weights))])

    if best_and_worst_solutions is not None:
        best_and_worst_solutions = np.array(
            [[value_mapper(item, criterion) for criterion in criteria] for item in best_and_worst_solutions])

    decision = topsis(values, weights, best_and_worst_solutions=best_and_worst_solutions)

    return decision


def normalize_label_value(label: str, label_list: List[str], lower_bound: float = 0.0, upper_bound: float = 1.0):
    try:
        index = label_list.index(label)
    except ValueError:
        return "Label not found in the list"

    min_val = 0
    max_val = len(label_list) - 1

    normalized_value = lower_bound + ((upper_bound - lower_bound) * (index - min_val) / (max_val - min_val))

    return normalized_value
