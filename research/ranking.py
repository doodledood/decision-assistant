from typing import Callable, List, TypeVar, Tuple, Dict, Optional

import numpy as np
from research.topsis import topsis

T = TypeVar('T')


def topsis_score(items: List[T], weights: Dict[str, float], value_mapper: Optional[Callable[[T, str], float]] = None) \
        -> List[float]:
    assert sum(weights.values()) != 0.0, 'Sum of weights cannot be zero.'
    assert all([0.0 <= weight for weight in weights.values()]), 'Weights must be non-negative.'

    if value_mapper is None:
        value_mapper = lambda item, criterion: item[criterion]

    criteria = list(weights.keys())

    values = np.array([[value_mapper(item, criterion) for criterion in criteria] for item in items])
    weights = np.array([weights[criterion] for criterion in criteria])
    benefit_or_cost = np.array([1 for _ in range(len(weights))])

    decision = topsis(values, weights, benefit_or_cost)
    decision.calc()

    return decision.C
