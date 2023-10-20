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


def normalize_label_value(label: str, label_list: List[str], lower_bound: float = 1.0, upper_bound: float = 10.0) \
        -> float:
    """
    Assign a normalized value between lower_bound and upper_bound to a string label.

    :param label: The label to normalize
    :param label_list: List of possible labels, each corresponding to a 1-based index
    :param lower_bound: The lower bound of the normalization
    :param upper_bound: The upper bound of the normalization

    :return: Normalized value for the given label
    """

    # Get the 1-based index of the label
    try:
        label_index = label_list.index(label) + 1
    except ValueError as e:
        raise ValueError(f'Label "{label}" not found in the list of labels.') from e

    # Total number of labels
    total_labels = len(label_list)

    # Normalize the value
    normalized_value = lower_bound + (upper_bound - lower_bound) * ((label_index - 1) / (total_labels - 1))

    return normalized_value
