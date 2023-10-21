from typing import Optional, Tuple

import numpy as np


def topsis(decision_matrix: np.ndarray, weights: np.ndarray,
           best_and_worst_solutions: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    # Step 1: Normalize the decision matrix
    row_sums = np.sqrt(np.sum(np.square(decision_matrix), axis=0))
    normalized_matrix = decision_matrix / row_sums

    # Step 2: Weighted normalized decision matrix
    weighted_normalized_matrix = normalized_matrix * weights

    # Step 3: Determine the ideal and anti-ideal solutions
    if best_and_worst_solutions is not None:
        ideal_solution, anti_ideal_solution = best_and_worst_solutions

        # Normalize and weight the provided ideal and anti-ideal solutions
        ideal_solution = (ideal_solution / row_sums) * weights
        anti_ideal_solution = (anti_ideal_solution / row_sums) * weights
    else:
        ideal_solution = np.max(weighted_normalized_matrix, axis=0)
        anti_ideal_solution = np.min(weighted_normalized_matrix, axis=0)

    # Step 4: Calculate the Euclidean distances
    distances_to_ideal = np.linalg.norm(weighted_normalized_matrix - ideal_solution, axis=1)
    distances_to_anti_ideal = np.linalg.norm(weighted_normalized_matrix - anti_ideal_solution, axis=1)

    # Step 5: Calculate the similarity to the ideal solution
    similarity_to_ideal = distances_to_anti_ideal / (distances_to_ideal + distances_to_anti_ideal)

    return similarity_to_ideal
