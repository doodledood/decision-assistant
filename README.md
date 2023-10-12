# Decision Assistant

## How It Works

Based on AHP:

### 1. **Goal Identification:**

- Chatbot engages with the user to ascertain the ultimate goal of the evaluation process.

### 2. **Alternatives Listing:**

- User, with the aid of the chatbot, lists the items (alternatives) to be evaluated.

### 3. **Criteria Development:**

- With chatbot assistance, the user identifies the criteria and possibly sub-criteria for evaluating the alternatives.
- Chatbot helps construct the hierarchy of criteria and sub-criteria.

### 4. **Pairwise Comparisons:**

- User performs pairwise comparisons of criteria, sub-criteria, and alternatives under the guidance of the chatbot.
- The comparisons are based on the provided intensity scale:

| Intensity | Description                                                                                                                               | Reciprocal Value                        |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| 1         | Equal Importance: Both activities contribute equally to the objective.                                                                    | 1 (Equal Importance)                    |
| 2         | Weak or slight                                                                                                                            | 1/2                                     |
| 3         | Moderate Importance: Experience and judgment slightly favor one activity over another.                                                    | 1/3                                     |
| 4         | Moderate plus                                                                                                                             | 1/4                                     |
| 5         | Strong Importance: Experience and judgment strongly favor one activity over another.                                                      | 1/5                                     |
| 6         | Strong plus                                                                                                                               | 1/6                                     |
| 7         | Very Strong Importance: An activity is favored very strongly over another, its dominance demonstrated in practice.                        | 1/7                                     |
| 8         | Very, very strong                                                                                                                         | 1/8                                     |
| 9         | Extreme Importance: The evidence favoring one activity over another is of the highest possible order of affirmation.                      | 1/9                                     |
| 1.1 - 1.9 | For activities that are very close in importance, where assigning a value may be challenging but still indicative of relative importance. | Reciprocal within range (1/1.1 - 1/1.9) |

- Note regarding Reciprocal Value: If activity i has one of the above non-zero numbers assigned to it when compared with
  activity j, then j has the reciprocal value when compared with i.
- For simplicity, map 5 levels of verbal judgments to intensities:

| Verbal Judgment     | Intensity |
|---------------------|-----------|
| Much More Important | 9         |
| More Important      | 5         |
| Equal Importance    | 1         |
| Less Important      | 1/5       |
| Much Less Important | 1/9       |

#### 5. **Comparison Matrix Construction:**

- Chatbot automates the construction of comparison matrices based on the pairwise comparisons for both criteria and
  sub-criteria.

#### 6. **Priority Vector Calculation:**

- Chatbot calculates the priority vectors from the comparison matrices, either through the Perron-Frobenius eigenvector
  method or geometric mean method, for both criteria and sub-criteria.

#### 7. **Pairwise Comparisons (Alternatives):**

- Perform pairwise comparisons for each sub-criteria among alternatives.

#### 8. **Comparison Matrix Construction (Alternatives):**

- Construct the comparison matrix for each sub-criteria among alternatives.

#### 9. **Priority Vector Calculation (Alternatives):**

- Calculate the priority vector for each sub-criteria among alternatives.

#### 10. **Global Weighted Sum Calculation:**

- For each alternative, calculate the global weighted sum by multiplying the priority vector of each (sub-)criterion
  with the corresponding priority vector of the alternative for that (sub-)criterion, and summing these products across
  all (sub-)criteria.

#### 11. **Ranking:**

- Rank the alternatives based on their global weighted sum, with higher sums indicating better alignment with the goal.

#### 12. **Consistency Ratio (CR) Calculation:**

- Chatbot calculates the Consistency Index (CI) and Consistency Ratio (CR) using the formulae:
    - \[ \text{CI}(A) = \left( \lambda_{\text{max}} - n \right) / (n - 1) \]
    - \[ \text{CR}(A) = \text{CI}(A) / \text{RI}_n \]
- Where:
    - \( \lambda_{\text{max}} \) is the maximum eigenvalue of the pairwise comparison matrix,
    - \( n \) is the number of alternatives,
    - \( \text{RI}_n \) is the average random index value for a matrix of size \( n \) (provided by Prof. Saaty).

#### 13. **Iterative Refinement:**

- If CR value is unacceptable (CR > 0.1), chatbot guides the user to revise the pairwise comparisons for better
  consistency.

#### 14. **Documentation and Review (Optional):**

- Chatbot documents each step, calculations, and rankings, providing a transparent evaluation process for the user.

#### 15. **User Engagement and Feedback (Optional):**

- Chatbot captures user feedback for continuous improvement in future evaluations.

#### 16. **Support and Guidance:**

- Chatbot provides support and guidance throughout the evaluation process, clarifying any doubts and ensuring the user
  feels confident in the evaluations and rankings.
