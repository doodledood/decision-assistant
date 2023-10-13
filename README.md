# Decision Assistant

## How It Works

Based on AHP:

Absolutely. Here's the revised plan based on the simplified and user-centric approach you've outlined:

### 1. **Goal Identification:**
   - Chatbot engages with the user to ascertain the ultimate goal of the evaluation process.

### 2. **Criteria and Sub-Criteria Development:**
   - With chatbot assistance, the user identifies and prioritizes the criteria and possibly sub-criteria for evaluating the alternatives.
   - Utilize the intensity scale and simplicity mapping for prioritizing criteria and sub-criteria:
     - *Intensity Scale and Reciprocal Values Table*
     - *Simplicity Mapping: Verbal Judgment to Intensity*

### 3. **Alternatives Listing:**
   - User, with the aid of the chatbot, lists the items (alternatives) to be evaluated.

### 4. **Research Phase:**
   - Chatbot autonomously conducts research on each alternative for each sub-criteria, leveraging both web resources and its own knowledge to generate summaries and conclusions.

### 5. **Information Validation:**
   - The compiled information is presented in a table for user review.
   - User and chatbot iterate until all information is validated and complete.

### 6. **Automated Pairwise Comparisons and AHP Calculations:**
   - Chatbot performs automated pairwise comparisons of criteria, sub-criteria, and alternatives based on the compiled information.
   - Constructs comparison matrices, calculates priority vectors, and computes global weighted sums using AHP methodology:
     - Formulas:
       - \( \text{Comparison Matrix Construction: based on intensity values derived from research summaries.} \)
       - \( \text{Priority Vector Calculation: either through the Perron-Frobenius eigenvector method or geometric mean method.} \)
       - \( \text{Global Weighted Sum Calculation: } \Sigma \text{ (priority vector of (sub-)criterion × priority vector of alternative for that (sub-)criterion)} \)
       - \( \text{Consistency Ratio (CR) Calculation: } CR(A) = \frac{CI(A)}{RI_n} \) where \( CI(A) = \frac{(\lambda_{\text{max}} - n)}{(n - 1)} \)
   - Ensures consistency with a Consistency Ratio (CR) check and iteratively refines comparisons if needed.

### 7. **Ranking and Presentation:**
   - Rank the alternatives based on their global weighted sum, with higher sums indicating better alignment with the goal.
   - Present the final rankings for each alternative, ordered by the best fit for the user.

### Tables

#### Intensity Scale and Reciprocal Values

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

#### Simplicity Mapping: Verbal Judgment to Intensity

| Verbal Judgment     | Intensity |
|---------------------|-----------|
| Much More Important | 9         |
| More Important      | 5         |
| Equal Importance    | 1         |
| Less Important      | 1/5       |
| Much Less Important | 1/9       |

### Example Goals
* Decide who to vote for in the next presidential elections
* Choose the best smartphone to buy
* Choose the best stock to invest in