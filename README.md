# Decision Assistant

A GPT-based chatbot that helps you make decisions. It helps you identify relevant criteria, assign weights to the criteria based on your own judgement and preferences, come up with relevant alternatives to choose from, research each alternative autonomously, and finally rank the alternatives based on the criteria and your preferences so you can make the best decision possible based on concrete researched data.

## How It Works

Based on [TOPSIS](https://robertsoczewica.medium.com/what-is-topsis-b05c50b3cd05):

### 1. **Goal Identification:**

- The user starts the process by giving the chatbot a decision goal. This could be anything like for example "Which
  smartphone should I buy?" or "Which candidate should I vote for in the next presidential elections?".

### 2. **Criteria Development:**

- With chatbot assistance, the user identifies and prioritizes the criteria for evaluating the alternatives.
- The chatbot should guide the process by first suggesting an initial set of criteria, and asking for feedback.
- Once agreed on the set of criteria, the chatbot moves to prioritization of the criteria and assigning weights to each
  that captures the intricacies and relative importance of the user preferences.
- The criterion's high value should always represent a better value than the low value and therefore should be named appropriately. Instead of "Price" use "Affordability", for example.
- The chatbot should also suggest a 5 point scale for each criterion. For example "Affordability" could be "Very Expensive", "Expensive", "Moderate", "Cheap", "Very Cheap".
- The chatbot should also keep a set of notes about how to assign the labels to each value of a criterion for the conversion to values later on. For example, "Very Expensive" could be "More than $1000", "Expensive" could be "Between $500 and $1000", etc. In case of subjective criteria like Price & affordability, the chatbot should inquire the user at this stage and keep a mapping table from the label to the explanation of how to assign the value.
- The chatbot should mark each criterion as objective or subjective.

### 3. **Criteria Prioritization**

- Since the user is most likely not a technical person, the chatbot then presents a set of hypothetical examples,
  one-by-one, to the user for evaluation.
- Each example should represent a meaningful comparison of two alternatives in various scenarios in a way that would
  maximally capture the user's preferences.
- The bot should give enough (smart) examples to the user to be able to make an accurate decision about the relative value of each
  criterion. However, it should aim to also minimize the number of questions to the user. The choice of questions should be strategic and smart to achieve this goal.
- Once the chatbot has collected enough data about the user's preferences and is confident about the relative value of
  each criterion, it assigns weights (1-100) to each criterion and presents them to the user for review.
- Once reviewed and approved, the chatbot moves on to the next step.

### 4. **Alternatives Listing:**

- User, with the aid of the chatbot, lists the items (alternatives) to be evaluated.

### 5. **Research Phase:**

#### 5.1. **Coming Up With Automated Research Queries:**

- Chatbot autonomously comes up with a set of queries for each alternative and each criterion.
- The queries should be smart and relevant, and should be able to capture the essence of the criterion based the scale and how to assign values.
- The queries should lead to answers that can be used to assign a value to each criterion for each alternative.
- There is a distinction between objective and subjective criteria. Objective factors are much more suited to these queries. However, subjective factors are impossible to capture through online queries - by definition. Therefore, online automated queries are only relevant for objective factors.

#### 5.2. **Researching Criteria:**

- Chatbot autonomously conducts research on each alternative for each sub-criteria, leveraging both web resources and
  its own knowledge to generate summaries and conclusions - based on the queries it came up with earlier (for consistency).
- The compiled information is presented to the user for review. In case of subjective criterion, there might not be any information available. In that case, the bot should just inquire the user about what it thinks about the criteria by asking guiding questions to extract as much information as possible from the user in order to be able to map the labels to values later on correctly.
- Based on the possible labels for each criterion, and the explanations provided by the user, the chatbot should also autonomously assign a numerical value to each criterion for each alternative (there are 5 ordinal labels assigned a value 1-5 accordingly).
- User and chatbot iterate until all information is validated and complete.

### 6. **Automated TOPSIS Calculations:**

- Construct a matrix consisting of M alternatives and N criteria called the "Evaluation Matrix" using the numerical data for each alternative and criteria.
- Normalizes the evaluation matrix so that each value is between 0 and 1 (higher is better)
- Calculate the weighted normalized decision matrix.
- Determine the ideal and negative-ideal solutions.
- Calculate the Euclidean distances from the ideal solution and negative-ideal solution.
- For each alternative, calculate similarity to the worst alternative given by: `S = D- / (D+ + D-)`, where `D+` is the
  distance to the ideal solution and `D-` is the distance to the negative-ideal solution.
- Rank the alternatives based on their TOPSIS score in descending order.

### 7. **Presentation:**
- Present the final rankings for each alternative, ordered by the best fit for the user.
- It should be in a table format, with the alternatives as rows and the criteria as columns, and a final column for the
  overall score based on TOPSIS. Criteria should be ordered by their weights descending. Also, values should be textual and meaningful, not numbers at this stage (for example, 0/10 in affordability would be equivalent to "very expensive"). The overall score should be a percentage, and the alternatives should be ordered by their overall score descending.