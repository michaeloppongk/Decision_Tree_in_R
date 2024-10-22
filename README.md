### Project Overview: Predicting Personal Loan Acceptance Using KNN and Decision Tree Models

In this project, we aim to develop machine learning models that predict whether a customer will accept a personal loan offer. We use the **UniversalBank dataset**, which contains demographic and financial information about 5000 customers, including whether they accepted a personal loan (the target variable). Two machine learning models were explored: K-Nearest Neighbors (KNN) and Decision Tree, with the goal of maximizing accuracy.

### Data Preprocessing

Before building the models, several preprocessing steps were performed:
- The **target variable** (`PersonalLoan`) was converted into a **factor** to indicate that it is categorical (loan acceptance: 0 = No, 1 = Yes).
- We created **dummy variables** for the `Education` feature (creating `Education_1` and `Education_2` to represent different education levels) to use them as binary predictors in the KNN model.
- For numerical features like `Age`, `Experience`, `Income`, and `Family`, **range normalization** was applied using the `preProcess` function from the `caret` package to scale these variables between 0 and 1, improving the performance of distance-based algorithms like KNN.

### K-Nearest Neighbors (KNN) Model

#### Model 1: Basic KNN with k = 3
The first KNN model was built using 6 predictors: `Age`, `Experience`, `Income`, `Family`, `Education_1`, and `Education_2`. We chose **k = 3** as the number of neighbors. The results were highly accurate:
- **Accuracy**: 98.72%
- **Sensitivity**: 98.67%
- **Specificity**: 99.29%
- **Kappa**: 0.922 (indicating strong agreement)

This model demonstrated high accuracy, but a relatively low **negative predictive value (NPV)** of 87.29% means that when the model predicts a customer will not accept the loan, there is a moderate chance of being wrong.

#### Model 2: Tuning the KNN Model with Cross-Validation
To improve upon Model 1, we applied **10-fold cross-validation** with a range of `k` values (from 1 to 17) to find the optimal number of neighbors. The best performing model was found with **k = 3**, with:
- **Accuracy**: 97.76%
- **Kappa**: 0.861 (still indicating strong agreement)

#### Model 3: Extended Tuning with Wider Range of k
An extended tuning approach was applied by allowing the algorithm to automatically search for the best `k` over a wider range (using `tuneLength = 50`). The final optimal value was **k = 5**, with:
- **Accuracy**: 97.62%
- **Kappa**: 0.847

The model's performance slightly decreased with higher `k` values, as expected in KNN, where smaller `k` values tend to capture local data patterns better.

#### Best Model Selection
After testing different values of `k`, we selected **k = 5** as the best model based on performance metrics:
- **Accuracy**: 98.28%
- **Sensitivity**: 98.24%
- **Specificity**: 98.76%
- **Kappa**: 0.8934

This model provides the most balanced trade-off between overfitting (lower k values) and generalization (higher k values).

### Decision Tree Model

#### Model 1: Tuning the Complexity Parameter (cp)
We also built a **Decision Tree** model using the `rpart` method and tuned its **complexity parameter (cp)** using 10-fold cross-validation. The results showed:
- **Accuracy**: ~97.66%
- **Kappa**: ~0.85

We tuned the cp parameter over a range from 0 to 0.1, finding that the best performance was obtained with a **cp value of 0.02**.

### Model Evaluation & Comparison

| Metric          | KNN (k = 3) | KNN (k = 5) | Decision Tree (cp = 0.02) |
|-----------------|-------------|-------------|----------------------------|
| **Accuracy**    | 98.72%      | 98.28%      | 97.66%                     |
| **Sensitivity** | 98.67%      | 98.24%      | 97.66%                     |
| **Specificity** | 99.29%      | 98.76%      | 97.66%                     |
| **Kappa**       | 0.922       | 0.8934      | 0.85                       |

- **KNN (k = 5)** provides the best balance between accuracy and generalization, with an accuracy of 98.28%.
- The **Decision Tree** model, while slightly less accurate than KNN, offers good interpretability and can be a suitable choice for scenarios where model simplicity and transparency are more important than marginal gains in accuracy.

### Conclusion

The KNN model with **k = 5** was selected as the final model due to its high accuracy and balanced performance across metrics. However, the **Decision Tree** model offers a simpler and interpretable alternative for loan prediction tasks. Both models are highly accurate and capable of predicting personal loan acceptance with confidence.

