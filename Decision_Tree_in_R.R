# Load necessary libraries
library(rpart)
library(rpart.plot)
library(caret)
library(readr)


# Load and prepare UniversalBank dataset
bank <- read.csv("UniversalBank.csv")
bank$PersonalLoan <- as.factor(bank$PersonalLoan)  # Convert target variable to factor

# Set seed for reproducibility
set.seed(1234)

# --- Model 1: Tuning the Complexity Parameter (cp) ---
# Train a decision tree model with cross-validation (cv) using a grid search for cp
tree_cp <- train(PersonalLoan ~ Age + Experience + Income + Family + Education,
                 data = bank,
                 method = "rpart",  # Using the rpart method to build a classification tree
                 metric = "Accuracy",  # Evaluating based on accuracy
                 trControl = trainControl(method = "cv", number = 10),  # 10-fold cross-validation
                 tuneGrid = expand.grid(cp = seq(0, 0.1, 0.01)))  # Tune complexity parameter

# Display the results and plot the cp tuning process
print(tree_cp)
plot(tree_cp)

# --- Model 2: Extended Tuning of Complexity Parameter ---
# Perform extended tuning with a larger range of cp values
tree_cp_extended <- train(PersonalLoan ~ Age + Experience + Income + Family + Education,
                          data = bank,
                          method = "rpart",
                          metric = "Accuracy",
                          trControl = trainControl(method = "cv", number = 10),
                          tuneLength = 50)  # Automatically select 50 cp values to tune

# Display results and plot the extended tuning
print(tree_cp_extended)
plot(tree_cp_extended)

# --- Model 3: Tuning the Maximum Depth ---
# Train a decision tree model by tuning the maximum tree depth (maxdepth)
tree_depth <- train(PersonalLoan ~ Age + Experience + Income + Family + Education,
                    data = bank,
                    method = "rpart2",  # Using rpart2 to tune tree depth
                    metric = "Accuracy",
                    trControl = trainControl(method = "cv", number = 10),
                    tuneLength = 5)  # Tune over a range of 5 depths

# Display results and plot the decision tree
print(tree_depth)
plot(tree_depth)
prp(tree_depth$finalModel, type = 1, extra = 1)  # Plot the final model with additional details

# --- Evaluate the Final Model ---
# Make predictions on the original dataset using the best tree model
bank$PersonalLoan_Pred <- predict(tree_depth, newdata = bank)

# Confusion matrix to evaluate model performance
confusionMatrix(bank$PersonalLoan, bank$PersonalLoan_Pred)

