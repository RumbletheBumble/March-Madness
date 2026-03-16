cor_matrix <- cor(X)
round(cor_matrix, 2)

library(corrplot)
corrplot(cor(X), method = "color", addCoef.col = "black", tl.col = "black")

individual_accuracies <- colMeans(X == y)
data.frame(Model = model_cols, Accuracy = individual_accuracies)

# Assuming X and y are already defined
X_df <- as.data.frame(X)
stack_model <- glm(y ~ ., data = X_df, family = "binomial")
summary(stack_model)  # Optional, to inspect weights

library(ggplot2)

coef_df <- data.frame(
  Model = names(coef(stack_model))[-1],
  Weight = coef(stack_model)[-1]
)

ggplot(coef_df, aes(x = reorder(Model, Weight), y = Weight)) +
  geom_col(fill = "#4ade80") +
  coord_flip() +
  labs(title = "Logistic Stacking Model Weights",
       x = "Model",
       y = "Coefficient (log-odds scale)") +
  theme_minimal()

# Logistic ensemble predictions
log_preds <- predict(stack_model, type = "response")
log_pred_classes <- ifelse(log_preds >= 0.5, 1, 0)
log_accuracy <- mean(log_pred_classes == y)

# Compare with earlier ensemble
ensemble_probs <- X %*% avg_weights
ensemble_preds <- ifelse(ensemble_probs >= 0.5, 1, 0)
constrained_accuracy <- mean(ensemble_preds == y)


# Create a bar chart
comparison_df <- data.frame(
  Ensemble = c("Constrained", "Logistic Stacking"),
  Accuracy = c(constrained_accuracy, log_accuracy)
)

ggplot(comparison_df, aes(x = Ensemble, y = Accuracy, fill = Ensemble)) +
  geom_col() +
  ylim(0, 1) +
  labs(title = "Ensemble Accuracy Comparison",
       y = "Accuracy",
       x = "") +
  theme_minimal() +
  scale_fill_manual(values = c("#60A5FA", "#4ADE80"))

cor_matrix <- cor(X)
round(cor_matrix, 2)

library(corrplot)
corrplot(cor(X), method = "color", addCoef.col = "black", tl.col = "black")

individual_accuracies <- colMeans(X == y)
data.frame(Model = model_cols, Accuracy = individual_accuracies)

# Assuming X and y are already defined
X_df <- as.data.frame(X)
stack_model <- glm(y ~ ., data = X_df, family = "binomial")
summary(stack_model)  # Optional, to inspect weights

library(ggplot2)

coef_df <- data.frame(
  Model = names(coef(stack_model))[-1],
  Weight = coef(stack_model)[-1]
)

ggplot(coef_df, aes(x = reorder(Model, Weight), y = Weight)) +
  geom_col(fill = "#4ade80") +
  coord_flip() +
  labs(title = "Logistic Stacking Model Weights",
       x = "Model",
       y = "Coefficient (log-odds scale)") +
  theme_minimal()

# Logistic ensemble predictions
log_preds <- predict(stack_model, type = "response")
log_pred_classes <- ifelse(log_preds >= 0.5, 1, 0)
log_accuracy <- mean(log_pred_classes == y)

# Compare with earlier ensemble
ensemble_probs <- X %*% avg_weights
ensemble_preds <- ifelse(ensemble_probs >= 0.5, 1, 0)
constrained_accuracy <- mean(ensemble_preds == y)


# Create a bar chart
comparison_df <- data.frame(
  Ensemble = c("Constrained", "Logistic Stacking"),
  Accuracy = c(constrained_accuracy, log_accuracy)
)

ggplot(comparison_df, aes(x = Ensemble, y = Accuracy, fill = Ensemble)) +
  geom_col() +
  ylim(0, 1) +
  labs(title = "Ensemble Accuracy Comparison",
       y = "Accuracy",
       x = "") +
  theme_minimal() +
  scale_fill_manual(values = c("#60A5FA", "#4ADE80"))

# Predictions from both models
log_preds <- ifelse(predict(stack_model, type = "response") >= 0.5, 1, 0)
cons_preds <- ifelse((X %*% avg_weights) >= 0.5, 1, 0)

# Create a contingency table
table_result <- table(log_preds == y, cons_preds == y)

# McNemar's Test
mcnemar.test(table_result)

mean(log_preds == y)         # Logistic stacking accuracy
mean(cons_preds == y)        # Constrained ensemble accuracy
sum((log_preds == y) & !(cons_preds == y))  # Wins for logistic
sum((cons_preds == y) & !(log_preds == y))  # Wins for constrained

table_result

