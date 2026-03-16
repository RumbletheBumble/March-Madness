library(caret)  # for createFolds

set.seed(42)  # for reproducibility
folds <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)

cv_results <- lapply(folds, function(val_idx) {
  train_idx <- setdiff(seq_len(nrow(X)), val_idx)
  
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_val   <- X[val_idx, ]
  y_val   <- y[val_idx]
  
  # Train weights using constrOptim
  theta <- rep(1 / ncol(X), ncol(X))
  ui <- diag(ncol(X))
  ci <- rep(0, ncol(X))
  
  loss_fn <- function(w, X, y) {
    preds <- ifelse(X %*% w >= 0.5, 1, 0)
    mean(preds != y)
  }
  
  opt <- constrOptim(
    theta = theta,
    f = loss_fn,
    grad = NULL,
    ui = ui,
    ci = ci,
    X = X_train,
    y = y_train
  )
  
  w_opt <- opt$par / sum(opt$par)  # normalize weights
  preds <- ifelse(X_val %*% w_opt >= 0.5, 1, 0)
  acc <- mean(preds == y_val)
  
  list(weights = w_opt, accuracy = acc)
})

# Get validation accuracies
cv_accuracies <- sapply(cv_results, function(res) res$accuracy)
mean_accuracy <- mean(cv_accuracies)

# Average weights across folds
avg_weights <- Reduce(`+`, lapply(cv_results, `[[`, "weights")) / length(cv_results)

print(paste("Cross-validated ensemble accuracy:", round(mean_accuracy, 4)))
data.frame(Model = model_cols, Weight = round(avg_weights, 4))

library(ggplot2)

accuracy_df <- data.frame(
  Fold = paste0("Fold ", seq_along(cv_accuracies)),
  Accuracy = cv_accuracies
)

ggplot(accuracy_df, aes(x = Fold, y = Accuracy)) +
  geom_col(fill = "#60A5FA") +
  ylim(0, 1) +
  labs(title = "Cross-Validation Accuracy by Fold",
       y = "Validation Accuracy",
       x = "Fold") +
  theme_minimal()

# Combine weights into long format for plotting
weights_matrix <- do.call(rbind, lapply(cv_results, `[[`, "weights"))
colnames(weights_matrix) <- model_cols

weights_long <- reshape2::melt(as.data.frame(weights_matrix))
names(weights_long) <- c("Model", "Weight")
weights_long$Fold <- rep(paste0("Fold ", seq_along(cv_results)), each = length(model_cols))

ggplot(weights_long, aes(x = Model, y = Weight, fill = Fold)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Optimized Model Weights Across Folds",
       y = "Weight",
       x = "Model") +
  theme_minimal() +
  scale_fill_brewer(palette = "Blues")

X_df <- as.data.frame(X)
data_cv <- cbind(X_df, y = as.factor(y))  # y must be a factor for classification

set.seed(42)  # For reproducibility

cv_control <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE
)

# Convert outcome to factor with labels for caret
data_cv$y <- factor(ifelse(data_cv$y == 1, "Win", "Lose"))

log_cv_model <- train(
  y ~ .,
  data = data_cv,
  method = "glm",
  family = "binomial",
  trControl = cv_control,
  metric = "Accuracy"
)

print(log_cv_model)
