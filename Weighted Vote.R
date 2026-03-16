# Load necessary libraries
library(readr)
library(dplyr)

# Load data
preds <- read_csv("Results.csv")
winners <- read_csv("Winners.csv")

# Define model columns and extract X and y
model_cols <- c("NetRank", "NetRankAdj", "AvgOppNet", "AvgOppNetAdj",
                "NetSos", "NetSosAdj", "Wab", "WabAdj")
X <- preds %>% select(all_of(model_cols)) %>% as.matrix()
y <- winners$Winners

# Loss function: binary classification error
weighted_vote_loss <- function(weights, X, y) {
  preds <- ifelse(X %*% weights >= 0.5, 1, 0)
  mean(preds != y)
}

# Number of models
n_models <- ncol(X)

# Initial weights (strictly inside bounds)
theta <- rep(1 / n_models, n_models)

# Constraints:
#   weights >= 0         (8 constraints)
#   sum(weights) >= 1    (1 constraint)
#   sum(weights) <= 1    (1 constraint, as -sum(weights) >= -1)

ui <- rbind(
  diag(n_models)      # sum(weights) <= 1 (as -sum >= -1)
)
ci <- rep(0, n_models)

# Test if theta is strictly feasible (should all be > ci)
feasibility_check <- ui %*% theta > ci
print(feasibility_check)

print(theta)
print(ui %*% theta)
print(ci)


# Run optimization
opt <- constrOptim(
  theta = theta,
  f = weighted_vote_loss,
  grad = NULL,
  ui = ui,
  ci = ci,
  X = X,
  y = y
)

opt_weights <- opt$par / sum(opt$par)
data.frame(Model = model_cols, Weight = round(opt_weights, 4))
# Predict outcomes
final_preds <- ifelse(X %*% opt_weights >= 0.5, 1, 0)

# Accuracy
accuracy <- mean(final_preds == y)
print(paste("Final ensemble accuracy:", round(accuracy, 4)))
individual_accuracies <- apply(X, 2, function(preds) mean(ifelse(preds >= 0.5, 1, 0) == y))
data.frame(Model = model_cols, Accuracy = round(individual_accuracies, 4))


library(ggplot2)

weights_df <- data.frame(
  Model = model_cols,
  Weight = opt_weights
)

ggplot(weights_df, aes(x = reorder(Model, Weight), y = Weight)) +
  geom_col(fill = "#3B82F6") +
  coord_flip() +
  labs(title = "Optimized Weights for Each Model",
       x = "Model",
       y = "Weight") +
  theme_minimal()

# Calculate individual model accuracies
individual_accuracies <- apply(X, 2, function(preds) {
  mean(ifelse(preds >= 0.5, 1, 0) == y)
})

# Combine into one data frame
accuracy_df <- data.frame(
  Model = model_cols,
  Accuracy = individual_accuracies
)

# Add ensemble
ensemble_accuracy <- mean(ifelse(X %*% opt_weights >= 0.5, 1, 0) == y)

accuracy_df <- rbind(
  accuracy_df,
  data.frame(Model = "Ensemble", Accuracy = ensemble_accuracy)
)

# Plot
ggplot(accuracy_df, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model == "Ensemble")) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("#93C5FD", "#1D4ED8"), guide = "none") +
  labs(title = "Model Accuracy Comparison",
       x = "Model",
       y = "Accuracy") +
  theme_minimal()

