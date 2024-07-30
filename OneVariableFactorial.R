library(synthpop)
library(tidyverse)
library(knitr)

compare_synthesized_linreg <- function(alpha, beta, gamma, method) {
  "This function takes in a alpha/beta/gamma/method and returns the p-values for"
  "Null Hy. that X_1 coefficients are the same pre/post synth"
  # Generate dataframe
  X_1 <- rnorm(100)
  X_2 <- gamma * X_1 + rnorm(100)
  Y <- alpha * X_1 + beta * X_2 + rnorm(100)
  
  data <- data.frame(X_1 = X_1, X_2 = X_2, Y = Y)
  
  # Fit linear regression
  model_pre_synth <- lm(Y ~ X_1 + X_2, data = data)
  coef_pre_synth <- summary(model_pre_synth)$coef
  
  # We want X_1 to be random, Y to be based on X_1, and X_2 to be based on X_1. Order of rows/cols is X_1, X_2, Y
  predictor_matrix <- matrix(c(0, 0, 0, 
                               1, 0, 0, 
                               1, 0, 0), 
                             nrow = 3, 
                             ncol = 3, 
                             byrow = TRUE)
  
  synthesize <- syn(data, predictor.matrix = predictor_matrix, method = method, print.flag = FALSE)
  synthesized_data <- synthesize$syn
  
  model_post_synth <- lm(Y ~ X_1 + X_2, data = synthesized_data)
  coef_post_synth <- summary(model_post_synth)$coef
  
  #We want to check whether or not the post-synth coefficient X_1/X_2 is stat sig 
  #different than pre-synth X_1/X_2
  
  pre_X1_estimate <- coef_pre_synth["X_1", "Estimate"]
  pre_X1_std <- coef_pre_synth["X_1", "Std. Error"]
  pre_X2_estimate <- coef_pre_synth["X_2", "Estimate"]
  pre_X2_std <- coef_pre_synth["X_2", "Std. Error"]
  
  post_X1_estimate <- coef_post_synth["X_1", "Estimate"]
  post_X1_std <- coef_post_synth["X_1", "Std. Error"]
  post_X2_estimate <- coef_post_synth["X_2", "Estimate"]
  post_X2_std <- coef_post_synth["X_2", "Std. Error"]  
  
  X1_zscore <- (pre_X1_estimate - post_X1_estimate) / sqrt(pre_X1_std^2 + post_X1_std^2)
  X2_zscore <- (pre_X2_estimate - post_X2_estimate) / sqrt(pre_X2_std^2 + post_X2_std^2)

  pval_X1 <- 2 * (1 - pnorm(abs(X1_zscore)))
  pval_X2 <- 2 * (1 - pnorm(abs(X2_zscore)))
  
  return(list(pval_X1, pval_X2))

}

get_pct_covered <- function(alpha, beta, gamma, method) {
  p_values <- replicate(100, compare_synthesized_linreg(alpha, beta, gamma, method))
  pct_X1_out <- mean(p_values[1, ] > 0.05)
  pct_X2_out <- mean(p_values[2, ] > 0.05)
  
  return (list(pct_X1_out, pct_X2_out))
}

alphas <- c(-2, 0, 2)
beta <- c(-2, 0, 2)
gammas <- c(-2, 0, 2)
methods <- c("cart", "norm")

combinations <- expand.grid(alpha = alphas, beta = betas, gamma = gammas, method = methods)

combinations <- combinations %>% 
  rowwise() %>% 
  mutate(covered = list(get_pct_covered(alpha, beta, gamma, method))) %>% 
  unnest_wider(covered, names_sep = "_") %>% 
  rename(covered_X1 = covered_1, 
         covered_X2 = covered_2)

reshaped_combinations <- combinations %>%
  arrange(method, beta, alpha, gamma) %>% 
  group_by(method, beta) %>% 
  summarize(values_X1 = list(covered_X1), 
            values_X2 = list(covered_X2)) %>% 
  rowwise() %>% 
  mutate(latex_X1 = paste(values_X1,collapse = " & "), 
         latex_X2 = paste(values_X2, collapse = " & "))
