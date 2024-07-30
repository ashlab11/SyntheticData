library(synthpop)
library(tidyverse)
library(haven)
library(glue)
library(forcats) #handles factors
library(stats)
library(ggplot2)

#Simple function. Defines X_1 ~ N(0, 1), X_2 = 2X_1 + N(0, 1), Y = 3X_1 + 4X_2 + N(0, 1)
#Then runs synthesization on Y ~ X_1, X_2 ~ X_1. We want to show that removing some cols from 
#Synthesization can drastically change coefficients
compare_synthesized_linreg <- function(alpha, beta, gamma, method) {
  # Generate dataframe
  X_1 <- rnorm(100)
  X_2 <- gamma * X_1 + rnorm(100)
  Y <- alpha * X_1 + beta * X_2 + rnorm(100)
  
  data <- data.frame(X_1 = X_1, X_2 = X_2, Y = Y)
  
  # Fit linear regression
  model_pre_synth <- lm(Y ~ X_1 + X_2, data = data)
  coef_pre_synth <- model_pre_synth$coef
  

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
  coef_post_synth <- model_post_synth$coef

  return (list(coef_pre_synth, coef_post_synth))
}

main <- function(cov, alpha, beta) {
  #Grabbing models
  models <- replicate(100, compare_synthesized_linreg(cov, alpha, beta))

  original_X_1 <- unlist(models[1,] %>% map(~ pluck(.x, 2)))
  original_X_2 <- unlist(models[1,] %>% map(~ pluck(.x, 3)))
  synthesized_X_1 <- unlist(models[2,] %>% map(~ pluck(.x, 2)))
  synthesized_X_2 <- unlist(models[2,] %>% map(~ pluck(.x, 3)))
  #p_val_original <- mean(models[3,] < 0.05)
  #p_val_synth <- mean(models[4,] < 0.05)


  # X_1_data <- tibble(
  #   value = c(original_X_1, synthesized_X_1),
  #   group = rep(c("Original", "Synthesized"), c(length(original_X_1), length(synthesized_X_1)))
  # )
  # 
  # X_2_data <- tibble(
  #   value = c(original_X_2, synthesized_X_2),
  #   group = rep(c("Original", "Synthesized"), c(length(original_X_2), length(synthesized_X_2)))
  # )
  # 
  # combined_data <- bind_rows(
  #   X_1_data %>% mutate(variable = "X1"),
  #   X_2_data %>% mutate(variable = "X2")
  # )
  
  # Create the faceted plot
  # plot <- ggplot(combined_data, aes(x = group, y = value)) +
  #   geom_boxplot(alpha = 0.7) +
  #   facet_wrap(~ variable, scales = "free_y", nrow=1) +
  #   theme_minimal() +
  #   labs(title = "Comparison of X1 and X2 Coefficient Distributions",
  #        subtitle = glue("Cov(X1, X2) = {cov}"),
  #        x = "Type of Analysis",
  #        y = "Coefficient") +
  #   theme(strip.background = element_rect(fill = "lightgrey"),
  #         strip.text = element_text(face = "bold"))
  
  return (list(original_X1 = mean(original_X_1), 
               original_X2 = mean(original_X_2), 
               synth_X_1 = mean(synthesized_X_1), 
               synth_X_2 = mean(synthesized_X_2)))
}

#Analyzing how changes in cov and stuff affect
alphas <- c(-2, 0, 2)
betas <- c(-2, 0, 2)
gammas <- c(-2, 0, 2)
methods <- c("CART", "norm")

results <- map(covs, function(cov) return(
  map(alphas, function(alpha) return (
    map(betas, function(beta) return(
      map(methods, function(method) return (main(cov, alpha, beta, method)))
  ))
), .progress = TRUE)))
               
results_df <- results %>%
  flatten_df() 

list_index <- map(covs, function(cov) return(
  map(alphas, function(alpha) return (
    map(betas, function(beta) return(list(cov, alpha, beta)))
  )))
)

list_index_flat <- list_index %>%
  flatten() %>%
  flatten() %>%
  map_df(~ as_tibble(.x, .name_repair = "minimal"), .id = NULL)

results_df <- bind_cols(list_index_flat, results_df) %>% 
  rename(cov = ...1, alpha = ...2, beta = ...3)

#Getting important linear regressions on X_2
X_2linreg <- lm(synth_X_2 ~ alpha + beta, data = results_df %>% filter(cov == 0))
