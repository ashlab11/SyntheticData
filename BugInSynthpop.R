library(haven)
library(glue)
library(forcats) #handles factors
library(stats)
library(ggplot2)
library(MASS)

cov <- 0.4
alpha <- 100
beta <- -100

data <- mvrnorm(1000, mu = c(0, 0), Sigma = matrix(c(1, cov,
                                                 cov, 1), 
                                                 nrow = 2, 
                                                 ncol = 2, byrow = TRUE)) %>% 
  as.data.frame() %>% 
  rename(X_1 = V1, X_2 = V2) %>% 
  mutate(Y = alpha * X_1 + beta * X_2 + rnorm(100, mean = 0, sd = 1))

data_alternate <-  rnorm(1000, 0, 1) %>% as.data.frame() %>% rename(X_1 = ".") %>% 
  mutate(X_2 = cov * X_1 + rnorm(1000, 0, 1), 
         Y = alpha * X_1 + beta * X_2 + rnorm(1000, 0, 1))

predictor_matrix <- matrix(c(0, 0, 0, 
                             1, 0, 0, 
                             1, 0, 0), 
                           nrow = 3, 
                           ncol = 3, 
                           byrow = TRUE)


model_pre_synth <- lm(Y ~ X_1 + X_2, data = data_alternate)

#RUN THIS A BUNCH OF TIMES, AND SEE THE ESTIMATE FOR X_2! 
synthesized <- syn(data, predictor.matrix = predictor_matrix, method = "norm", print.flag = FALSE,
                   seed = NA)$syn
model_post_synth <- lm(Y ~ X_1 + X_2, data = synthesized)
print(summary(model_post_synth))