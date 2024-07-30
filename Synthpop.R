# Load necessary libraries
library(synthpop)
library(tidyverse)
library(haven)
library(glue)
library(forcats) #handles factors




get_synthetic_data <- function(file_path) {
  #Gets 5 synthetic datasets
  dataset <- read_csv(glue("Datasets/{file_path}")) %>%
    syn(m = 5) %>% 
    pluck("syn")
  
  splits <- str_split(file_path, '_|\\.')[[1]] # Will look like c("housing", "mar", "test", "3", "csv")
  
  map(1:5, function(run) {
    output_path <- glue("Datasets/synthpop_{splits[1]}_{splits[2]}_train_{splits[4]}_{run}.csv")
    write_csv(dataset %>% pluck(run), output_path)
  })
  
  return()
}

files <- list.files("Datasets") %>% keep(~ str_detect(., "train") & !str_detect(., "synth"))


map(files, get_synthetic_data)
