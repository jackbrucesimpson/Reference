# --------------------
# Post process with R
# Author: Jack Simpson
# --------------------

rm(list = ls())

# libraries
library(tidyverse)
library(data.table)
library(furrr)

# use multicore
plan(multiprocess)

# define directories
str_base_directory <- 'results'
str_time_period <- 'Interval'

str_search <- file.path(str_base_directory, 'model_1', str_time_period)

str_regex <- 'ST Region'

list_csv <- list.files(path = str_search, pattern = str_regex)

dt_id2name <- read_csv(file.path(str_base_directory, 'model_1', 'id2name.csv'))

model_name <- dt_id2name %>%
  filter(class == 'Model')
  
model_name <- model_name$name[1]

dt <- tibble(csvs = list_csv) %>%
  mutate(filenames = file.path(str_search, csvs),
         model_name = model_name) %>%
  separate(model_name, c('blank','esoo_label','scenario','horizon',
                         'poe','refyear','model_type','outage',
                         'test_label'), sep='_') %>%
  mutate(csvs = gsub('\\.csv','', csvs)) %>%
  separate(csvs, c('class', 'metric'), sep='\\.') %>%
  separate(class, c('phase', 'class'), sep=' ') %>%
  mutate(id = parse_number(class), 
         class = gsub('\\(.*', '', class)) %>%
  left_join(dt_id2name) %>%
  group_by(phase, class, metric, filenames, poe, refyear, name) %>%
  nest %>%
  mutate(data2 = future_map(filenames, fread, .progress = TRUE)) %>%
  select(-data) %>%
  unnest() %>%
  ungroup() %>%
  select(-filenames) %>%
  spread(metric, VALUE)

dt_one <- dt$data2[[1]]

str_filename <- dt$filenames[1]



