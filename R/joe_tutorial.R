rm(list = ls())

library(tidyverse)
library(data.table)
library(lubridate)

df <- fread('../data/trading.csv') %>%
  select(SETTLEMENTDATE, REGIONID, RRP) %>%
  mutate(SETTLEMENTDATE = ymd_hms(SETTLEMENTDATE))

group_df <- df %>%
  group_by(REGIONID) %>%
  summarise(MeanRRP = mean(RRP))

group_df <- df %>%
  group_by(REGIONID) %>%
  mutate(MeanRRP = mean(RRP))

fig <- group_df %>%
  ggplot(aes(x = SETTLEMENTDATE, y = RRP, color=REGIONID)) +
  geom_line() +
  facet_grid(REGIONID~.)

fig

fig <- group_df %>%
  ggplot(aes(x = SETTLEMENTDATE, y = RRP, color=REGIONID)) +
  geom_line() +
  facet_grid(REGIONID~MeanRRP)

fig

floor_df <- df %>%
  mutate(HOURLY = floor_date(SETTLEMENTDATE, 'hour')) %>%
  group_by(HOURLY, REGIONID) %>%
  summarise(HOURLYRRP = mean(RRP))

hour_df <- df %>%
  mutate(HOUR = hour(SETTLEMENTDATE)) %>%
  group_by(HOUR, REGIONID) %>%
  summarise(HOURRRP = mean(RRP))

region = 'NSW1'

create_fig <- function(region) {
  fig <- hour_df %>%
    filter(REGIONID == region) %>%
    ggplot(aes(x = HOUR, y = HOURRRP, color=REGIONID)) +
    geom_line() +
    theme_bw()
  
  ggsave(paste0(region, '.png'), fig, width = 15, height = 10)
}

regions <- unique(hour_df$REGIONID)

map(regions, create_fig)

create_fig('VIC1')

fig

