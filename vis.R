#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
v_num <- args[1]
if (length(args) == 1) {
    cutoff = 10
} else {
    cutoff = strtoi(args[2])
}

metrics_path <- paste(
    "logs",
    "lightning_logs",
    paste("version_", v_num, sep = ""),
    "metrics.csv",
    sep = "/"
)

metrics <- read_csv(metrics_path)

metrics %>%
    pivot_longer(!c(epoch, step), names_to = "loss", values_to = "value") %>%
    group_by(epoch, loss) %>%
    summarise(value = mean(value)) %>%
    filter(epoch >= cutoff) %>%
    ggplot() +
        aes(x = epoch, y = value) +
        facet_wrap(~loss, scales = "free_y") +
        ylim(0, NA) +
        geom_line() +
        geom_smooth() +
        xlab("Epoch") +
        ylab("Loss")

ggsave(paste("plots", paste(args[1], ".png", sep = ""), sep = "/"), width = 30, height = 16, units = "cm", dpi = 600)
