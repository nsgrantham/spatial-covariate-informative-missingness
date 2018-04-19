library(dplyr)
library(lubridate)

data_dir <- file.path('pm-aod-analysis', 'data')
processed_data_dir <- file.path(data_dir, 'processed')
split_by_month_dir <- file.path(data_dir, 'split-by-month')

source(file.path('pm-aod-analysis', 'data', 'utils.R'))

read.csv_ <- read.csv.maker(processed_data_dir)

month_dirs <- list(
    '1'  = '01-jan',
    '2'  = '02-feb',
    '3'  = '03-mar',
    '4'  = '04-apr',
    '5'  = '05-may',
    '6'  = '06-jun',
    '7'  = '07-jul',
    '8'  = '08-aug',
    '9'  = '09-sep',
    '10' = '10-oct',
    '11' = '11-nov',
    '12' = '12-dec'
)

# Sm.csv, S.csv, and mapping.csv do not depend on dates, but write them to each
# month dir. A bit wasteful to store multiple copies of the same file, but doing
# so makes it easy for fit-model.R to run since all required data files live in 
# a single directory.
message("Reading processed Sm.csv, S.csv, and mapping.csv")
Sm <- read.csv_('Sm.csv')
S <- read.csv_('S.csv')
mapping <- read.csv_('mapping.csv')
message("Writing split-by-month Sm.csv, S.csv, and mapping.csv")
for (month_dir in month_dirs) {
    write.csv(Sm, file.path(split_by_month_dir, month_dir, 'Sm.csv'), 
              row.names = FALSE, quote = FALSE)
    write.csv(S, file.path(split_by_month_dir, month_dir, 'S.csv'),
              row.names = FALSE, quote = FALSE)
    write.csv(mapping, file.path(split_by_month_dir, month_dir, 'mapping.csv'),
              row.names = FALSE, quote = FALSE)
}

# For the rest of the data files, group by month(date) and write to month dirs.
write_to_month_dir <- function(df, filename) {
    month_dir <- month_dirs[[unique(df$month)]]
    write.csv(df %>% select(-one_of(c('month'))),  # do not write month column
              file.path(split_by_month_dir, month_dir, filename),
              row.names = FALSE, quote = FALSE)
    df
}
files <- c('zm.csv', 'z.csv', 'ym.csv', 'y.csv', 'Xm.csv', 'X.csv')
for (file in files) {
    message(paste("Reading processed", file))
    df <- read.csv_(file)
    message(paste("Writing split-by-month", file))
    df %>%
        transform(month = as.character(month(ymd(date)))) %>%
        group_by(month) %>%
        do(write_to_month_dir(., file))
}
