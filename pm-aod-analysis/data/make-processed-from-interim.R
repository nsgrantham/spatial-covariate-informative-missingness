library(dplyr)

data_fp <- file.path('pm-aod-analysis', 'data')
interim_data_fp <- file.path(data_fp, 'interim')
processed_data_fp <- file.path(data_fp, 'processed')

source(file.path('pm-aod-analysis', 'data', 'utils.R'))

read.csv_ <- read.csv.maker(interim_data_fp)
write.csv_ <- write.csv.maker(processed_data_fp)

scale_ <- function(x) as.vector(scale(x))

covariate_cols <- c('lat', 'lon', 'lat2', 'lon2', 'latlon', 
                    'elevation', 'forestcover', 'hwy.length', 'lim.hwy.length',
                    'local.rd.length', 'point.emi', 'tmp', 'wind')

message("Reading interim Sm.csv")
Sm <- read.csv_('Sm.csv')
message("Writing processed Sm.csv")
Sm %>%
  select(lat, lon) %>%
  write.csv_(., file = 'Sm.csv')

message("Reading interim S.csv")
S <- read.csv_('S.csv')
message("Writing processed S.csv")
S %>%
  select(lat, lon) %>%
  write.csv_(., file = 'S.csv')

message("Reading interim Xm.csv")
Xm <- read.csv_('Xm.csv')
message("Reading processed Xm.csv")
Xm %>%
  left_join(S, by = 'gridID') %>%
  transform(intercept = 1, lat2 = lat^2, lon2 = lon^2, latlon = lat * lon) %>%
  select_(.dots = c('date', 'intercept', covariate_cols)) %>%
  mutate_at(covariate_cols, scale_) %>%
  write.csv_(., file = 'Xm.csv')

message("Reading interim X.csv")
X <- read.csv_('X.csv')
message("Writing processed X.csv")
X %>%
  left_join(S, by = 'gridID') %>%
  transform(intercept = 1, lat2 = lat^2, lon2 = lon^2, latlon = lat * lon) %>%
  select_(.dots = c('date', 'intercept', covariate_cols)) %>%
  mutate_at(covariate_cols, scale_) %>%
  write.csv_(., file = 'X.csv')

message("Reading interim zm.csv")
zm <- read.csv_('zm.csv')
message("Reading processed zm.csv")
zm %>%
  select(date, aod) %>%
  mutate_at('aod', scale_) %>%
  write.csv_(., file = 'zm.csv')

message("Reading interim z.csv")
z <- read.csv_('z.csv')
message("Writing processed z.csv")
z %>%
  select(date, aod) %>%
  mutate_at('aod', scale_) %>%
  write.csv_(., file = 'z.csv')

message("Reading interim ym.csv")
ym <- read.csv_('ym.csv')
message("Writing processed ym.csv")
ym %>%
  select(date, pm25) %>%
  write.csv_(., file = 'ym.csv')

message("Reading interim y.csv")
y <- read.csv_('y.csv')
message("Writing processed y.csv")
y %>%
  group_by(date, gridID) %>%
  summarise(pm25 = ifelse(all(is.na(pm25)), NA, mean(pm25, na.rm = TRUE))) %>%
  write.csv_(., file = 'y.csv')

message("Writing processed mapping.csv")
y %>%
  transform(cellID = as.numeric(factor(gridID))) %>%
  filter(date == date[1], !is.na(monID)) %>%
  select(monID, cellID) %>%
  write.csv_(., file = 'mapping.csv')
