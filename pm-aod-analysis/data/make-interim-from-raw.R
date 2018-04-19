library(dplyr)
library(sp)  # for loading Grid.RData

data_fp <- file.path('pm-aod-analysis', 'data')
raw_data_fp <- file.path(data_fp, 'raw')
interim_data_fp <- file.path(data_fp, 'interim')

source(file.path('pm-aod-analysis', 'data', 'utils.R'))

load_ <- function(x) load(file.path(raw_data_fp, x), .GlobalEnv)
read.csv_ <- read.csv.maker(raw_data_fp)
write.csv_ <- write.csv.maker(interim_data_fp)

date_from_dayofyear <- function(doy, year) {
  jan1_year <- paste0(year, '-01-01')
  doy_offset <- doy - 1  # i.e., as.Date(0, origin = date) returns date
  as.Date(doy_offset, origin = jan1_year)  
}

# Monitor locations
message("Reading Location.csv")
mon <- read.csv_('Location.csv')
message("Writing Sm.csv")
mon %>%
  rename(monID = ID, gridID = Grid_Cell, lat = pm_lat, lon = pm_lon) %>%
  select(monID, gridID, lat, lon) %>%
  write.csv_(., file = 'Sm.csv')

# Grid cell locations
message("Reading Grid.RData")
load_('Grid.RData')
grid <- grid@data
message("Writing S.csv")
grid %>%
  rename(gridID = GridID, lat = LATITUDE, lon = LONGITUDE) %>%
  select(gridID, lat, lon) %>%
  write.csv_(., file = 'S.csv')

# Daily spatial covariate values
message("Reading AOD.RData")
load_('AOD.RData')
drop_cols <- c('year', 'doy', 'vgrd', 'ugrd', 'dateorder', 'point.emi.any') 
dat <- dat %>%
  transform(date = date_from_dayofyear(doy, year)) %>% 
  select(-one_of(drop_cols)) %>%
  arrange(date, gridID)

# Daily PM_2.5 measurements
message("Reading PM25.csv")
pm25 <- read.csv_('PM25.csv')
pm25 <- pm25 %>%
  transform(date = date_from_dayofyear(day, year)) %>%
  arrange(date, monID) %>%
  select(date, monID, gridID, pm25)
message("Writing ym.csv")
pm25 %>% 
  write.csv_(., file = 'ym.csv')
message("Writing y.csv")
dat %>%
  select(date, gridID) %>%
  left_join(pm25, by = c('date', 'gridID')) %>%
  write.csv_(., file = 'y.csv')

# Covariate values at monitor locations
pm25_dat <- pm25 %>%
  select(-pm25) %>%
  left_join(dat, by = c('date', 'gridID'))

# Missing covariate
cols <- c('date', 'gridID', 'aod')
message("Writing z.csv")
dat %>%
  select_(.dots = cols) %>%
  write.csv_(., file = 'z.csv')
message("Writing zm.csv")
pm25_dat %>%
  select_(.dots = cols) %>%
  write.csv_(., file = 'zm.csv')

# All other observed covariates
cols <- c('date', 'gridID', 'elevation', 'forestcover', 'hwy.length', 
          'lim.hwy.length', 'local.rd.length', 'point.emi', 'tmp', 'wind')
message("Writing X.csv")
dat %>% 
  select_(.dots = cols) %>%
  write.csv_(., file = 'X.csv')
message("Writing Xm.csv")
pm25_dat %>%
  select_(.dots = cols) %>%
  write.csv_(., file = 'Xm.csv')
