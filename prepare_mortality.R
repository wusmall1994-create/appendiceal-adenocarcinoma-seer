prepare_mortality <- function(read_export) {
  death_year <- "Year of death recode"
  death_age <- "Age at death with <1 yr olds and 90+"
  diag_year <- "Year of diagnosis"
  diag_age <- "Age recode with <1 year olds and 90+"
  expected_ages <- c(0L, 1L, seq(5L, 90L, 5L))
  grid <- expand.grid(year = 2000:2023, age = expected_ages)
  key <- function(year, age) paste(year, age, sep = "/")

  read_two <- function(file) {
    x <- read_export(file)
    year <- suppressWarnings(as.integer(x[[death_year]]))
    age <- suppressWarnings(as.integer(substr(x[[death_age]], 1, 2)))
    rows <- which(year %in% 2000:2023 & age %in% expected_ages)
    observed <- key(year[rows], age[rows])
    stopifnot(length(observed) == nrow(grid), !anyDuplicated(observed))
    order <- rows[match(key(grid$year, grid$age), observed)]
    stopifnot(!anyNA(order), all(x$Count[order] >= 0), all(x$Population[order] > 0))
    data.frame(grid, deaths=x$Count[order], population=x$Population[order])
  }

  specific <- read_two("R01_AppendixCOD_DeathYear_Age_SEER17_v01.csv")
  four <- read_export("R01_AppendixCOD_DxYear_DeathYear_Age_SEER17_v01.csv")
  d <- data.frame(year=suppressWarnings(as.integer(four[[death_year]])),
                  age=suppressWarnings(as.integer(substr(four[[death_age]],1,2))),
                  dxyear=suppressWarnings(as.integer(four[[diag_year]])),
                  dxage=suppressWarnings(as.integer(substr(four[[diag_age]],1,2))),
                  count=four$Count)
  annual <- subset(d, year %in% 2000:2023 & dxyear %in% 2000:2023)
  invalid <- subset(annual, !(age %in% expected_ages & dxage %in% expected_ages))
  stopifnot(all(invalid$count==0))
  detail <- subset(annual, age %in% expected_ages & dxage %in% expected_ages)
  stopifnot(!anyDuplicated(detail[c("year","age","dxyear","dxage")]), all(detail$count>=0))
  detail <- subset(detail,count>0)
  detail$period_gap <- detail$year - detail$dxyear
  stopifnot(all(detail$period_gap>=0))
  rows <- match(key(detail$year,detail$age),key(grid$year,grid$age))
  counts <- function(records) {
    out <- numeric(nrow(grid))
    if (!nrow(records)) return(out)
    idx <- match(key(records$year,records$age),key(grid$year,grid$age))
    stopifnot(!anyNA(idx))
    totals <- tapply(records$count,idx,sum)
    out[as.integer(names(totals))] <- totals
    out
  }
  stopifnot(!anyNA(rows), all(counts(detail)==specific$deaths))
  specific$deaths3 <- counts(subset(detail,period_gap<=2))
  specific$deaths5 <- counts(subset(detail,period_gap<=4))
  young <- subset(detail, year>=2004 & dxage>=20 & dxage<50)
  summarize_intervals <- function(x,label) {
    data.frame(diagnosis_period=label, observed_deaths=sum(x$count),
               median_calendar_year_difference=median(rep(x$period_gap,x$count)),
               deaths_0_1=sum(x$count[x$period_gap<=1]),
               deaths_2_3=sum(x$count[x$period_gap>=2 & x$period_gap<=3]),
               deaths_4_or_more=sum(x$count[x$period_gap>=4]),
               death_age_50_or_older=sum(x$count[x$age>=50]))
  }
  latency <- rbind(summarize_intervals(subset(young,dxyear<=2018),"2000-2018"),
                   summarize_intervals(subset(young,dxyear>=2019),"2019-2023"),
                   summarize_intervals(young,"All"))
  check <- data.frame(compared_cells=nrow(grid), appendix_cause=sum(specific$deaths))
  list(specific=specific, latency=latency, check=check)
}
