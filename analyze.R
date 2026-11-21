# Reproduce the paper analyses from authorized SEER*Stat exports.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) stop("Usage: Rscript analyze.R <SEER export directory> <results directory>")
src <- normalizePath(args[1],mustWork=TRUE)
out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
script <- sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])
source(file.path(dirname(normalizePath(script,mustWork=TRUE)),"prepare_mortality.R"))
read <- function(name) {
  path <- file.path(src,name)
  if (!file.exists(path)) stop("Required SEER*Stat export is missing: ",path)
  read.csv(path,check.names=FALSE,na.strings=c("~",""))
}
write <- function(x, name) write.csv(x,file.path(out,paste0(name,".csv")),row.names=FALSE)
yearnum <- function(x) suppressWarnings(as.integer(x))
agenum <- function(x) suppressWarnings(as.integer(substr(x,1,2)))
grp <- function(a) cut(a,c(-1,19,49,64,99),c("Under 20","20-49","50-64","65+"))
stage0 <- read("adenocarcinoma_stage_age_incidence_seer17_2000_2023.csv")
b <- data.frame(year=yearnum(stage0[[1]]),age=agenum(stage0[[2]]),raw=stage0[[3]],count=stage0$Count,pop=stage0$Pop)
b <- subset(b,!is.na(year)&!is.na(age))
stopifnot(nrow(unique(b[c("year","age","raw")]))==nrow(b))
b$group <- grp(b$age)
b$endpoint <- ifelse(grepl("^Localized",b$raw),"Localized",ifelse(grepl("^Regional",b$raw),"Regional",ifelse(grepl("^Distant",b$raw),"Distant",ifelse(grepl("^Unknown|^Blanks",b$raw),"Unknown","Other"))))
pop <- unique(b[c("year","age","group","pop")])
stopifnot(!anyDuplicated(pop[c("year","age")]))
overall <- aggregate(count~year+age+group,b,sum)
overall$endpoint <- "Overall"
detail <- aggregate(count~year+age+group+endpoint,b,sum)
fine <- merge(rbind(overall,detail),pop,by=c("year","age","group"))
inc <- aggregate(cbind(count,pop)~year+group+endpoint,fine,sum)
adult <- subset(inc,year>=2004 & group!="Under 20")
hist0 <- read("adenocarcinoma_histology_year_age20_49_seer17_2000_2023.csv")
h <- data.frame(year=yearnum(hist0[[1]]),code=substr(hist0[[2]],1,6),count=hist0$Count,pop=hist0$Population)
h <- subset(h,year>=2004 & grepl("/3$",code))
h$endpoint <- ifelse(grepl("^(847[012]|848[01])/3",h$code),"Mucinous",ifelse(h$code=="8490/3","Signet ring","Conventional"))
hpop <- unique(h[c("year","pop")])
stopifnot(nrow(hpop)==20)
ha <- merge(aggregate(count~year+endpoint,h,sum),hpop)
stopifnot(all(aggregate(count~year,ha,sum)$count==subset(adult,endpoint=="Overall"&group=="20-49")$count))
mortality <- prepare_mortality(read)
write(mortality$latency,"ibm_latency")
write(mortality$check,"ibm_exports_audit")
cod <- mortality$specific
cod$group <- grp(cod$age)
death <- subset(transform(cod,group=factor(group,levels=levels(b$group))),year>=2004 & age>=20)
w <- setNames(c(18257225,17722067,19511370,22179956,22479229,19805793,17224359,13307234,10654272,9409940,8725574,7414559,4900234,2678567,1580606),seq(20,90,5))
z <- qnorm(.975)
periodstats <- function(d,yrs) {
 x <- subset(d,year %in% yrs)
 n <- sum(x$count)
 p <- sum(x$pop)
 c(n=n,pop=p,rate=1e5*n/p,lo=if(n==0)0 else 1e5*qchisq(.025,2*n)/2/p,hi=1e5*qchisq(.975,2*(n+1))/2/p)
}
cmp <- function(x,p1=2004:2008,p2=2019:2023) {
 a <- periodstats(x,p1)
 b <- periodstats(x,p2)
 rr <- unname(b["rate"]/a["rate"])
 lowerupper <- rr*exp(c(-1,1)*z*sqrt(1/a["n"]+1/b["n"]))
 c(setNames(a,paste0("a_",names(a))),setNames(b,paste0("b_",names(b))),rr=rr,rrlo=lowerupper[1],rrhi=lowerupper[2])
}
asr <- function(d,yrs) {
 x <- aggregate(cbind(count,pop)~age,subset(d,year %in% yrs),sum)
 ww <- w[as.character(x$age)]
 ww <- 1e5*ww/sum(ww)/x$pop
 r <- sum(ww*x$count)
 v <- sum(ww^2*x$count)
 m <- max(ww)
 c(rate=r,lo=qgamma(.025,r^2/v,scale=v/r),hi=qgamma(.975,(r+m)^2/(v+m^2),scale=(v+m^2)/(r+m)))
}
estimate <- function(m,L) {
 phi <- max(1,sum(resid(m,type="pearson")^2)/df.residual(m))
 b <- sum(L*coef(m))
 se <- sqrt(drop(L%*%(phi*vcov(m))%*%L))
 v <- 100*expm1(b+c(0,-1,1)*z*se)
 c(apc=v[1],lo=v[2],hi=v[3],p=2*pnorm(-abs(b/se)))
}
fitone <- function(x) {
 x$t <- x$year-2004
 m <- glm(count~t+offset(log(pop)),data=x,family=poisson())
 estimate(m,c(0,1))
}
fitgroups <- function(x) {
 x$group <- factor(as.character(x$group),levels=c("20-49","50-64","65+"))
 x$t <- x$year-2004
 m <- glm(count~t*group+offset(log(pop)),data=x,family=poisson())
 do.call(rbind,lapply(seq_len(3),function(i) {
  L <- c(0,1,0,0,0,0)
  if(i>1) L[3+i] <- 1
  comp <- if(i==1) NA_real_ else estimate(m,replace(numeric(6),3+i,1))["p"]
  c(estimate(m,L),interaction=unname(comp))
 }))
}
incfits <- list()
for(e in c("Overall","Distant","Regional","Localized")) {
 x <- subset(adult,endpoint==e)
 for(model in c("Full","No2020")) {
  d <- if(model=="Full") x else subset(x,year!=2020)
  incfits[[paste(e,model)]] <- data.frame(endpoint=e,model=model,group=c("20-49","50-64","65+"),fitgroups(d))
 }
}
write(do.call(rbind,incfits),"inc_apc")
young <- subset(adult,group=="20-49")
la <- do.call(rbind,lapply(c("Overall","Distant","Regional","Localized"),function(e) {
 s <- subset(young,endpoint==e)
 t <- sapply(s$year,function(y) unname(fitone(subset(s,year!=y))["apc"]))
 data.frame(endpoint=e,min=min(t),max=max(t))
}))
write(la,"loo")
yfine <- subset(fine,year>=2004 & group=="20-49" & endpoint=="Overall")
m <- glm(count~I(year-2004)+factor(age)+offset(log(pop)),data=yfine,family=poisson())
young_adj_apc <- estimate(m,c(0,1,0,0,0,0,0))
md <- subset(yfine,year<=2008 | year>=2019)
mod <- glm(count~I(year>=2019)+factor(age)+offset(log(pop)),data=md,family=poisson())
l <- coef(mod)[2]
adjusted_rr <- exp(l+c(0,-1,1)*z*sqrt(vcov(mod)[2,2]*max(1,sum(resid(mod,type="pearson")^2)/df.residual(mod))))
write(data.frame(as.list(c(young_adj_apc,setNames(adjusted_rr,c("rr","rrlo","rrhi"))))),"young_overall_age_adjusted")
y0423 <- subset(b,year>=2004)
adult_flow <- aggregate(count~group,subset(y0423,group!="Under 20"),sum)
write(adult_flow,"incidence_flow_age")
write(data.frame(incidence_export=sum(b$count),incidence_adults=sum(adult_flow$count),
                 ibm_export=sum(cod$deaths),ibm_adults=sum(death$deaths),ibm_fixed_adults=sum(death$deaths5)),"analysis_flow")
periods <- list(2004:2008,2009:2013,2014:2018,2019:2023)
periodlabels <- c("2004-2008","2009-2013","2014-2018","2019-2023")
allper <- function(d,items,key){
 do.call(rbind,lapply(items,function(e) do.call(rbind,lapply(seq_along(periods),function(i){
  val <- as.data.frame(as.list(periodstats(d[d[[key]]==e,],periods[[i]])))
  data.frame(series=e,period=periodlabels[i],val)
 }))))
}
write(allper(young,c("Overall","Localized","Regional","Distant"),"endpoint"),"inc_period")
write(allper(ha,c("Conventional","Mucinous","Signet ring"),"endpoint"),"hist_period")
prerule <- do.call(rbind,lapply(c("20-49","50-64","65+"),function(g) data.frame(group=g,as.list(cmp(subset(adult,group==g & endpoint=="Distant"),2004:2008,2017:2021)))))
write(prerule,"prerule")
hd <- merge(aggregate(count~year,subset(h,code=="8480/3"),sum),hpop)
h1 <- subset(young,endpoint=="Overall")
h1$count <- h1$count-hd$count[match(h1$year,hd$year)]
write(as.data.frame(as.list(cmp(h1))),"overall_excluding_8480_3")
xb <- subset(death,year>=2004)
mort <- list()
mperiod <- list()
for(col in c("deaths5","deaths","deaths3")) {
 x <- transform(xb,count=xb[[col]],pop=population)
 y <- aggregate(cbind(count,pop)~year+group,x,sum)
 for(k in c("Full","No2020")){
  rows <- if(k=="Full")y else subset(y,year!=2020)
  mort[[paste(col,k)]] <- data.frame(endpoint=col,model=k,group=c("20-49","50-64","65+"),fitgroups(rows))
 }
 mperiod[[col]] <- do.call(rbind,lapply(c("20-49","50-64","65+"),function(g) {
  q <- subset(x,group==g)
  u <- asr(q,2004:2008)
  v <- asr(q,2019:2023)
  data.frame(endpoint=col,group=g,as.list(c(cmp(q),setNames(u,paste0("std_a_",names(u))),setNames(v,paste0("std_b_",names(v))))))
 }))
 if(col=="deaths5")write(allper(y,c("20-49","50-64","65+"),"group"),"ibm_period")
}
write(do.call(rbind,mort),"mort_apc")
write(do.call(rbind,mperiod),"mort_period")
coh <- read("adenocarcinoma_age_year_incidence_seer8_1975_2023.csv")
c <- data.frame(year=yearnum(coh[[1]]),age=agenum(coh[[2]]),count=coh$Count,pop=coh$Population)
c <- subset(c,year>=1975 & year<=2023 & age>=20 & age<50 & pop>0)
stopifnot(!anyDuplicated(c[c("year","age")]) && sum(c$count)==951)
c$cohort <- c$year-c$age-2
c$decade <- floor(c$cohort/10)*10
c$ageband <- ifelse(c$age<35,"20-34","35-49")
cr <- function(x,early,late,name) {
 q <- x[x$cohort%in%early,]
 r <- x[x$cohort%in%late,]
 a <- sum(q$count); b <- sum(r$count)
 ra <- a/sum(q$pop)*1e5; rb <- b/sum(r$pop)*1e5; rr <- rb/ra
 v <- rr*exp(c(-1,1)*1.96*sqrt(1/a+1/b))
 data.frame(group=name,a=a,b=b,ra=ra,rb=rb,rr=rr,lo=v[1],hi=v[2])
}
ct <- rbind(cr(c,1930:1959,1970:2023,"20-49"),cr(subset(c,ageband=="35-49"),1950:1969,1970:1989,"35-49"),cr(subset(c,ageband=="20-34"),1940:1969,1980:1999,"20-34"))
write(ct,"cohort_comparison")
m <- glm(count~I((decade-1950)/10)+ageband+offset(log(pop)),data=subset(c,cohort>=1930),family=poisson())
beta <- coef(m)[2]
s <- sqrt(vcov(m)[2,2]*max(1,sum(resid(m,type="pearson")^2)/df.residual(m)))
write(data.frame(as.list(setNames(exp(unname(beta)+c(0,-1.96,1.96)*s),c("rr","lo","hi")))),"cohort_trend")
write(data.frame(young_cases=sum(c$count)),"cohort_flow")
cd <- aggregate(cbind(count,pop)~decade,c,sum)
cd <- subset(cd,decade>=1930 & count>1)
cd$rate <- 1e5*cd$count/cd$pop
cd$lo <- 1e5*qchisq(.025,2*cd$count)/2/cd$pop
cd$hi <- 1e5*qchisq(.975,2*(cd$count+1))/2/cd$pop
write(cd,"cohort_rates")


# Main incidence tables, including the unknown-stage row and standardization.
inc_specs <- rbind(data.frame(group="20-49",endpoint=c("Overall","Distant","Regional","Localized","Unknown")),
                   data.frame(group=c("50-64","65+"),endpoint="Distant"))
inc_compare <- do.call(rbind,lapply(seq_len(nrow(inc_specs)),function(i) {
  g <- inc_specs$group[i]; e <- inc_specs$endpoint[i]
  x <- subset(adult,group==g & endpoint==e)
  data.frame(group=g,endpoint=e,as.list(cmp(x)))
}))
write(inc_compare,"incidence_comparison")

std_specs <- rbind(data.frame(group=c("20-49","50-64","65+"),endpoint="Overall"),
                   data.frame(group=c("20-49","50-64","65+"),endpoint="Distant"),
                   data.frame(group="20-49",endpoint=c("Regional","Localized")))
inc_std <- do.call(rbind,lapply(seq_len(nrow(std_specs)),function(i) {
  g <- std_specs$group[i]; e <- std_specs$endpoint[i]
  x <- subset(fine,group==g & endpoint==e & year>=2004)
  early <- asr(x,2004:2008)
  late <- asr(x,2019:2023)
  data.frame(group=g,endpoint=e,as.list(c(setNames(early,paste0("a_",names(early))),
                                       setNames(late,paste0("b_",names(late))))))
}))
write(inc_std,"incidence_standardized")

wilson <- function(n,N) {
  p <- n/N
  (p+z^2/(2*N)+c(-1,1)*z*sqrt(p*(1-p)/N+z^2/(4*N^2)))/(1+z^2/N)
}
total <- subset(inc_compare,group=="20-49" & endpoint=="Overall")
distant <- subset(inc_compare,group=="20-49" & endpoint=="Distant")
p1 <- distant$a_n/total$a_n
p2 <- distant$b_n/total$b_n
a_ci <- wilson(distant$a_n,total$a_n)
b_ci <- wilson(distant$b_n,total$b_n)
diff <- p2-p1
write(data.frame(early_pct=p1*100,late_pct=p2*100,diff_pp=diff*100,
                 lo_pp=100*(diff-sqrt((p2-b_ci[1])^2+(a_ci[2]-p1)^2)),
                 hi_pp=100*(diff+sqrt((b_ci[2]-p2)^2+(p1-a_ci[1])^2))),
      "distant_stage_proportion")

# Histology APCs and behavior-code comparisons for Tables 3, S3, and S6.
aggregate_histology <- function(raw) {
  x <- data.frame(year=yearnum(raw[[1]]),code=substr(raw[[2]],1,6),
                  count=raw$Count,pop=raw$Population)
  x <- subset(x,year>=2004 & grepl("/3$",code))
  x$endpoint <- ifelse(grepl("^(847[012]|848[01])/3",x$code),"Mucinous",
                       ifelse(x$code=="8490/3","Signet ring","Conventional"))
  merge(aggregate(count~year+endpoint,x,sum),unique(x[c("year","pop")]))
}
all_hist <- aggregate_histology(read("adenocarcinoma_histology_year_seer17_2000_2023.csv"))
hist_sets <- list("20-49"=ha,"All ages"=all_hist)
hist_apc <- do.call(rbind,lapply(names(hist_sets),function(group) {
  x <- hist_sets[[group]]
  do.call(rbind,lapply(c("Conventional","Mucinous","Signet ring"),function(e) {
    y <- subset(x,endpoint==e)
    rbind(data.frame(group=group,endpoint=e,model="Full",as.list(fitone(y))),
          data.frame(group=group,endpoint=e,model="No2020",as.list(fitone(subset(y,year!=2020)))))
  }))
}))
write(hist_apc,"hist_apc")
write(do.call(rbind,lapply(c("Conventional","Mucinous","Signet ring"),function(e) {
  data.frame(endpoint=e,as.list(cmp(subset(ha,endpoint==e))))
})),"hist_comparison")
behavior <- read("F03_Appendix_8480_Behavior_Year_SEER17_Nov2025_v01.csv")
bh <- data.frame(year=yearnum(behavior[[1]]),code=substr(behavior[[2]],1,6),count=behavior$Count)
bh <- subset(bh,year>=2004 & code %in% c("8480/2","8480/3"))
bh$period <- ifelse(bh$year<=2021,"2004-2021",as.character(bh$year))
write(aggregate(count~period+code,bh,sum),"mucinous_behavior")
write(data.frame(recent_young_diagnoses=sum(young$count[young$year>=2019 & young$endpoint=="Overall"]),
                 diagnoses_final_two_years=sum(young$count[young$year>=2022 & young$endpoint=="Overall"])),
      "recent_diagnosis_counts")

# Sex-stratified sensitivity analysis. This reproduces the original
# sex-specific quasi-Poisson model with the estimated Pearson scale.
sex <- read("adenocarcinoma_sex_age_year_incidence_seer17_2000_2023.csv")
sx <- data.frame(year=yearnum(sex[[1]]),age=agenum(sex[[2]]),sex=sex$Sex,
                 count=sex$Count,pop=sex$Population)
sx <- subset(sx,year>=2004 & age>=20 & sex %in% c("Male","Female"))
sx$group <- grp(sx$age)
sx$sex <- factor(sx$sex, levels=c("Male","Female"))
sx <- aggregate(cbind(count,pop)~year+group+sex,sx,sum)
sex_period <- list()
sex_trend <- list()
for(g in c("20-49","50-64","65+")) {
  d <- subset(sx,group==g)
  d$t <- d$year-2004
  m <- glm(count~t*sex+offset(log(pop)),data=d,family=poisson())
  V <- vcov(m)*sum(resid(m,type="pearson")^2)/df.residual(m)
  contrast <- function(L) {
    b <- sum(L*coef(m))
    s <- sqrt(drop(L%*%V%*%L))
    r <- 100*expm1(b+c(0,-1,1)*1.96*s)
    c(apc=r[1],lo=r[2],hi=r[3],p=2*pnorm(-abs(b/s)))
  }
  interaction <- unname(contrast(c(0,0,0,1))["p"])
  for(s in levels(d$sex)) {
    key <- paste(g,s)
    sex_period[[key]] <- data.frame(group=g,sex=s,as.list(cmp(subset(d,sex==s))))
    sex_trend[[key]] <- data.frame(group=g,sex=s,as.list(contrast(c(0,1,0,as.integer(s=="Female")))),
                                   interaction=interaction)
  }
}
write(do.call(rbind,sex_period),"sex_period")
write(do.call(rbind,sex_trend),"sex_apc")
