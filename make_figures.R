library(ggplot2)
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) stop("Usage: Rscript make_figures.R <results directory> <figure output directory>")
p <- normalizePath(args[1],mustWork=TRUE)
figure_dir <- args[2]
dir.create(figure_dir,recursive=TRUE,showWarnings=FALSE)
read <- function(n) read.csv(file.path(p,paste0(n,".csv")))
save_fig <- function(g,n,h) {
 ggsave(file.path(figure_dir,paste0(n,".png")),g,width=6.5,height=h,dpi=450,device=ragg::agg_png,bg="white")
}
base <- theme_minimal(base_size=11,base_family="Arial")+
 theme(plot.margin=margin(7,11,6,8),axis.title.x=element_text(margin=margin(t=6)),
  axis.title.y=element_text(margin=margin(r=6)),axis.text=element_text(colour="#303640"),
  panel.grid.minor=element_blank(),panel.grid.major.x=element_blank(),
  panel.grid.major.y=element_line(colour="#E0E4E9",linewidth=.28),
  legend.position="bottom",legend.margin=margin(0,0,0,0),
  legend.text=element_text(size=9.5),legend.key.width=unit(17,"pt"))
draw_period <- function(d,labels,cols,ylabel,xlabel) {
 lev <- unique(d$series)
 d$series <- factor(d$series,levels=lev,labels=labels)
 d$x <- match(d$period,sort(unique(d$period)))
 d$x <- d$x+(as.integer(d$series)-(length(lev)+1)/2)*.09
 ggplot(d,aes(x,rate,colour=series,shape=series,group=series))+
  geom_line(linewidth=.55)+geom_errorbar(aes(ymin=lo,ymax=hi),width=.045,linewidth=.35)+
  geom_point(size=2)+scale_colour_manual(values=cols)+scale_shape_manual(values=c(16,17,15,18)[seq_along(lev)])+
  scale_x_continuous(breaks=1:4,labels=c("2004-2008","2009-2013","2014-2018","2019-2023"),expand=expansion(add=.30))+
  scale_y_continuous(limits=c(0,NA),expand=expansion(mult=c(0,.10)))+
  labs(x=xlabel,y=ylabel,colour=NULL,shape=NULL)+guides(colour=guide_legend(nrow=1))+base
}
palette <- c("#264F81","#777C85","#A04031","#237A70")
save_fig(draw_period(read("inc_period"),c("Overall","Localized","Regional","Distant"),palette,
                     "Cases per 100,000 person-years","Diagnosis period"),"fig1",3.32)
save_fig(draw_period(read("hist_period"),c("Conventional / intestinal","Mucinous type","Signet-ring cell"),palette[c(1,3,4)],
                     "Cases per 100,000 person-years","Diagnosis period"),"fig2",3.32)
save_fig(draw_period(read("ibm_period"),c("20-49 years","50-64 years","65 years or older"),palette[c(1,3,4)],
                     "Deaths per 100,000 person-years","Death period"),"fig3",3.32)
inc <- subset(read("inc_apc"),group=="20-49"&model=="Full")
mo <- read("mort_apc")
m <- subset(mo,endpoint=="deaths5"&model=="Full")
hist <- subset(read("hist_apc"),group=="20-49"&model=="Full")
forest <- rbind(
 data.frame(label=paste(c("Overall","Distant stage","Regional stage","Localized stage"),"incidence (20-49)"),type="Incidence",inc[c("apc","lo","hi")]),
 data.frame(label=paste("Appendix cancer IBM",paste0("(",m$group,")")),type="Mortality",m[c("apc","lo","hi")]),
 data.frame(label=c("Conventional / intestinal (20-49)","Mucinous type (20-49)","Signet-ring cell (20-49)"),
            type="Incidence",hist[c("apc","lo","hi")])
)
draw_forest <- function(d) {
 d$label <- factor(d$label,levels=rev(d$label))
 ggplot(d,aes(apc,label,colour=type))+
  geom_vline(xintercept=0,linetype="dashed",colour="#737983",linewidth=.36)+
  geom_segment(aes(x=lo,xend=hi,yend=label),linewidth=.55)+
  geom_point(aes(shape=type),size=2.25)+
  scale_colour_manual(values=c(Incidence="#264F81",Mortality="#A04031"))+
  scale_shape_manual(values=c(Incidence=16,Mortality=17))+
  scale_x_continuous(expand=expansion(mult=c(.04,.06)))+
  labs(x="Annual percent change (95% CI)",y=NULL)+base+
  theme(legend.position="none",panel.grid.major.y=element_blank(),panel.grid.major.x=element_line(colour="#E0E4E9",linewidth=.28))
}
save_fig(draw_forest(forest),"fig4",4)
ys <- subset(mo,group=="20-49"&model=="Full")
ys$label <- c("0-4 calendar-year difference","Unrestricted","0-2 calendar-year difference")[match(ys$endpoint,c("deaths5","deaths","deaths3"))]
ys$type <- "Mortality"
ys <- ys[match(c("deaths","deaths5","deaths3"),ys$endpoint),]
save_fig(draw_forest(ys),"figS1",2.0)
c <- read("cohort_rates")
g <- ggplot(c,aes(decade,rate))+
 geom_errorbar(aes(ymin=lo,ymax=hi),width=2,linewidth=.38,colour="#264F81")+
 geom_point(colour="#264F81",size=2.1)+
 scale_x_continuous(breaks=c$decade)+scale_y_continuous(limits=c(0,NA),expand=expansion(mult=c(0,.08)))+
 labs(x="Approximate birth decade",y="Cases per 100,000 person-years")+base
save_fig(g,"figS2",2.95)

flow_data <- read("analysis_flow")
age_flow <- read("incidence_flow_age")
cohort <- read("cohort_flow")
fmt <- function(x) format(x,big.mark=",",trim=TRUE,scientific=FALSE)
by_age <- setNames(age_flow$count,age_flow$group)
flow <- data.frame(y=c(3,2,1),head=c("SEER 17 incidence","SEER 17 appendix cancer IBM","SEER 8 cohort sensitivity"),
 detail=c(
  paste0("2000-2023 export: ",fmt(flow_data$incidence_export)," cases\n2004-2023: ",
         fmt(flow_data$incidence_adults)," cases diagnosed at ages 20+\nAges 20-49: ",fmt(by_age["20-49"]),
         " | 50-64: ",fmt(by_age["50-64"])," | 65+: ",fmt(by_age["65+"])),
  paste0("2000-2023 export: ",fmt(flow_data$ibm_export)," deaths after deduplication\n2004-2023: ",
         fmt(flow_data$ibm_adults)," deaths at ages 20+\nFixed diagnostic window: ",fmt(flow_data$ibm_fixed_adults)," deaths"),
  paste0("1975-2023: ",fmt(cohort$young_cases)," cases diagnosed at ages 20-49\nAnnual diagnosis-year by diagnosis-age cells")))
g <- ggplot(flow)+
 geom_rect(aes(xmin=.05,xmax=.95,ymin=y-.385,ymax=y+.385),fill="#F3F6FA",colour="#BBC7D4",linewidth=.32)+
 geom_text(aes(x=.085,y=y+.20,label=head),hjust=0,family="Arial",fontface="bold",size=3.6,colour="#243D58")+
 geom_text(aes(x=.085,y=y-.07,label=detail),hjust=0,family="Arial",size=3.05,lineheight=1.19,colour="#293240")+
 coord_cartesian(xlim=c(0,1),ylim=c(.57,3.43),expand=FALSE)+theme_void()+
 theme(plot.margin=margin(3,0,3,0))
save_fig(g,"figS3",3.62)
