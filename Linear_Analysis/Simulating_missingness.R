#' ---
#' title: "Simulating missingness in data"
#' subtitle: ""
#' author: "James Twose"
#' date: ""
#' ---

library(VIM)
library(ggplot2)
library(reshape2)
library(casnet)

##### generic data setup:
set.seed(42) # this makes the simulation exactly reproducible
ni     = 50  # 100 people
nj     =  10  # 10 week study
id     = rep(1:ni, each=nj)
cond   = rep(c("control", "diet"), each=nj*(ni/2))
base   = round(rep(rnorm(ni, mean=250, sd=10), each=nj))
week   = rep(1:nj, times=ni)
y      = round(base + rnorm(ni*nj, mean=0, sd=1))

# MCAR
prop.m = .07  # 7% missingness
mcar   = runif(ni*nj, min=0, max=1)
y.mcar = ifelse(mcar<prop.m, NA, y)  # unrelated to anything
mcar_df <- data.frame(id, week, cond, base, y, y.mcar, type = "mcar")
mcar_df$type <- as.character(mcar_df$type); str(mcar_df)
mcar_miss <- aggr(mcar_df)
summary(mcar_miss)

# Plot the missingness in the time series
plot.ts(mcar_df$y.mcar)
points(is.na(mcar_df$y.mcar)*mean(mcar_df$y.mcar, na.rm=T), col = "blue", pch = 20)

# MAR
y.mar = matrix(y, ncol=nj, nrow=ni, byrow=TRUE)
for(i in 1:ni){
  for(j in 4:nj){
    dif1 = y.mar[i,j-2]-y.mar[i,j-3]
    dif2 = y.mar[i,j-1]-y.mar[i,j-2]
    if(dif1>0 & dif2>0){  # if weight goes up twice, drops out
      y.mar[i,j:nj] = NA;  break
    }
  }
}
y.mar = as.vector(t(y.mar))
mar_df <- data.frame(id, week, cond, base, y, y.mar, type = "mar")
mar_df$type <- as.character(mar_df$type); str(mar_df)
mar_miss <- aggr(mar_df)
summary(mar_miss)

# Plot the missingness in the time series
plot.ts(mar_df$y.mar)
points(is.na(mar_df$y.mar)*mean(mar_df$y.mar, na.rm=T), col = "blue", pch = 20)

# NMAR
sort.y = sort(y, decreasing=TRUE)
nmar   = sort.y[ceiling(prop.m*length(y))]
y.nmar = ifelse(y>nmar, NA, y)  # doesn't show up when heavier
nmar_df <- data.frame(id, week, cond, base, y, y.nmar, type = "nmar"); str(nmar_df)
nmar_df$type <- as.character(nmar_df$type); str(nmar_df)
nmar_miss <- aggr(nmar_df)
summary(nmar_miss)

# Plot the missingness in the time series
plot.ts(nmar_df$y.nmar)
points(is.na(nmar_df$y.nmar)*mean(nmar_df$y.nmar, na.rm=T), col = "blue", pch = 20)

# Plot all the time series and their missingness
par(mfrow = c(3, 1))
plot.ts(nmar_df$y.nmar)
points(is.na(nmar_df$y.nmar)*mean(nmar_df$y.nmar, na.rm=T), col = "blue", pch = 20)
plot.ts(mar_df$y.mar)
points(is.na(mar_df$y.mar)*mean(mar_df$y.mar, na.rm=T), col = "red", pch = 20)
plot.ts(mcar_df$y.mcar)
points(is.na(mcar_df$y.mcar)*mean(mcar_df$y.mcar, na.rm=T), col = "green", pch = 20)
par(mfrow = c(1, 1))

miss_nmar_rp <- rp(is.na(nmar_df$y.nmar))
crqa_cl(is.na(nmar_df$y.nmar), emRad=NA)
rp_nzdiags(miss_nmar_rp)
rp_plot(miss_nmar_rp, Chromatic = TRUE, plotSurrogate = TRUE)
rn_plot(miss_nmar_rp)

miss_mar_rp <- rp(is.na(mar_df$y.mar))
#crqa_rp(miss_nmar_rp, emRad = 0.05)
rp_plot(miss_mar_rp)

miss_mcar_rp <- rp(is.na(mcar_df$y.mcar))
#crqa_rp(miss_nmar_rp, emRad = 0.05)
rp_plot(miss_mcar_rp)

?casnet

?rp_plot
?rp
