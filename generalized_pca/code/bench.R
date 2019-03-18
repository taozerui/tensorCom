# generate data
library(MASS)
sampleSize <- 100
mean <- c(0, 0)
var <- matrix(c(1, 1, 1, 5), 2, 2)
# natural parameter
theta <- mvrnorm(n = sampleSize, mu = mean, Sigma = var)
thetaMean <- apply(theta, 2, mean)
theta <- t(apply(theta, 1, '-', thetaMean))
plot(theta[, 1], theta[, 2])
# probability via logit transformation
logit <- function(x) {
  return(1 / (1 + exp(-x)))
}
XProb <- logit(theta)
# observed binary data
bin <- function(x) {
  return(max(x - 0.5, 0)/abs(x - 0.5))
}
XBin <- matrix(lapply(XProb, bin), sampleSize, 2)
XBin <- matrix(do.call(rbind, XBin), sampleSize, 2)
# logistic pca
library(logisticPCA)
model <- logisticPCA(XBin, k=1, main_effects = F)
thetaProject <- XBin %*% model$U %*% t(model$U)
plot(thetaProject[, 1], thetaProject[, 2])