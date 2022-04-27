library("foreign")
library("ggplot2")
library("BaylorEdPsych")
library("miscTools")
library("pROC")
library("margins")
library("boot")
library("lmtest")
library("numDeriv")
library("rJava")
library("xlsx")
library("forecast")
library("rugarch")

func_stat <- function(type, tau1, tau2=NULL, sigma=NULL) {
  if (type == 'polynomial') {
    mu <- 0
    var <- tau1 ^ 2 + 2 * tau2 ^ 2
    return(c(mu, var))
    } else if (type == 'egarch') {
        mu <- 0
        var = tau1 ^ 2 + tau2 ^ 2*(1 - 2 / pi)
        return(c(mu,var))
    } else if (type == 'gjr') {
        mu <- (tau1 * sigma ^ 2) / 2
        var <- (tau1 ^ 2 * sigma ^ 4) / 2
        return(c(mu, var))
    }
}


func_value <- function(type, tau1, xi, tau2=NULL, sigma=NULL) {
  if (type == 'polynomial') {
    return(tau1 * xi + tau2 * (xi ^ 2 - 1)) 
  } else if (type == 'egarch') {
      return(tau1 * xi + tau2 * (abs(xi) - sqrt(2 / pi)))
  } else if (type == 'gjr') {
      return(tau1 * xi ^ 2 * sigma ^ 2 * ifelse(xi < 0, xi, 0))
  }
}

func_value(type='polynomial',tau1=params$tau1,tau2=params$tau2,xi=1)

generate <- function(type, specification, params, n_observation = 10 ^ 5, random_state=2001) {
  mu <- params$mu
  omega <- params$omega
  beta <- params$beta
  gamma <- params$gamma
  dz <- params$dz
  phi <- params$phi
  tau1 <- params$tau1
  tau2 <- params$tau2
  sigma_u <- params$sigma_u
  sigma0 <- params$sigma0
  set.seed(random_state)
  xi <- rnorm(n_observation, mean=0, sd=1)
  u <- rnorm(n_observation, mean=0, sd=sigma_u) 
  
  sigma <- rep(NaN, times = n_observation)
  eps <- rep(NaN, times = n_observation)
  x <- rep(NaN, times = n_observation)
  
  if (is.null(sigma0)) {
    sigma0 <- sqrt((omega + gamma * dz)/(1 - (beta + phi * gamma))) 
  }
  if (specification == 'linear') {
    sigma[1] <- sigma0
  
    eps[1] = sigma[1] * xi[1]
    x[1] <- dz + phi * sigma[1] ^ 2 + func_value(type=type, tau1=tau1, tau2=tau2, sigma=sigma[1], xi=xi[1]) + u[1]
    
    for (t in 2:n_observation) {
      sigma[t] <- sqrt(omega + beta * sigma[t - 1] ^ 2 + gamma * x[t - 1])
      x[t] <- dz + phi*sigma[t]**2 + func_value(type=type, tau1=tau1, tau2=tau2, sigma=sigma[t], xi=xi[t]) + u[t]
      eps[t] <- sigma[t] * xi[t]
    }
    return(list(y=eps+mu,
                x=x))
    
  } else if (specification == 'log') {
      sigma[1] <- sigma0
      eps[1] <- sigma[1] * xi[1]
      x[1] <- exp(dz + phi * log(sigma[1] ^ 2) + func_value(type=type, tau1=tau1, tau2=tau2, sigma=sigma[1], xi=xi[1]) + u[1])
      
      for (t in 2:n_observation) {
        sigma[t] <- sqrt(exp(omega + beta * log(sigma[t - 1] ^ 2) + gamma * log(x[t - 1])))
        x[t] <- exp(dz + phi * log(sigma[t] ^ 2) + func_value(type=type, tau1=tau1, tau2=tau2, sigma=sigma[t], xi=xi[t]) + u[t])
        eps[t] <- sigma[t] * xi[t]
      }
      return(list(y=eps+mu,
                  x=x))
    }
}
params = list(
  mu=0.5, 
  omega = 0.001,
  beta=0.02,
  gamma=0.01,
  dz=0.005,
  phi=-0.15,
  tau1=0.009,
  tau2=-0.0036,
  sigma_u=0.04,
  sigma0= 0.1
  )
gen <- generate(type = 'polynomial',specification = 'log', params = params, n_observation = 10^4)
log_likelihood_gen <- function(type, specification, params, y, x)
{
  mu <- params$mu
  omega <- params$omega
  beta <- params$beta
  gamma <- params$gamma
  dz <- params$dz
  phi <- params$phi
  tau1 <- params$tau1
  tau2 <- params$tau2
  sigma_u <- params$sigma_u
  sigma0 <- params$sigma0
  eps <- y - mu
  sigma <- rep(NaN, times = length(y))
  
  if (is.null(sigma0)) {
    sigma0 <- sqrt((omega + gamma * dz) / (1 - (beta + phi * gamma)))
  } else {
    sigma[1] = sigma0
  }
  
  xi <- rep(NaN, times = length(y))
  u <- rep(NaN, times = length(y))
  
  xi[1] <- eps[1] / sigma[1]
  
  if (specification == 'linear') {
    u[1] <- x[1] - dz - phi * sigma[1] ^ 2 - func_value(type, tau1=tau1, tau2=tau2, xi=xi[1], sigma=sigma[1])
  
    for (t in 2:length(y)) {
      sigma[t] <- sqrt(omega + beta * sigma[t - 1] ^ 2 + gamma * x[t - 1])
      xi[t] <- eps[t] / sigma[t]
      u[t] <- x[t] - dz - phi * sigma[t] ^ 2 - func_value(type=type, tau1=tau1, xi=xi[t], tau2=tau2, sigma=sigma[t])
    }
  
    L <- dnorm(eps, 0, sigma) * dnorm(u, 0, sigma_u)
    
  } else if (specification == 'log') {
      u[1] <- log(x[1]) - dz - phi * log(sigma[1] ^ 2) - func_value(type=type, tau1=tau1, xi=xi[1], tau2=tau2, sigma=sigma[1])
      
      for (t in 2:length(y)) {
        sigma[t] <- sqrt(exp(omega + beta * log(sigma[t - 1] ^ 2) + gamma * log(x[t - 1])))
        xi[t] <- eps[t] / sigma[t]
        u[t] <- log(x[t]) - dz - phi * log(sigma[t] ^ 2) - func_value(type=type, tau1=tau1, xi=xi[t], tau2=tau2, sigma=sigma[t])
      }
      L <- dnorm(eps, 0, sigma) * dnorm(u, 0, sigma_u)
  }
  
  return(sum(log(L)))
}

params_c <- c(0.3, 0.02, 0.05, 0.03, 0.01, -0.001, 0.002, 0.00004, 0.1, 1)

#for (i in 1:length(params))
#{
#  params_c[i]<-as.numeric(params[i])
#}
data <- data.frame('y'=gen$y,'x'=gen$x)
data <- as.matrix(data)
log_likelihood <- function(params, data, type='polynomial', specification='log')
{
  y <- data[,1]
  x<-data[,2]
  mu <- params_c[1]
  omega <- params_c[2]
  beta <- params_c[3]
  gamma <- params_c[4]
  dz <- params_c[5]
  phi <- params_c[6]
  tau1 <- params_c[7]
  tau2 <- params_c[8]
  sigma_u <- params_c[9]
  sigma0 <- params_c[10]
  eps <- y - mu
  sigma <- rep(NaN, times = length(y))
  
  if (is.null(sigma0)) {
    sigma0 <- sqrt((omega + gamma * dz) / (1 - (beta + phi * gamma)))
  } else {
    sigma[1] = sigma0
  }
  
  xi <- rep(NaN, times = length(y))
  u <- rep(NaN, times = length(y))
  
  xi[1] <- eps[1] / sigma[1]
  
  if (specification == 'linear') {
    u[1] <- x[1] - dz - phi * sigma[1] ^ 2 - func_value(type, tau1=tau1, tau2=tau2, xi=xi[1], sigma=sigma[1])
    
    for (t in 2:length(y)) {
      sigma[t] <- sqrt(omega + beta * sigma[t - 1] ^ 2 + gamma * x[t - 1])
      xi[t] <- eps[t] / sigma[t]
      u[t] <- x[t] - dz - phi * sigma[t] ^ 2 - func_value(type=type, tau1=tau1, xi=xi[t], tau2=tau2, sigma=sigma[t])
    }
    
    L <- dnorm(eps, 0, sigma) * dnorm(u, 0, sigma_u)
    
  } else if (specification == 'log') {
    u[1] <- log(x[1]) - dz - phi * log(sigma[1] ^ 2) - func_value(type=type, tau1=tau1, xi=xi[1], tau2=tau2, sigma=sigma[1])
    
    for (t in 2:length(y)) {
      sigma[t] <- sqrt(exp(omega + beta * log(sigma[t - 1] ^ 2) + gamma * log(x[t - 1])))
      xi[t] <- eps[t] / sigma[t]
      u[t] <- log(x[t]) - dz - phi * log(sigma[t] ^ 2) - func_value(type=type, tau1=tau1, xi=xi[t], tau2=tau2, sigma=sigma[t])
    }
    L <- dnorm(eps, 0, sigma) * dnorm(u, 0, sigma_u)
  }
  return(sum(log(L)))
}

log_likelihood(params_c,type='polynomial',specification = 'log',data)

result <- optim(par = params_c,       
                method = "BFGS",                
                fn = log_likelihood,                              
                control = list(maxit = 10000,          
                               fnscale = -1,           
                               reltol = 1e-10),                
                hessian = FALSE,                      
                data = data) 
