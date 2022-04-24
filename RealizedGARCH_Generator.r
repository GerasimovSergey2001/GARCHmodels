
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


generate <- function(type, specification, params, n_observation = 10 ^ 5) {
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
    }
    
    x[t] <- dz + phi*sigma[t]**2 + func_value(type=type, tau1=tau1, tau2=tau2, sigma=sigma[t], xi=xi[t]) + u[t]
    eps[t] <- sigma[t] * xi[t]
    
    return(c(eps + mu, x, xi, u))
    
  } else if (specification == 'log') {
      sigma[1] <- sigma0
      eps[1] <- sigma[1] * xi[1]
      x[1] <- exp(dz + phi * log(sigma[1] ^ 2) + func_value(type=type, tau1=tau1, tau2=tau2, sigma=sigma[1], xi=xi[1]) + u[1])
      
      for (t in 2:n_observation) {
        sigma[t] <- sqrt(exp(omega + beta * log(sigma[t - 1] ^ 2) + gamma * log(x[t - 1])))
      }
      
      x[t] <- exp(dz + phi * log(sigma[t] ^ 2) + func_value(type=type, tau1=tau1, tau2=tau2, sigma=sigma[t], xi=xi[t]) + u[t])
      eps[t] <- sigma[t] * xi[t]
      
      return(c(eps + mu, x))
    }
}

log_likelihood <- function(type, specification, params, y, x)
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
    }
  
    xi[t] <- eps[t] / sigma[t]
    u[t] <- x[t] - dz - phi * sigma[t] ^ 2 - func_value(type=type, tau1=tau1, xi=xi[t], tau2=tau2, sigma=sigma[t])
  
    L <- dnorm(eps, 0, sigma) * dnorm(u, 0, sigma_u)
    
  } else if (specification == 'log') {
      u[1] <- log(x[1]) - dz - phi * log(sigma[1] ^ 2) - func_value(type=type, tau1=tau1, xi=xi[1], tau2=tau2, sigma=sigma[1])
      
      for (t in 2:length(y)) {
        sigma[t] <- sqrt(exp(omega + beta * log(sigma[t - 1] ^ 2) + gamma * log(x[t - 1])))
      }
      
      xi[t] <- eps[t] / sigma[t]
      u[t] <- log(x[t]) - dz - phi * log(sigma[t] ^ 2) - func_value(type=type, tau1=tau1, xi=xi[t], tau2=tau2, sigma=sigma[t])
     
      L <- dnorm(eps, 0, sigma) * dnorm(u, 0, sigma_u)
  }
  
  return(-sum(log(L)))
}
