import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from scipy.stats import norm
from scipy.optimize import minimize


class leverage_func():
    def __init__(self,func_type):
        """
        func_type: string which indicates one of the types of leverage function: polynomial, egarch, gjr.
        """
        self.type = func_type
    def func_stat(self,tau1,tau2=None,sigma=None):
        if self.type == 'polynomial':
            mu = 0
            var = tau1**2+2*tau2**2
            return np.array([mu,var])
        elif self.type == 'egarch':
            mu = 0
            var = tau1**2+tau2**2*(1-2/np.pi)
            return np.array([mu,var])
        elif self.type =='gjr':
            mu = (tau1*sigma**2)/2
            var = (tau1**2*sigma**4)/2
            return np.array([mu,var])
    def func_value(self,tau1,xi,tau2=None,sigma=None):
        if self.type == 'polynomial':
            return tau1*xi+tau2*(xi**2-1)
        elif self.type == 'egarch':
            return tau1*xi + tau2*(abs(xi)-np.sqrt(2/np.pi))
        elif self.type =='gjr':
            return tau1*xi**2*sigma**2*np.where(xi<0,xi,0)

class RealGARCH_generate_data():
    def __init__(self,specification='linear',leverage_func = leverage_func(func_type='polynomial'),random_state=None):
        self.random_state = random_state
        self.specification = specification
        self.leverage_func = leverage_func
    def  generate(self,params,n_observation=10**5):
        mu = params['mu']
        omega = params['omega']
        beta = params['beta']
        gamma = params['gamma']
        dz = params['dz']
        phi = params['phi']
        tau1 = params['tau1']
        tau2 = params['tau2']
        sigma_u = params['sigma_u']
        sigma0 = params['sigma0']
        np.random.seed(self.random_state)
        xi = np.random.normal(0,1,n_observation)
        u =  np.random.normal(0,sigma_u,n_observation)
        sigma = np.array([np.nan]*n_observation)
        eps = np.array([np.nan]*n_observation)
        X = np.array([np.nan]*n_observation)
        if sigma0==None:
            sigma0 = np.sqrt((omega+gamma*dz)/(1-(beta+phi*gamma)))
        if self.specification == 'linear':
            sigma[0] = sigma0
            eps[0] = sigma[0] * xi[0]
            X[0] = dz + phi*sigma[0]**2 + self.leverage_func.func_value(tau1=tau1,tau2=tau2,sigma = sigma[0],xi=xi[0]) + u[0]
            for t in range(1,n_observation):
                sigma[t] = np.sqrt(omega + beta*sigma[t-1]**2 + gamma*X[t-1])
                X[t] = dz + phi*sigma[t]**2 + self.leverage_func.func_value(tau1=tau1,tau2=tau2,sigma = sigma[t],xi=xi[t]) + u[t]
                eps[t] = sigma[t] * xi[t]
            return [eps+mu,X,xi,u]
        elif self.specification == 'log':
            sigma[0] = sigma0
            eps[0] = sigma[0] * xi[0]
            X[0] = np.exp(dz + phi*np.log(sigma[0]**2) +\
                          self.leverage_func.func_value(tau1=tau1,tau2=tau2,sigma = sigma[0],xi=xi[0]) + u[0])
            for t in range(1,n_observation):
                sigma[t] = np.sqrt(np.exp(omega + beta*np.log(sigma[t-1]**2) + gamma*np.log(X[t-1])))
                X[t] = np.exp(dz + phi*np.log(sigma[t]**2) +\
                              self.leverage_func.func_value(tau1=tau1,tau2=tau2,sigma = sigma[t],xi=xi[t]) + u[t])
                eps[t] = sigma[t] * xi[t]
            return [eps+mu,X]
    def log_likelihood(self,params,y,X):
        mu = params['mu']
        omega = params['omega']
        beta = params['beta']
        gamma = params['gamma']
        dz = params['dz']
        phi = params['phi']
        tau1 = params['tau1']
        tau2 = params['tau2']
        sigma_u = params['sigma_u']
        sigma0 = params['sigma0']
        eps = y - mu
        sigma = np.array([np.nan]*len(y))
        if sigma0 == None:
            sigma0 = np.sqrt((omega+gamma*dz)/(1-(beta+phi*gamma)))
        else:
            sigma[0] = sigma0
        xi = np.array([np.nan]*len(y))
        u = np.array([np.nan]*len(y))
        xi[0] = eps[0]/sigma[0]
        if self.specification == 'linear':
            u[0] = X[0]-dz - phi*sigma[0]**2 -self.leverage_func.func_value(tau1=tau1,tau2=tau2,xi=xi[0],sigma=sigma[0])
            for t in range(1,len(y)):
                sigma[t] = np.sqrt(omega+beta*sigma[t-1]**2+gamma*X[t-1])
                xi[t] = eps[t]/sigma[t]
                u[t] = X[t]-dz - phi*sigma[t]**2 -self.leverage_func.func_value(tau1=tau1,xi=xi[t],tau2=tau2,sigma=sigma[t])
            L = norm.pdf(eps, 0,sigma) * norm.pdf(u,0,sigma_u)
        elif self.specification == 'log':
            u[0] = np.log(X[0])-dz - phi*np.log(sigma[0]**2) - self.leverage_func.func_value(tau1=tau1,xi=xi[0],tau2=tau2,sigma=sigma[0])
            for t in range(1,len(y)):
                sigma[t] = np.sqrt(np.exp(omega+beta*np.log(sigma[t-1]**2)+gamma*np.log(X[t-1])))
                xi[t] = eps[t]/sigma[t]
                u[t] = np.log(X[t])-dz - phi*np.log(sigma[t]**2) - self.leverage_func.func_value(tau1=tau1,xi=xi[t],tau2=tau2,sigma=sigma[t])
            L = norm.pdf(eps,0,sigma) *norm.pdf(u,0,sigma_u)
        return -sum(np.log(L))