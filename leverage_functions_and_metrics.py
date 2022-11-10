import numpy as np
import math

##################################################### normal distribution's pdf ########################################################

def norm(x, mean=0, sigma = 1):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mean)**2/(2*sigma**2))

##################################################### leverage functions ##############################################################

def e_leverage(theta,nu):
    return theta*nu + ( abs(nu)-np.sqrt(2/np.pi) )
def t_leverage(theta_plus,theta_minus, eps):
    return theta_plus*eps*(eps>=0) - theta_minus*eps *(eps<0)
def gjr_leverage(theta_gjr1, theta_gjr2,eps):
    return theta_gjr1*eps**2 +theta_gjr2 * (eps<0)*eps**2
def poly_leverage(theta1, theta2, nu):
    return theta1*nu + theta2*(nu**2-1)
def multy_poly_leverage(theta1, theta2, theta4,nu):
    return theta1*nu + theta2*(nu**2-1)+theta4*(nu**4-3)

####################################################### generator #####################################################################

def gen_t(x, nu):
    n = len(nu)
    eps_tg = np.array([np.nan]*n)
    sigmalog2 = np.array([np.nan]*n)
    
    omega = x[0]
    beta = x[1]
    theta_plus = x[2]
    theta_minus = x[3]

    sigma0 = x[4]

    eps_tg[0] = sigma0*nu[0]
    sigmalog2[0] = np.log(sigma0**2)

    for t in range(1, n):
        sigmalog2[t] = omega + theta_plus*eps_tg[t-1]*(eps_tg[t-1]>=0) - theta_minus*eps_tg[t-1] \
        *(eps_tg[t-1]<0) + beta*sigmalog2[t-1]
        eps_tg[t] = np.sqrt(np.exp(sigmalog2[t]))*nu[t]
    return np.sqrt(np.exp(sigmalog2))

def gen_e(x, nu):
    n = len(nu)
    eps_egarch = np.array([np.nan]*n)
    sigmalog2 = np.array([np.nan]*n)

    w = x[0]
    a = x[1]
    b = x[2]
    theta = x[3]
    sigma0 = x[4]

    eps_egarch[0] = sigma0*nu[0]
    sigmalog2[0] = np.log(sigma0**2)

    for t in range(1, n):
        sigmalog2[t] = w + a*(theta*nu[t-1] + ( abs(nu[t-1])-math.sqrt(2/np.pi) )) + b*sigmalog2[t-1]
        eps_egarch[t] = np.sqrt(np.exp(sigmalog2[t]))*nu[t]
    return np.sqrt(np.exp(sigmalog2))

def gen_gjr(x, nu):
    n = len(nu)
    eps_gjr = np.array([np.nan]*n)
    sigmalog2 = np.array([np.nan]*n)

    w = x[0]
    a = x[1]
    b = x[2]
    theta = x[3]
    sigma0 = x[4]

    eps_gjr[0] = sigma0*nu[0]
    sigmalog2[0] = np.log(sigma0**2)

    for t in range(1, n):
        sigmalog2[t] = w + a*eps_gjr[t-1]**2 +theta * (eps_gjr[t-1]<0)*eps_gjr[t-1]**2 + b*sigmalog2[t-1]
        eps_gjr[t] = np.sqrt(np.exp(sigmalog2[t]))*nu[t]
    return np.sqrt(np.exp(sigmalog2))

def gen_poly(x, nu):
    n = len(nu)
    eps_poly = np.array([np.nan]*n)
    sigmalog2 = np.array([np.nan]*n)

    w = x[0]
    a = x[1]
    b = x[2]
    theta1 = x[3]
    theta2 = x[4]
    sigma0 = x[5]

    eps_poly[0] = sigma0*nu[0]
    sigmalog2[0] = np.log(sigma0**2)

    for t in range(1, n):
        sigmalog2[t] = w + theta1*nu[t-1] + theta2*(nu[t-1]**2-1) + b*sigmalog2[t-1]
        eps_poly[t] = np.sqrt(np.exp(sigmalog2[t]))*nu[t]
    return np.sqrt(np.exp(sigmalog2))

def gen_multy_poly(x, nu):
    n = len(nu)
    eps_poly = np.array([np.nan]*n)
    sigmalog2 = np.array([np.nan]*n)

    w = x[0]
    a = x[1]
    b = x[2]
    theta1 = x[3]
    theta2 = x[4]
    theta4 = x[5]
    sigma0 = x[6]

    eps_poly[0] = sigma0*nu[0]
    sigmalog2[0] = np.log(sigma0**2)

    for t in range(1, n):
        sigmalog2[t] = w + theta1*nu[t-1] + theta2*(nu[t-1]**2-1)+theta4*(nu[t-1]**4-3) \
        + b*sigmalog2[t-1]
        eps_poly[t] = np.sqrt(np.exp(sigmalog2[t]))*nu[t]
    return np.sqrt(np.exp(sigmalog2))


########################################################## Loglikelihood ###################################################

def l_e(x, y):
    n = len(y)
    w = x[0]
    a = x[1]
    b = x[2]
    theta = x[3]
    eps = y 
    sigma0 = x[4]
    sigmalog2 = np.array([np.nan]*n)
    sigmalog2[0] = np.log(sigma0**2)
    nu = np.array([np.nan]*n)
    nu[0] =eps[0]/np.sqrt(np.exp(sigmalog2[0]))
    for t in range(1,n):
        sigmalog2[t] = w + a*(theta*nu[t-1] + ( abs(nu[t-1])-math.sqrt(2/np.pi) )) + b*sigmalog2[t-1]
        nu[t] = eps[t]/np.sqrt(np.exp(sigmalog2[t]))
    L = np.array([np.nan]*n)
    sigma = np.sqrt(np.exp(sigmalog2))
    for t in range(n):
        L[t] = norm(eps[t], 0, sigma[t])
    l = np.sum(np.log(L))
    return -l

def l_t(x, y):
    n = len(y)
    omega = x[0]
    beta = x[1]
    theta_plus = x[2]
    theta_minus = x[3]
    eps = y 
    sigma0 = x[4]
    sigmalog2 = np.array([np.nan]*n)
    sigmalog2[0] = np.log(sigma0**2)
    for t in range(1,n):
        sigmalog2[t] = omega + theta_plus*eps[t-1]*(eps[t-1]>=0) - theta_minus*eps[t-1]*(eps[t-1]<0) + beta*sigmalog2[t-1]
    L = np.array([np.nan]*n)
    sigma = np.sqrt(np.exp(sigmalog2))
    for t in range(n):
        L[t] = norm(eps[t], 0, sigma[t])
    l = np.sum(np.log(L))
    return -l

def l_gjr(x, y):
    n = len(y)
    eps_gjr = y
    sigmalog2 = np.array([np.nan]*n)

    w = x[0]
    a = x[1]
    b = x[2]
    theta = x[3]
    sigma0 = x[4]
    nu = np.array([np.nan]*len(y))
    nu[0] = eps_gjr[0]/sigma0
    sigmalog2[0] = np.log(sigma0**2)
    for t in range(1,n):
        sigmalog2[t] = w + (a +theta * (eps_gjr[t-1]<0))*eps_gjr[t-1]**2 + b*sigmalog2[t-1]
        nu[t] = eps_gjr[t]/np.sqrt(np.exp(sigmalog2[t]))
    L = np.array([np.nan]*n)
    sigma = np.sqrt(np.exp(sigmalog2))
    for t in range(n):
        L[t] = norm(eps_gjr[t], 0, sigma[t])
    l = np.sum(np.log(L))
    return -l

def l_poly(x, y):
    n = len(y)
    eps_poly = y
    sigmalog2 = np.array([np.nan]*n)
    nu = np.array([np.nan]*n)
    w = x[0]
    a = x[1]
    b = x[2]
    theta1 = x[3]
    theta2 = x[4]
    sigma0 = x[5]
    
    nu[0] = eps_poly[0]/sigma0
    sigmalog2[0] = np.log(sigma0**2)
    for t in range(1, n):
        sigmalog2[t] = w + theta1*nu[t-1] + theta2*(nu[t-1]**2-1) + b*sigmalog2[t-1]
        nu[t] = eps_poly[t]/np.sqrt(np.exp(sigmalog2[t]))
    L = np.array([np.nan]*n)
    sigma = np.sqrt(np.exp(sigmalog2))
    for t in range(n):
        L[t] = norm(eps_poly[t], 0, sigma[t])
    l = np.sum(np.log(L))
    return -l

def l_multy_poly(x, y):
    n = len(y)
    eps_poly = y
    sigmalog2 = np.array([np.nan]*n)

    w = x[0]
    a = x[1]
    b = x[2]
    theta1 = x[3]
    theta2 = x[4]
    theta4 = x[5]
    sigma0 = x[6]
    nu[0] = eps_poly[0]/sigma0
    sigmalog2[0] = np.log(sigma0**2)
    for t in range(1, n):
        sigmalog2[t] = w + theta1*nu[t-1] + theta2*(nu[t-1]**2-1)+ theta4*(nu[t-1]**4-3)+ b*sigmalog2[t-1]
        nu[t] = eps_poly[t]/np.sqrt(np.exp(sigmalog2[t]))
    L = np.array([np.nan]*n)
    sigma = np.sqrt(np.exp(sigmalog2))
    for t in range(n):
        L[t] = norm(eps_poly[t], 0, sigma[t])
    l = np.sum(np.log(L))
    return -l
#################################################### leverage linear combination ###################################################
def gen_lin(x, nu):
    n = len(nu)
    eps = np.array([np.nan]*n)
    sigmalog2 = np.array([np.nan]*n)
    
    omega = x[0]
    a_e = x[1]
    a_t = x[2]
    a_gjr = x[3]
    a_poly = x[4]
    beta = x[5]
    theta = x[6]
    theta_plus,theta_minus = x[7],x[8]
    theta_gjr1,theta_gjr2 = x[9], x[10]
    theta1, theta2 = x[11],x[12]
    sigma0 = x[13]
    eps[0] = sigma0*nu[0]
    sigmalog2[0] = np.log(sigma0**2)

    for t in range(1, n):
        sigmalog2[t] = omega + a_e*e_leverage(theta, nu[t-1]) + a_t*t_leverage(theta_plus,theta_minus, eps[t-1])\
        +a_gjr*gjr_leverage(theta_gjr1, theta_gjr2, eps[t-1]) + a_poly*poly_leverage(theta1, theta2, nu[t-1]) \
        + beta*sigmalog2[t-1]
        eps[t] = np.sqrt(np.exp(sigmalog2[t]))*nu[t]
    return np.sqrt(np.exp(sigmalog2))

#################################################### KL divergence #################################################################

# sigma  - std of the process which you want to approximate

# gen - generator of the process which you want to measure KL

# nu - shocks

# x0 - parameters

def KL(x0, sigma, nu, gen, is_aggregate = True, mean = True):
    sigma_teta = gen(x0, nu)
    kl = -1/2 - np.log(sigma)+np.log(sigma_teta)+0.5*(sigma/sigma_teta)**2
    if is_aggregate == True:
        if mean:
            return np.mean(kl)
        else:
            return np.sum(kl)
    else:
        return kl
    
####################################################### Moments ##############################################################
from scipy.integrate import quad

class expectation_2m_:
    def __init__(self,params, leverage ='egarch', prod_iter = 100):
        self.leverage = leverage
        assert len(params) == 4
        self.w, self.teta, self.gamma, self.b = [x for x in params]
        self.prod_iter = prod_iter
    def Prod(self,x, m,i):
        return np.exp(m*self.b**(i-1)*self.g(x)-x**2/2)/(np.sqrt(2*np.pi))
    def Normal_2m_moment(self,x,m):
        return np.abs(x)**(2*m)/np.sqrt(2*np.pi)*np.exp(-x**2/2)
    def g(self,x):
        if self.leverage == 'egarch':
            return self.teta*x + self.gamma * (np.abs(x) - np.sqrt(2/np.pi))
        elif self.leverage == 'poly':
            return self.teta*x + self.gamma * (x**2 - 1)
    def calc(self, m):
        prod = np.array([quad(self.Prod, -np.inf, np.inf, args=(m,i))[0] for i in range(1, self.prod_iter)])
        return quad(self.Normal_2m_moment, -np.inf, np.inf,args=(m))[0] * np.exp(m*self.w/(1-self.b))*np.prod(prod)
   