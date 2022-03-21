import numpy as np

def premium(q, p=0.2):
    """ 
    Function that captures the insurance company's premium policy 
    
    Args: 
    q: input array (coverage amount)
    p: scalar (probability of monetary loss which is 0.2 by default)
    
    Returns:
    pi: output array (premium)

    """
    pi = p*q
    return pi

def utility(z, theta=-2):
    """ 
    CRRA utility function
    
    Args: 
    z: input array (what the agent values)
    theta: scalar (constant that captures the agent's risk aversion which is -2 by default)
    
    Returns:
    u: output array (agent's utility)
    
    """
    u = z**(1+theta)/(1+theta)
    return u

def expected_utility(q, x, p=0.2, y=1.0):
    """ 
    Function that calculates an insured agent's expected utility (for q=0 is holds for agents that are not insured)
    
    Args: 
    q: input array (coverage amount)
    x: input array (monetary loss)
    p: scalar (probability of monetary loss which is 0.2 by default)
    y: scalar (agent's assets which is 1.0 by default)
    
    Returns:
    V: output array (expected utility)
    
    """
    V = p*utility(y-x+q-premium(q)) + (1-p)*utility(y-premium(q))
    return V

def expected_utility_pi(q,pi,x=0.6,p=0.2,y=1.0):
    """ 
    Function that calculates an insured agent's expected utility where the premium is a variable
    
    Args: 
    q:  input array (coverage amount)
    pi: input array (premium)
    x:  scalar (monetary loss which is 0.6 by default)
    p:  scalar (probability of monetary loss which is 0.2 by default)
    y:  scalar (agent's assets which is 1.0 by default)
    
    Returns:
    V_pi: output array (expected utility)
    
    """
    V_pi = p*utility(y-x+q-pi) + (1-p)*utility(y-pi)
    return V_pi

def indifferent(q,pi):
    """ 
    Function that calculates difference in expected utility from not having an insurance
    
    Args: 
    q:  input array (coverage amount)
    pi: input array (premium)
    
    Returns:
    dV: output array (difference in expected utility from not having an insurance)
    
    """
    dV = expected_utility_pi(q,pi)-expected_utility_pi(0,0)
    return dV

def expected_utility_MC(gamma, pi, y=1.0):
    """ 
    Function that is a modification of previous function for expected utility. 
    This fuunction calculates the agent's expected utility when the loss, x, is
    drawn from a beta distribution and a fraction, gamma, is covered. 
    
    Args:
    gamma: scalar (coverage fraction of x)
    pi:    scalar (premium)
    y:  scalar (agent's assets which is 1.0 by default)
    
    Returns:
    V_mod: output array (agent's expected utility after modification)
    
    """
    X = np.random.beta(a=2,b=7,size=10**6)
    V_mod = np.mean(utility(y-(1 - gamma)*X - pi))
    return  V_mod

def mc_indifferent(pi):
    X = np.random.beta(a=2,b=7,size=10**6)
    return np.mean(utility(y-(1-0.95)*X-pi)-utility(y-1*X))