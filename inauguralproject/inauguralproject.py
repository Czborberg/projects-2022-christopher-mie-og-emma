def premium(q,p=0.2):
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


def utility(z,theta=-2):
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

def expected_utility(q,x,p=0.2,y=1.0):
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


