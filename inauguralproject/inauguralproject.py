def premium(q,p=0.2):
    """ 
    premium function, pi
    
    Args: 
    q: input array
    p: scalar (0.2 by default)
    
    Returns 
    pi: output array
    
    """
    pi = p*q
    return pi


def utility(z,theta=-2):
    """ 
    CRRA utility function, u(z)
    
    Args: 
    z: input array
    theta: scalar (-2 by default )
    
    Returns 
    u: output array
    
    """
    u = z**(1+theta)/(1+theta)
    return u

