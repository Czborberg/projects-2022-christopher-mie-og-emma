import numpy as np
from types import SimpleNamespace
from scipy import optimize

class RamseyModel():

    def initialmodel(self):
        """ create model 
        
        NB: The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.
        
        """
        
        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()
        
        self.setup()
        self.allocate()

    def setup(self):
        """ parameters of the model """

        par = self.par

        par.alpha = 0.3 # capital share
        par.beta = np.nan # discount factor
        par.A=np.nan
        par.delta = 0.05 # depreciation rate
        par.theta = 2 # coefficient of constant relative risk aversion (CRRA)
        par.K_initial = 1.0
        par.transition_path = 500 
    
    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvariables = ['A','K','C','w','r','Y','K_lag']

        for variablename in allvariables:
            setattr(path, variablename, np.nan*np.ones(par.transition_path))
    
    def steady_state(self, KY_ss, do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss
       
        ss.K = KY_ss
        Y,_,_ = production(par,1.0,ss.K)
        ss.A = 1/Y
        
        # factor prices
        ss.Y, ss.r, ss.w = production(par,ss.A, ss.K)
        assert np.isclose(ss.Y, 1.0)
        
        # implied discount factor
        par.beta = 1/(1+ss.r)

        # consumption (goods market clear Y = C + I, I = delta*K)
        ss.C = ss.Y - par.delta*ss.K 

        if do_print:

            print(f'Y_ss = {ss.Y:.4f}')
            print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
            print(f'r_ss = {ss.r:.4f}')
            print(f'w_ss = {ss.w:.4f}')
            print(f'beta = {par.beta:.4f}')
            print(f'A = {ss.A:.4f}')
        
    def evaluate_path_errors(self):
        """ evaluate errors """

        par = self.par
        ss = self.ss
        path = self.path
  
        C = path.C
        C_plus = np.append(path.C[1:], ss.C)

        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0, par.K_initial)

        path.Y, path.r, path.w = production(par, ss.A, K_lag)
        r_plus = np.append(path.r[1:], ss.r)

        errors = np.nan*np.ones((2, par.transition_path))
        errors[0,:] = C**(-par.theta) - par.beta*(1+r_plus)*C_plus**(-par.theta)
        errors[1,:] = K - ((1-par.delta)*K_lag + path.Y - C)
        
        return errors.ravel()
        
    def calculate_jacobian(self, h = 1e-6):
        """ calculate jacobian """
        
        par = self.par
        ss = self.ss
        path = self.path
        
        # a. allocate
        Njac = 2*par.transition_path
        jac = self.jac = np.nan*np.ones((Njac,Njac))
        
        x_ss = np.nan*np.ones((2,par.transition_path))
        x_ss[0,:] = ss.C
        x_ss[1,:] = ss.K
        x_ss = x_ss.ravel()

        # b. baseline errors
        path.C[:] = ss.C
        path.K[:] = ss.K
        base = self.evaluate_path_errors()

        # c. jacobian
        for i in range(Njac):
            
            # i. add small number to a single x (single K or C) 
            x_jac = x_ss.copy()
            x_jac[i] += h
            x_jac = x_jac.reshape((2, par.transition_path))
            
            # ii. alternative errors
            path.C[:] = x_jac[0,:]
            path.K[:] = x_jac[1,:]
            alt = self.evaluate_path_errors()

            # iii. numerical derivative
            jac[:,i] = (alt-base)/h
        
    def solve(self):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((2, par.transition_path))
            path.C[:] = x[0,:]
            path.K[:] = x[1,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # b. initial guess
        x0 = np.nan*np.ones((2, par.transition_path))
        x0[0,:] = ss.C
        x0[1,:] = ss.K
        x0 = x0.ravel()

        # c. call solver
        x = broyden_solver(eq_sys,x0,self.jac)
            
        # d. final evaluation
        eq_sys(x)

def production(par, A, K_lag):
    """ production and factor prices """

    # a. production
    Y = A*K_lag**par.alpha * 1**(1-par.alpha)

    # b. factor prices
    r = A*par.alpha * K_lag**(par.alpha-1) * 1**(1-par.alpha)
    w = A*(1-par.alpha) * K_lag**(par.alpha) * 1**(-par.alpha)

    return Y,r,w            

def broyden_solver(f, x0, jac, tol=1e-8, maxiter=100, do_print=True):
    """ numerical equation system solver using the broyden method 
    
        f (callable): function return errors in equation system
        jac (ndarray): initial jacobian
        tol (float,optional): tolerance
        maxiter (int,optional): maximum number of iterations
        do_print (bool,optional): print progress

    """

    # a. initial
    x = x0.ravel()
    y = f(x)

    # b. iterate
    for it in range(maxiter):
        
        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print: print(f' it = {it:3d} -> max. abs. error = {abs_diff:12.8f}')

        if abs_diff < tol: return x
        
        # ii. new x
        dx = np.linalg.solve(jac,-y)
        assert not np.any(np.isnan(dx))
        
        # iii. evaluate
        ynew = f(x+dx)
        dy = ynew-y
        jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx)
        y = ynew
        x += dx
            
    else:

        raise ValueError(f'no convergence after {maxiter} iterations')       