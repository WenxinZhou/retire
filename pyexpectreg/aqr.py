import numpy as np
import numpy.random as rgt
from scipy.stats import norm
from scipy.optimize import minimize



class low_dim():
    '''
        Approximate Quantile Regression

    Approximate check function: l(x) = tau * x^{1+delta} + (1-tau) * (-x)^{1+delta}, 
                                where 0<tau<1 is a quantile level, 0<delta<=1 is an
                                approximation parameter. When delta=1, this becomes 
                                the asymmetric L2-loss (for expectile regression).
    '''

    def __init__(self, X, Y, intercept=True):
        '''
        Arguments
        ---------
        X : n by p numpy array of covariates; each row is an observation vector.
           
        Y : n by 1 numpy array of response variables.
            
        intercept : logical flag for adding an intercept to the model.
        '''
        self.Y, self.n = Y.reshape(len(Y)), len(Y)
        if X.shape[1] >= self.n: raise ValueError("covariate dimension exceeds sample size")
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X]
            self.X1 = np.c_[np.ones(self.n,), (X - self.mX)/self.sdX]
        else:
            self.X = np.copy(X)
            self.X1 = self.X/self.sdX

    def ols(self):
        return np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(self.Y))

    def check(self, x, tau=0.5):
        return np.mean(np.where(x > 0, tau, 1 - tau) * abs(x))

    def acheck(self, x, tau=0.5, delta=0.2):
        '''
            Empirical Approximate Check Loss
        '''
        pos = x >= 0
        tmp = abs(x) ** (1+delta)
        return (tau*sum(tmp[pos]) + (1-tau)*sum(tmp[~pos])) / len(x)
    
    def acheck_weight(self, x, tau=0.5, delta=0.2):
        '''
            Derivaties of Approximate Check Loss
        '''
        pos = x >= 0
        tmp = abs(x) ** delta
        tmp[pos] *= tau 
        tmp[~pos] *= tau - 1
        return (1+delta) * tmp

    def check_weight(self, x, tau=0.5):
        '''
            Subgradient of Check Loss
        '''
        return np.where(x>0, tau, tau-1)

    def bfgs(self, tau=0.5, delta=0.2, beta0=np.array([]), tol=None, options=None):
        '''
            Approximate Quantile Regression via BFGS Algorithm
        '''
        y, X = self.Y, self.X

        if len(beta0) == 0: beta0 = np.zeros(X.shape[1])
        
        if 0 < delta <= 1:
            fun = lambda b : self.acheck(y - X.dot(b), tau, delta)
            grad = lambda b : -X.T.dot(self.acheck_weight(y-X.dot(b), tau, delta)/X.shape[0])
        elif delta == 0:
            fun = lambda b : self.check(y - X.dot(b), tau)
            grad = lambda b : -X.T.dot(np.where(y > X.dot(b), tau, tau-1)) / X.shape[0]
        else:
            raise ValueError('delta must be between 0 and 1')

        model = minimize(fun, beta0, method='BFGS', jac=grad, tol=tol, options=options)
        return {'beta': model['x'],
                'res': y - X.dot(model['x']),
                'niter': model['nit'],
                'loss_val': model['fun'],
                'grad_val': model['jac'],
                'message': model['message']}

    def gd_bls(self, tau=0.5, delta=0.2, beta0=np.array([]),
               standardize=True, adjust=True, options=dict()):
        '''
            Approximate Quantile Regression 
            via Gradient Descent and Backtracking Line Search
        '''
        opt = {'lr': 50, 'gamma': 0.95, 'max_nit': 1e3,
               'max_nls': 50, 'tol': 1e-5}
        opt.update(options)

        if standardize: X = self.X1
        else: X = self.X
        Y, itcp = self.Y, self.itcp
        
        if len(beta0) == 0:
            beta0 = np.zeros(X.shape[1])
            if itcp: beta0[0] = np.quantile(Y, tau)
            res = Y - beta0[0]
        else:
            res = Y - X.dot(beta0)
        grad0 = -X.T.dot(self.acheck_weight(res, tau, delta)) / X.shape[0]

        nls, lval, lr_seq  = [], [], []
        lr, dev, t = opt['lr'], 1, 0
        while dev > opt['tol'] and t < opt['max_nit']:      
            loss0 = self.acheck(res, tau, delta)
            beta1 = beta0 - lr * grad0
            res = Y - X.dot(beta1)
            loss1 = self.acheck(res, tau, delta)

            k = 0
            while loss0 <= loss1 and k < opt['max_nls']:
                lr *= opt['gamma']
                beta1 = beta0 - lr * grad0
                res = Y - X.dot(beta1)
                loss1 = self.acheck(res, tau, delta)
                k += 1

            nls.append(k)
            lval.append(loss1)
            lr_seq.append(lr)
            dev = max(abs(beta1 - beta0)) + max(loss0 - loss1, 0)
            beta0 = beta1
            grad0 = -X.T.dot(self.acheck_weight(res, tau, delta)) / X.shape[0]
            t += 1

        if standardize and adjust:
            beta1[itcp:] = beta1[itcp:]/self.sdX
            if itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return {'beta': beta1, 'res': res, 
                'niter': t, 'delta': delta,
                'lval_seq': np.array(lval),
                'lval': lval[-1],
                'grad_val': grad0,
                'lr_seq': np.array(lr_seq),
                'nls_seq': np.array(nls)}