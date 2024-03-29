import numpy as np
import numpy.random as rgt
from scipy.stats import norm
import warnings



class low_dim():
    '''
        Robust Expectile Regression
        via Gradient Descent with Barzilai-Borwein Step Size
    '''
    weights = ["Exponential", "Multinomial", "Rademacher", 
               "Gaussian", "Uniform", "Folded-normal"]
    opt = {'max_iter': 1e3, 'max_lr': 50, 'tol': 1e-5, 'nboot': 200}

    def __init__(self, X, Y, intercept=True, options=dict()):
        '''
        Arguments
        ---------
        X : n by p numpy array of covariates; each row is an observation vector.
           
        Y : n by 1 numpy array of response variables.
            
        intercept : logical flag for adding an intercept to the model.

        options : a dictionary of internal statistical and optimization parameters.

            max_iter : maximum numder of iterations in the GD-BB algorithm; default is 500.

            max_lr : maximum step size/learning rate.
            
            tol : the iteration will stop when max{|g_j|: j = 1, ..., p} <= tol 
                  where g_j is the j-th component of the (smoothed) gradient; default is 1e-5.

            nboot : number of bootstrap samples for inference; default is 200.
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

        self.opt.update(options)


    def mad(self, x):
        '''
            Median Absolute Deviation (MAD)
        '''
        return np.median(abs(x - np.median(x))) / norm.ppf(0.75)

    def ols(self):
        return np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(self.Y))

    def _asym(self, x, tau):
        '''
            Compute Asymmetric Residuals
        '''
        return 2 * np.where(x < 0, (1-tau) * x, tau * x)

    def _boot_weight(self, weight):
        boot = {'Multinomial': lambda n : rgt.multinomial(n, pvals=np.ones(n)/n), 
                'Exponential': lambda n : rgt.exponential(size=n), 
                'Rademacher': lambda n : 2*rgt.binomial(1, 1/2, n),
                'Gaussian': lambda n : rgt.normal(1, 1, n), 
                'Uniform': lambda n : rgt.uniform(0, 2, n), 
                'Folded-normal': lambda n : abs(rgt.normal(size=n)) * np.sqrt(np.pi / 2)}

        return boot[weight](self.n)

    def _retire_weight(self, x, tau, c=False, w=np.array([])):
        if not c:
            tmp = np.where(x > 0, tau * x, (1 - tau) * x)
        elif c>0:
            pos = x > 0
            tmp = np.minimum(abs(x), c)
            tmp[pos] *= tau
            tmp[~pos] *= tau - 1
        else:
            raise ValueError('robust parameter must be positive')

        if not w.any():
            return - tmp 
        else:
            return - tmp * w

    def fit(self, tau=0.5, robust=False, scale_invariant=True,
            beta0=np.array([]), res=np.array([]), weight=np.array([]),
            standardize=True, adjust=True):
        '''
            Robust/Huberized Expectile Regression

        Arguments
        ---------
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes expectile regression estimator;
                 if robust > 0, the function computes Huberized expectile regression estimator.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          "robust * MAD of residuals" at each iteration.
                        
        beta0 : initial estimate; default is np.array([]).
        
        res : initial vector of residuals; default is np.array([]).
        
        weight : ndarray of observation weights; default is np.array([]).
        
        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale.

        Returns
        -------
        'beta' : an ndarray of estimated regression coefficients.

        'res' : an ndarray of fitted residuals.

        'robust_para' : robustification parameter.

        'niter' : number of iterations.
        '''

        if standardize: X = self.X1
        else: X = self.X

        if len(beta0) == 0:
            beta0 = np.zeros(X.shape[1])
            if self.itcp: beta0[0] = np.quantile(self.Y, tau)
            res = self.Y - beta0[0]
        elif len(beta0) == X.shape[1]:
            res = self.Y - X.dot(beta0)
        else:
            raise ValueError("dimension of beta0 must match parameter dimension")

        c = robust
        if robust > 0 and scale_invariant:
            c0 = robust * self.mad(self._asym(self.Y, tau))
            c = max( robust * self.mad(self._asym(res, tau)) , 
                     0.1 * c0 )

        grad0 = X.T.dot(self._retire_weight(res, tau, c, weight)) / self.n
        diff_beta = -grad0
        beta = beta0 + diff_beta
        res, t = self.Y - X.dot(beta), 0
        lr_seq = []

        while max(abs(diff_beta)) > self.opt['tol'] and t < self.opt['max_iter']:
            if robust > 0 and scale_invariant:
                c = max( robust * self.mad(self._asym(res, tau)) , 
                         0.1 * c0 )
            grad1 = X.T.dot(self._retire_weight(res, tau, c, weight)) / self.n
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad)
            if r1 == 0: lr = 1
            else:
                r01 = diff_grad.dot(diff_beta)
                lr = min(abs(r01/r1), abs(r0/r01))
            
            if self.opt['max_lr']: lr = min(lr, self.opt['max_lr'])
            grad0 = grad1 
            diff_beta = - lr * grad1
            beta += diff_beta
            res = self.Y - X.dot(beta)
            lr_seq.append(lr)
            t += 1

        if standardize and adjust:
            beta[self.itcp:] /= self.sdX
            if self.itcp: beta[0] -= self.mX.dot(beta[1:])

        return {'beta': beta, 'res': res, 'robust_para': c, 
                'niter': t, 'lr_seq': np.array(lr_seq)}


    def norm_ci(self, tau=0.5, robust=False, scale_invariant=True,
                alpha=0.05, standardize=True):
        '''
            Normal-based Confidence Intervals

        Parameters
        ----------
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes expectile regression estimator;
                 if robust > 0, the function computes Huberized expectile regression estimator.

        scale_invariant : logical flag for making the estimate scale invariant. If True, the scale 
                          parameter will be estimated by the MAD of residuals at each iteration.
                        
        alpha : miscoverage level for each CI; default is 0.05.

        Returns
        -------
        'beta' : an ndarray of estimated regression coefficients.
        
        'normal_ci' : numpy array. Normal CIs based on estimated asymptotic covariance matrix.

        'robust_para' : robust parameter used in the estimate.

        'acov' : estimated asymptotic covariance matrix.
        '''
        X = self.X
        model = self.fit(tau, robust, scale_invariant, standardize=standardize)
        robust = model['robust_para']

        if not robust:
            retire_grad = lambda x : np.where(x > 0, tau * x, (1 - tau) * x)
            retire_hess = lambda x : np.where(x > 0, tau, 1 - tau)
        elif robust > 0:
            retire_grad = lambda x : np.where(x>0, tau, tau-1) * np.minimum(abs(x), robust)
            retire_hess = lambda x : tau * (x >= 0) * (x <= robust) \
                                     + (1 - tau) * (x < 0) * (x >= -robust)

        grad_weight = retire_grad(model['res']) ** 2
        hess_weight = retire_hess(model['res'])

        C1 = (X.T * grad_weight).dot(X) / self.n
        C2 = np.linalg.inv((X.T * hess_weight).dot(X) / self.n)
        ACov = C2.dot(C1).dot(C2)

        rad = norm.ppf(1 - 0.5 * alpha) * np.sqrt(np.diag(ACov) / self.n)        
        ci = np.c_[model['beta'] - rad, model['beta'] + rad]

        return {'beta': model['beta'],
                'normal_ci': ci,
                'robust_para': robust,
                'acov': ACov}


    def mb(self, tau=0.5, robust=False, scale_invariant=True,
           weight="Multinomial", standardize=True):
        '''
            Multiplier Bootstrap Estimates
   
        Parameters
        ----------
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes expectile regression estimator;
                 if robust > 0, the function computes Huberized expectile regression estimator.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          the MAD of residuals at each iteration.

        weight : a character string representing one of the built-in bootstrap weight distributions; 
                 default is "Multinomial".

        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.

        Returns
        -------
        mb_beta : numpy array. 
                  1st column: regression estimate; 
                  2nd to last: bootstrap estimates.
        '''
        if weight not in self.weights:
            raise ValueError("weight distribution must be either Exponential, Rademacher, \
                              Multinomial, Gaussian, Uniform or Folded-normal")
           
        model = self.fit(tau, robust, scale_invariant, standardize=standardize, adjust=False)
        robust = model['robust_para']

        mb_beta = np.zeros([len(model['beta']), self.opt['nboot'] + 1])
        mb_beta[:,0], res = model['beta'], model['res']

        for b in range(self.opt['nboot']):
            model = self.fit(tau, robust, scale_invariant=False,
                             beta0=mb_beta[:,0], res=res,
                             weight=self._boot_weight(weight), standardize=standardize)
            mb_beta[:,b + 1] = model['beta']

        if standardize:
            mb_beta[self.itcp:,0] = mb_beta[self.itcp:,0] / self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX.dot(mb_beta[1:,0])

        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[: , ~np.isnan(mb_beta).any(axis=0)]
        return mb_beta

    
    def mb_ci(self, tau=0.5, robust=False, scale_invariant=True,
              weight="Multinomial", standardize=True, alpha=0.05):
        '''
            Multiplier Bootstrap Confidence Intervals

        Arguments
        ---------
        tau : location parameter between 0 and 1; default is 0.5.
        
        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes expectile regression estimator;
                 if robust > 0, the function computes Huberized expectile regression estimator.

        scale_invariant : logical flag for making the estimate scale invariant. If True, the scale 
                          parameter will be estimated by the MAD of residuals at each iteration.

        weight : a character string representing the random weight distribution;
                 default is "Multinomial".

        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.

        alpha : miscoverage level for each CI; default is 0.05.

        Returns
        -------
        'boot_beta' : numpy array. 
                      1st column: regression estimate; 
                      2nd to last: bootstrap estimates.
        
        'percentile_ci' : numpy array. Percentile bootstrap CI.

        'pivotal_ci' : numpy array. Pivotal bootstrap CI.

        'normal_ci' : numpy array. Normal-based CI using bootstrap variance estimates.
        '''        
        mb_beta = self.mb(tau, robust, scale_invariant, weight, standardize)
        if weight in self.weights[:4]:
            adj = 1
        elif weight == 'Uniform':
            adj = np.sqrt(1/3)
        elif weight == 'Folded-normal':
            adj = np.sqrt(0.5 * np.pi - 1)

        percentile_ci = np.c_[np.quantile(mb_beta[:,1:], 0.5 * alpha, axis=1), \
                              np.quantile(mb_beta[:,1:], 1 - 0.5 * alpha, axis=1)]
        pivotal_ci = np.c_[(1 + 1/adj) * mb_beta[:,0] - percentile_ci[:,1] / adj, \
                           (1 + 1/adj) * mb_beta[:,0] - percentile_ci[:,0] / adj]

        radi = norm.ppf(1- 0.5 * alpha) * np.std(mb_beta[:,1:], axis=1) / adj
        normal_ci = np.c_[mb_beta[:,0] - radi, mb_beta[:,0] + radi]

        return {'boot_beta': mb_beta, 
                'percentile_ci': percentile_ci,
                'pivotal_ci': pivotal_ci,
                'normal_ci': normal_ci}


    def _find_root(self, f, tmin, tmax, tol=1e-5):
        while tmax - tmin > tol:
            tau = (tmin + tmax) / 2
            if f(tau) > 0:
                tmin = tau
            else: 
                tmax = tau
        return tau


    def adaptive_fit(self, dev_prob=None, max_iter=50):
        '''
            Adaptive Huber Regression
        '''
        if dev_prob == None: dev_prob = 1 / self.n

        beta_hat = self.ols()
        rel, err = (len(self.mX) + np.log(1 / dev_prob)) / self.n, 1
        
        count = 0
        while err > self.opt['tol'] and count < max_iter:
            res = self.Y - self.X.dot(beta_hat)
            f = lambda t : np.mean(np.minimum((res / t) ** 2, 1)) - rel
            robust = self._find_root(f, np.min(abs(res)), np.sum(res ** 2))
            model = self.fit(tau=0.5, robust=robust, scale_invariant=False)
            err = np.sum((model['beta'] - beta_hat) ** 2) / np.sum(beta_hat ** 2)
            beta_hat = model['beta']
            count += 1

        return {'beta': beta_hat, 
                'res': res, \
                'robust_para': robust}



class high_dim(low_dim):
    '''
        Penalized Robust Expectile Regression 
        via Iterative Local Adaptive Majorization-Minimization

    References
    ----------
    One-step Sparse Estimates in Nonconcave Penalized Likelihood Models (2008)
    by Hui Zou and Runze Li
    The Annals of Statistics 36(4): 1509–1533.

    I-LAMM for Sparse Learning: Simultaneous Control of 
    Algorithmic Complexity and Statistical Error (2018)
    by Jianqing Fan, Han Liu, Qiang Sun and Tong Zhang
    The Annals of Statistics 46(2): 814-841.

    Iteratively Reweighted l1-Penalized Robust Regression (2021)
    by Xiaoou Pan, Qiang Sun and Wenxin Zhou
    Electronic Journal of Statistics 15(1): 3287-3348.
    '''
    
    opt = {'phi': 0.5, 'gamma': 1.25, 'max_iter': 2e3, \
           'tol': 1e-8, 'iter_warning': False, \
           'irw_tol': 1e-4, 'nboot': 200}
    penalties = {'L1', 'SCAD', 'MCP', 'CapppedL1'}


    def __init__(self, X, Y, intercept=True, options={}):

        '''
        Arguments
        ---------
        X : n by p numpy array of covariates; each row is an observation vector.
           
        Y : an ndarray of response variables.
            
        intercept : logical flag for adding an intercept to the model.

        options : a dictionary of internal statistical and optimization parameters.
        
            phi : initial quadratic coefficient parameter in the ILAMM algorithm; default is 0.5.
        
            gamma : adaptive search parameter that is larger than 1; default is 1.25.
        
            max_iter : maximum numder of iterations in the ILAMM algorithm; default is 2e3.
        
            tol : the ILAMM iteration stops when |beta^{k+1} - beta^k|_max <= tol; default is 1e-8.
            
            iter_warning : logical flag for warning when the maximum number 
                           of iterations is achieved for the l1-penalized fit.

            irw_tol : tolerance parameter for stopping iteratively reweighted L1-penalizations; 
                      default is 1e-4.

            nboot : number of bootstrap samples for post-selection inference; default is 200.
        '''
        self.n, self.p = X.shape
        self.Y = Y.reshape(len(Y))
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X]
            self.X1 = np.c_[np.ones(self.n), (X - self.mX)/self.sdX]
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt.update(options)


    def _soft_thresh(self, x, c):
        '''
            Soft-thresholding Operator
        '''
        return np.sign(x) * np.maximum(abs(x) - c, 0)
        

    def lambda_seq(self, nlambda=100, eps=1e-3, standardize=True):
        '''
        Arguments
        ---------
        nlambda : number of lambda values in the sequence; default is 100.

        eps : minimum lambda is set to be eps * maximum lambda; default is 1e-3.
        '''
        if standardize: X = self.X1
        else: X = self.X
        lambda_max = np.max(np.abs(X.T.dot(self.Y))) / self.n
        return np.exp(np.linspace(np.log(eps*lambda_max), np.log(lambda_max), num=nlambda))


    def _grad_weight(self, x, tau=0.5, c=False):
        if not c:
            return -2 * np.where(x>=0, tau*x, (1-tau)*x) / len(x)
        elif c > 0:
            pos = x > 0
            tmp = np.minimum(abs(x), c)
            tmp[pos] *= tau
            tmp[~pos] *= tau - 1
            return -2 * tmp / len(x)
        else: 
            raise ValueError("robustification parameter should be strictly positive")


    def retire_loss(self, x, tau=0.5, c=False):
        '''
            Asymmetric L2 or Huber Loss 
        '''
        if not c:
            return np.mean( abs(tau - (x<0) ) * x**2 )
        elif c > 0:
            y = abs(x)
            out = (y<=c) * y**2 + (2*c*y-c**2)*(y>c)
            return np.mean(np.where(x<0, 1-tau, tau) * out)
        else:
            raise ValueError("robustification parameter should be strictly positive")
    

    def _concave_weight(self, x, penalty="SCAD", a=None):
        if penalty == "SCAD":
            if a==None: a = 3.7
            tmp = 1 - (abs(x)-1)/(a-1)
            tmp = np.where(tmp<=0, 0, tmp)
            return np.where(tmp>1, 1, tmp)
        elif penalty == "MCP":
            if a==None: a = 3
            tmp = 1 - abs(x)/a 
            return np.where(tmp<=0, 0, tmp)
        elif penalty == "CapppedL1":
            if a==None: a = 3
            return abs(x) <= a/2


    def l1(self, tau=0.5, Lambda=np.array([]), 
           robust=False, scale_invariant=True,
           beta0=np.array([]), res=np.array([]),
           standardize=True, adjust=True):
        '''
            L1-Penalized Robust/Huberized Expectile Regression

        Arguments
        ---------
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.

        Lambda : regularization parameter. This should be either a scalar, or 
                 a vector of length equal to the column dimension of X. If unspecified, 
                 it will be computed by self.lambda_seq().
        
        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes l1-penalized ER estimate;
                 if robust > 0, the function computes l1-penalized robust ER estimate.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          the standard deviation of residuals at each iteration.

        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.

        Returns
        -------
        'beta' : an ndarray of estimated coefficients.
        
        'res' : an ndarray of fitted residuals.

        'intercept' : logical flag for fitting an intercept to the model.

        'count' : number of iterations. 

        'lambda' : lambda value.

        'robust' : output robustification parameter.
        '''
        if not np.array(Lambda).any(): 
            Lambda = max(tau, 1-tau)*np.median(self.lambda_seq(standardize=standardize))

        if standardize: X = self.X1
        else: X = self.X
        
        if len(beta0)==0:
            beta0 = np.zeros(X.shape[1])
            if self.itcp: beta0[0] = np.quantile(self.Y, tau)
            res = self.Y - beta0[0]

        c = robust
        if robust: c0 = robust * self.mad(self._asym(self.Y, tau))
        phi, dev, count = self.opt['phi'], 100, 0
        while dev > self.opt['tol'] and count < self.opt['max_iter']:
            
            if robust > 0 and scale_invariant:
                c = max( robust * self.mad(self._asym(res, tau)), \
                         0.1 * c0 )
            
            grad0 = X.T.dot(self._grad_weight(res, tau, c))
            loss_eval0 = self.retire_loss(res, tau, c)
            beta1 = beta0 - grad0/phi
            beta1[self.itcp:] = self._soft_thresh(beta1[self.itcp:], Lambda/phi)
            diff_beta = beta1 - beta0
            dev = diff_beta.dot(diff_beta)
            
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*dev
            loss_eval1 = self.retire_loss(res, tau, c)
            
            while loss_proxy < loss_eval1:
                phi *= self.opt['gamma']
                beta1 = beta0 - grad0/phi
                beta1[self.itcp:] = self._soft_thresh(beta1[self.itcp:], Lambda/phi)
                diff_beta = beta1 - beta0
                dev = diff_beta.dot(diff_beta)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*dev
                loss_eval1 = self.retire_loss(res, tau, c)
            
            beta0, phi = beta1, (self.opt['phi'] + phi)/2
            count += 1

        if count == self.opt['max_iter'] and self.opt['iter_warning']: 
            warnings.warn("Maximum number of iterations achieved when applying l1() with Lambda={} and tau={}".format(Lambda, tau))

        if standardize and adjust:
            beta1[self.itcp:] /= self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return {'beta': beta1, 'res': res, 
                'intercept': self.itcp, 
                'niter': count, 
                'lambda': Lambda, 
                'robust': c}


    def irw(self, tau=0.5, Lambda=np.array([]),
            robust=False, scale_invariant=True, 
            beta0=np.array([]), res=np.array([]), 
            penalty="SCAD", a=3.7, nstep=3,
            standardize=True, adjust=True):
        '''
            Iteratively Reweighted L1-Penalized Robust Expectile Regression
            
        Arguments
        ---------
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.

        Lambda : regularization parameter. If unspecified, 
                 it will be computed by self.lambda_seq().
        
        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes irw-l1-penalized ER estimate;
                 if robust > 0, the function computes irw-l1-penalized robust ER estimate.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          the standard deviation of residuals at each iteration.
    
        penalty : a character string representing one of the built-in concave penalties; 
                  default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : number of iterations/steps of the IRW algorithm; default is 3.

        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.        
        
        Returns
        -------
        'beta' : an ndarray of estimated coefficients.
        
        'res' : an ndarray of fitted residuals.

        'intercept' : logical flag for fitting an intercept to the model.

        'lambda' : lambda value.

        'robust' : output robustification parameter.
        '''
        
        if not np.array(Lambda).any(): 
            Lambda = 2*max(tau, 1-tau)*np.median(self.lambda_seq())

        if len(beta0)==0:
            model = self.l1(tau, Lambda, robust, scale_invariant, \
                            standardize=standardize, adjust=False)
        else:
            model = self.l1(tau, Lambda, robust, scale_invariant, beta0, res, \
                            standardize, adjust=False)
        beta, res = model['beta'], model['res']
        
        lam = Lambda * np.ones(self.X.shape[1] - self.itcp)
        pos = lam > 0
        rw_lam = np.zeros(self.X.shape[1] - self.itcp)

        max_dev, step = 1, 1
        while max_dev > self.opt['irw_tol'] and step <= nstep:
            rw_lam[pos] = lam[pos] * \
                          self._concave_weight(beta[self.itcp:][pos]/lam[pos], penalty, a)
            model = self.l1(tau, rw_lam, robust, scale_invariant, \
                            model['beta'], model['res'], standardize, adjust=False)
            max_dev = max(abs(model['beta'] - beta))
            beta, res = model['beta'], model['res']
            step += 1
        
        if standardize and adjust:
            beta[self.itcp:] /= self.sdX
            if self.itcp: beta[0] -= self.mX.dot(beta[1:])
    
        return {'beta': beta, 'res': res, 
                'intercept': self.itcp,
                'lambda': Lambda,
                'robust': model['robust'],
                'nstep': step}


    def l1_path(self, tau=0.5, 
                lambda_seq=np.array([]), nlambda=50, order='descend',
                robust=False, scale_invariant=True, standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        tau : location parameter between 0 and 1; default is 0.5.

        lambda_seq : a numpy array of lambda values.

        nlambda : number of lambda values (int).

        order : a character string indicating the order of lambda values 
                along which the solution path is obtained; default is 'descend'.

        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes l1-penalized ER estimates;
                 if robust > 0, the function computes l1-penalized robust ER estimates.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          the standard deviation of residuals at each iteration.
        
        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.     

        Returns
        -------
        'beta_seq' : a sequence of l1-retire estimates. 
                     Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of fitted residual vectors.

        'size_seq' : a sequence of selected model sizes. 

        'lambda_seq' : a sequence of lambda values.
        '''
        if type(lambda_seq) == float or type(lambda_seq) == int:
            raise ValueError("lambda_seq should be an ndarray")
        if len(lambda_seq) == 0:
            lambda_seq = self.lambda_seq(nlambda, standardize=standardize)
        
        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq  = np.sort(lambda_seq)[::-1]
        nlambda = len(lambda_seq)
        beta_seq = np.zeros(shape=(self.X.shape[1], nlambda))
        res_seq = np.zeros(shape=(self.n, nlambda))
        robust_seq, nit_seq = [], []

        model = self.l1(tau, lambda_seq[0], robust, scale_invariant, \
                        standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        robust_seq.append(model['robust'])
        nit_seq.append(model['niter'])
        
        for l in range(1, nlambda):
            model = self.l1(tau, lambda_seq[l], robust, scale_invariant, \
                            beta_seq[:,l-1], res_seq[:,l-1], \
                            standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
            robust_seq.append(model['robust'])
            nit_seq.append(model['niter'])

        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta_seq': beta_seq, \
                'res_seq': res_seq, \
                'size_seq': np.sum(beta_seq!=0, axis=0), \
                'lambda_seq': lambda_seq, \
                'robust_seq': np.array(robust_seq),
                'nit_seq': np.array(nit_seq)}


    def irw_path(self, tau=0.5, 
                 lambda_seq=np.array([]), nlambda=50, order='descend',
                 robust=False, scale_invariant=True,
                 penalty="SCAD", a=3.7, nstep=3, standardize=True, adjust=True):
        '''
            Solution Path of IRW-L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.

        lambda_seq : a numpy array of lambda values.
        
        nlambda : number of lambda values (int).
        
        order : a character string indicating the order of lambda values 
                along which the solution path is obtained; default is 'descend'.

        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes penalized ER estimates;
                 if robust > 0, the function computes penalized robust ER estimates.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          the standard deviation of residuals at each iteration.
                
        penalty : a character string representing one of the built-in concave penalties; 
                  default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : number of iterations/steps of the IRW algorithm; default is 3.
        
        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.

        Returns
        -------
        'beta_seq' : a sequence of irw-l1-retire estimates. 
                     Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of fitted residual vectors.

        'size_seq' : a sequence of selected model sizes. 

        'lambda_seq' : a sequence of lambda values.
        '''
        if type(lambda_seq) == float or type(lambda_seq) == int:
            raise ValueError("lambda_seq should be a numpy array; otherwise, try irw()")
        if not lambda_seq.any():
            lambda_seq = self.lambda_seq(nlambda, standardize=standardize)

        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq  = np.sort(lambda_seq)[::-1]
        nlambda = len(lambda_seq)
        beta_seq = np.empty(shape=(self.X.shape[1], nlambda))
        res_seq = np.empty(shape=(self.n, nlambda))
        robust_seq = []

        model = self.irw(tau, lambda_seq[0], robust, scale_invariant, \
                         penalty=penalty, a=a, nstep=nstep, \
                         standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        robust_seq.append(model['robust'])

        for l in range(1, nlambda):
            model = self.irw(tau, lambda_seq[l], robust, scale_invariant, \
                             beta_seq[:,l-1], res_seq[:,l-1], \
                             penalty, a, nstep, standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
            robust_seq.append(model['robust'])
        
        size_seq = np.sum(beta_seq!=0, axis=0)
        if standardize and adjust:
            beta_seq[self.itcp:,] /= self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
    
        return {'beta_seq': beta_seq, \
                'res_seq': res_seq, \
                'size_seq': size_seq, \
                'lambda_seq': lambda_seq, \
                'robust_seq': np.array(robust_seq)}


    def bic(self, tau=0.5, \
            lambda_seq=np.array([]), nlambda=50, order='descend', \
            robust=False, scale_invariant=True, \
            max_size=False, C=None, penalty="SCAD", a=3.7, nstep=3, \
            standardize=True, adjust=True):
        '''
            Model Selection via Bayesian Information Criterion
        
        Arguments
        ---------
        see l1_path() and irw_path() 

        max_size : an upper bound on the selected model size; 
                   default is FALSE (no size restriction).
        
        C : a positive constant in the BIC-like criterion; default is loglog(p). 
            Larger values of C lead to sparser models.

        Returns
        -------
        'bic_beta' : an ndarray of estimated coefficients for the BIC-selected model.

        'bic_seq' : an ndarray of fitted residuals for the BIC-selected model.

        'bic_size' : selected model size.

        'bic_lambda' : lambda value that corresponds to the BIC-selected model.

        'bw' : bandwidth.
        '''    

        if not lambda_seq.any():
            lambda_seq = self.lambda_seq(nlambda=nlambda, standardize=standardize)
        else:
            nlambda = len(lambda_seq)

        if C==None: C = np.ceil(np.log(np.log(self.p)))

        if penalty not in self.penalties: 
            raise ValueError("penalty must be either L1, SCAD or MCP")
        elif penalty == "L1":
            model_all = self.l1_path(tau, lambda_seq, nlambda, order, \
                                     robust, scale_invariant, \
                                     standardize, adjust)
        else:
            model_all = self.irw_path(tau, lambda_seq, nlambda, order, \
                                      robust, scale_invariant, \
                                      penalty, a, nstep, standardize, adjust)

        robust_seq = model_all['robust_seq']
        BIC = np.array([self.retire_loss(model_all['res_seq'][:,l], tau, robust_seq[l]) \
                        for l in range(nlambda)])
        BIC = np.log(BIC) + C * model_all['size_seq'] * np.log(self.n) / self.n
        if not max_size:
            bic_select = np.argmin(BIC)
        else:
            bic_select = np.where(BIC==min(BIC[model_all['size_seq'] <= max_size]))[0][0]

        return {'bic_beta': model_all['beta_seq'][:,bic_select], \
                'bic_res':  model_all['res_seq'][:,bic_select], \
                'bic_size': model_all['size_seq'][bic_select], \
                'bic_lambda': model_all['lambda_seq'][bic_select],\
                'beta_seq': model_all['beta_seq'], \
                'res_seq': model_all['res_seq'], \
                'lambda_seq': model_all['lambda_seq'], \
                'robust_seq': robust_seq, \
                'bic': BIC}


    def sparse_proj(self, x, s):
        return np.where(abs(x) < np.sort(abs(x))[-s], 0, x)


    def sparse_supp(self, x, s):
        y = abs(x)
        return y >= np.sort(y)[-s]


    def l0(self, tau=0.5, robust=False, scale_invariant=True,
           sparsity=5, exp_size=5, beta0=np.array([]),
           standardize=True, adjust=True,
           tol=1e-5, max_iter=1e3):
        '''
            L0-Penalized (Robust) Expectile Regression 
            via Two-Step Iterative Hard-Thresholding

        Reference
        ---------
        On iterative hard thresholding methods for high-dimensional M-estimation (2014)
        by Prateek Jain, Ambuj Tewari and Purushottam Kar
        Advances in Neural Information Processing Systems 27

        Arguments
        ---------
        tau : location parameter between 0 and 1 (float); default is 0.5.

        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes penalized ER estimates;
                 if robust > 0, the function computes penalized robust ER estimates.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          the standard deviation of residuals at each iteration.

        sparsity : sparsity level (int, >=1); default is 5.

        exp_size : expansion size (int, >=1); default is 5.

        beta0 : initial estimate.

        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.

        adjust : logical flag for returning coefficients on the original scale. 

        tol : tolerance level in the IHT convergence criterion; default is 1e-5.

        max_iter : maximum number of iterations; default is 1e3.

        Returns
        -------
        'beta' : an ndarray of estimated coefficients.

        'select' : indices of non-zero estimated coefficients (intercept excluded).

        'robust_para' : robustification parameter.

        'niter' : number of IHT iterations.
        '''
        X, Y = self.X, self.Y
        itcp, c = self.itcp, robust
        if len(beta0) == 0: beta0 = np.zeros(X.shape[1])
        
        t, dev = 0, 1
        while t < max_iter and dev > tol:
            res = Y - X.dot(beta0)
            if robust > 0 and scale_invariant:
                c = robust * self.mad(self._asym(res, tau))
            grad0 = X.T.dot(self._grad_weight(res, tau, c))
            supp0 = self.sparse_supp(grad0[itcp:], exp_size) + (beta0[itcp:] != 0)
            beta1 = np.zeros(X.shape[1])
            out0 = low_dim(X[:,itcp:][:,supp0], Y, intercept=itcp)\
                   .fit(tau=tau, robust=c, scale_invariant=False, \
                        standardize=standardize, adjust=adjust)
            beta1[itcp:][supp0] = out0['beta'][itcp:]
            if itcp: beta1[0] = out0['beta'][0]
            beta1[itcp:] = self.sparse_proj(beta1[itcp:], sparsity)
            supp1 = beta1[itcp:] != 0
            out1 = low_dim(X[:,itcp:][:,supp1], Y, intercept=itcp)\
                   .fit(tau=tau, robust=c, scale_invariant=False, \
                        standardize=standardize, adjust=adjust)
            beta1[itcp:][supp1] = out1['beta'][itcp:]
            if itcp: beta1[0] = out1['beta'][0]
            dev = max(abs(beta1 - beta0))
            beta0 = np.copy(beta1)
            t += 1

        return {'beta': beta0, 
                'select': np.where(beta0[itcp:] != 0)[0],
                'robust_para': c,
                'niter': t}


    def l0_path(self, tau=0.5, robust=False, scale_invariant=True,
                sparsity_seq=np.array([]), order='ascend',
                sparsity_max=20, exp_size=5, 
                standardize=True, adjust=True,
                tol=1e-5, max_iter=1e3):
        '''
            Solution Path of L0-Penalized (Robust) Expectile Regression

        Arguments
        ---------
        tau : location parameter between 0 and 1 (float); default is 0.5.

        robust : robustification/tuning parameter in the Huber loss.
                 If robust = False, the function computes penalized ER estimates;
                 if robust > 0, the function computes penalized robust ER estimates.

        scale_invariant : logical flag for making the estimate scale invariant. 
                          If True, the scale parameter will be estimated by 
                          the standard deviation of residuals at each iteration.

        sparsity_seq : a predetermined sequence of sparsity levels.
        
        order : a character string indicating the order of sparsity levels along 
                which the solution path is computed; default is 'ascend'.

        sparsity_max : maximum sparsity level (int); default is 20.

        exp_size : expansion size (int, >=1); default is 5.

        standardize : logical flag for x variable standardization prior to fitting the model; 
                      default is TRUE.

        adjust : logical flag for returning coefficients on the original scale. 

        tol : tolerance level in the IHT convergence criterion; default is 1e-5.

        max_iter : maximum number of iterations; default is 1e3.
        '''
        if len(sparsity_seq) == 0:
            sparsity_seq = np.array(range(1, sparsity_max+1))

        if order=='ascend':
            sparsity_seq = np.sort(sparsity_seq)
        elif order=='descend':
            sparsity_seq = np.sort(sparsity_seq)[::-1]
        nsparsity = len(sparsity_seq)

        beta_seq = np.zeros((self.X.shape[1], nsparsity+1))
        nit_seq, robust_seq = [], []
        for k in range(nsparsity):
            model = self.l0(tau, robust, scale_invariant, \
                            sparsity_seq[k], exp_size, beta_seq[:,k-1], \
                            standardize, adjust, tol, max_iter)
            beta_seq[:,k] = model['beta']
            nit_seq.append(model['niter'])
            robust_seq.append(model['robust_para'])

        return {'beta_seq': beta_seq[:,:nsparsity],  
                'size_seq': np.sum(beta_seq[self.itcp:,:nsparsity] != 0, axis=0),
                'robust_seq': np.array(robust_seq),
                'nit_seq': np.array(nit_seq)}


class cv(high_dim):
    '''
        Cross-Validated Penalized Expectile Regression
    '''
    penalties = ["L1", "SCAD", "MCP"]
    opt = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, \
           'tol': 1e-5, 'iter_warning': False, 'irw_tol': 1e-4}


    def __init__(self, X, Y, intercept=True, options={}):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(len(Y))
        self.itcp = intercept
        self.opt.update(options)


    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds


    def asyhuber(self, x, tau=0.5, c=False):
        '''
            Asymmetric L2 or Huber Loss 
        '''
        if not c:
            return np.sum( abs(tau - (x<0)) * x**2 )
        elif c > 0:
            tmp = abs(x)
            tmp = (tmp<=c) * tmp**2 + (2*c*tmp-c**2)*(tmp>c)
            return np.sum( np.where(x<0, 1-tau, tau) * tmp)
        else:
            raise ValueError("robustification parameter should be strictly positive")


    def fit(self, tau=0.5, \
            lambda_seq=np.array([]), nlambda=50, order='descend',
            robust=False, scale_invariant=True, \
            nfolds=5, penalty="SCAD", a=3.7, nstep=3, \
            standardize=True, adjust=True):

        rgs = high_dim(self.X, self.Y, self.itcp, self.opt)
        itcp = self.itcp

        if len(lambda_seq)==0:
            lambda_seq = rgs.lambda_seq(nlambda, standardize=standardize)
        else:
            nlambda = len(lambda_seq)

        if penalty not in self.penalties: 
            raise ValueError("penalty must be either L1, SCAD or MCP")

        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))

        for v in range(nfolds):
            X_train = self.X[np.setdiff1d(idx,folds[v]),:]
            Y_train = self.Y[np.setdiff1d(idx,folds[v])]
            X_val, Y_val = self.X[folds[v],:], self.Y[folds[v]]
            train = high_dim(X_train, Y_train, itcp, self.opt)

            if penalty == "L1":
                model = train.l1_path(tau, lambda_seq, nlambda, order, \
                                      robust, scale_invariant, \
                                      standardize, adjust)
            else:
                model = train.irw_path(tau, lambda_seq, nlambda, order, \
                                       robust, scale_invariant, \
                                       penalty, a, nstep, standardize, adjust)
                       
            val_err[v,:] = np.array([self.asyhuber(Y_val - model['beta_seq'][0,l]*itcp \
                                     - X_val.dot(model['beta_seq'][itcp:,l]), tau) \
                                     for l in range(nlambda)])
        
        cv_err = np.sum(val_err, axis=0) / self.n
        lambda_min = model['lambda_seq'][np.argmin(cv_err)]
        
        if penalty == "L1":
            cv_model = rgs.l1(tau, lambda_min, robust, scale_invariant,\
                              standardize=standardize, adjust=adjust)
        else:
            cv_model = rgs.irw(tau, lambda_min, robust, scale_invariant, \
                               penalty=penalty, a=a, nstep=nstep, \
                               standardize=standardize, adjust=adjust)

        return {'cv_beta': cv_model['beta'], 
                'cv_res': cv_model['res'], \
                'lambda_min': lambda_min, \
                'lambda_seq': model['lambda_seq'], \
                'min_cv_err': min(cv_err), \
                'cv_err': cv_err, \
                'robust': cv_model['robust']}