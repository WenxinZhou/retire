import numpy as np

class ilamm():
    '''
        Penalized Expectile Regression via Iterative Local Adaptive Majorization-Minimization (ILAMM)

    References
    ----------
    One-step Sparse Estimates in Nonconcave Penalized Likelihood Models (2008)
    by Hui Zou and Runze Li
    The Annals of Statistics 36(4): 1509–1533.

    I-LAMM for Sparse Learning: Simultaneous Control of Algorithmic Complexity and Statistical Error (2018)
    by Jianqing Fan, Han Liu, Qiang Sun and Tong Zhang
    The Annals of Statistics 46(2): 814-841.

    Iteratively Reweighted l1-Penalized Robust Regression (2021)
    by Xiaoou Pan, Qiang Sun and Wenxin Zhou
    Electronic Journal of Statistics 15(1): 3287-3348.
    '''
    
    opt = {'phi': 0.5, 'gamma': 1.25, 'max_iter': 1e3, \
           'tol': 1e-5, 'irw_tol': 1e-4, 'nboot': 200}  

    def __init__(self, X, Y, intercept=True, options={}):

        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.
           
        Y : n-dimensional vector of response variables.
            
        intercept : logical flag for adding an intercept to the model.

        options : a dictionary of internal statistical and optimization parameters.
        
            phi : initial quadratic coefficient parameter in the ILAMM algorithm; default is 0.1.
        
            gamma : adaptive search parameter that is larger than 1; default is 1.25.
        
            max_iter : maximum numder of iterations in the ILAMM algorithm; default is 1e3.
        
            tol : the ILAMM iteration stops when |beta^{k+1} - beta^k|^2/|beta^k|^2 <= tol; default is 1e-5.

            irw_tol : tolerance parameter for stopping iteratively reweighted L1-penalizations; default is 1e-4.

            nboot : number of bootstrap samples for post-selection inference; default is 200.
        '''
        self.n, self.p = X.shape
        self.Y = Y
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X]
            self.X1 = np.c_[np.ones(self.n), (X - self.mX)/self.sdX]
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt.update(options)

    def soft_thresh(self, x, c):
        '''
            Soft-thresholding Operator
        '''
        tmp = abs(x) - c
        return np.sign(x)*np.where(tmp<=0, 0, tmp)
    
    def mad(self, x):
        '''
            Median Absolute Deviation
        '''
        return np.median(abs(x - np.median(x)))*1.4826
        
    def lambda_seq(self, nlambda=50, eps=0.01, standardize=True):
        '''
        Arguments
        ---------
        nlambda : number of lambda values in the sequence; default is 50.

        eps : minimum lambda is set to be eps * maximum lambda; default is 0.01.
        '''
        if standardize: X = self.X1
        else: X = self.X
        lambda_max = np.max(np.abs(X.T.dot(self.Y - np.mean(self.Y))))/(self.n)
        return np.exp(np.linspace(np.log(eps*lambda_max), np.log(lambda_max), num=nlambda))
    
    def grad_weight(self, x, tau=0.5, c=None):
        '''
            Gradient Weight
        '''
        if c == None:
            return -2*np.where(x>=0, tau*x, (1-tau)*x)/len(x)
        if c > 0:
            tmp1 = tau*c*(x>c) - (1-tau)*c*(x<-c)
            tmp2 = tau*x*(x>=0)*(x<=c) + (1-tau)*x*(x<0)*(x>=-c)   
            return -2*(tmp1 + tmp2)/len(x)
        if c <= 0: 
            raise ValueError("robustification parameter should be strictly positive")

    
    def retire_loss(self, x, tau=0.5, c=None):
        '''
            Asymmetric Quadratic/Huber Loss 
        '''
        if c == None:
            return np.mean( abs(tau - (x<0))* x**2 )
        if c > 0:
            out = (abs(x)<=c) * x**2 + (2*c*abs(x)-c**2)*(abs(x)>c)
            return np.mean( abs(tau - (x<0))*out )
        if c <= 0: 
            raise ValueError("robustification parameter should be strictly positive")
    
    def concave_weight(self, x, penalty="SCAD", a=None):
        if penalty == "SCAD":
            if a==None: a = 3.7
            tmp = 1 - (abs(x)-1)/(a-1)
            tmp = np.where(tmp<=0, 0, tmp)
            return np.where(tmp>1, 1, tmp)
        if penalty == "MCP":
            if a==None: a = 3
            tmp = 1 - abs(x)/a 
            return np.where(tmp<=0, 0, tmp)
        if penalty == "CapppedL1":
            if a==None: a = 3
            return 1*(abs(x) <= a/2)
    
    def l1(self, Lambda=np.array([]), tau=0.5, robust=None, \
           beta0=np.array([]), res=np.array([]), \
           standardize=True, adjust=True):
        '''
                L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        Lambda : regularization parameter. This should be either a scalar, or 
                 a vector of length equal to the column dimension of X. If unspecified, 
                 it will be computed by self.lambda_seq().

        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        robust : robustification constant in the Huberization parameter. 
                 If robust = None, the function computes penalized expectile regression estimator;
                 if robust > 0, the function computes penalized Huberized expectile regression estimator.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.

        Returns
        -------
        'beta' : a numpy array of estimated coefficients.
        
        'res' : a numpy array of fitted residuals.

        'intercept' : logical flag for fitting an intercept to the model.

        'count' : number of iterations. 

        'lambda' : lambda value.

        'robust' : output robustification parameter.
        '''
        if not np.array(Lambda).any(): Lambda = max(tau, 1-tau)*np.median(self.lambda_seq)

        if standardize: X = self.X1
        else: X = self.X
        
        if len(beta0)==0:
            beta0 = np.zeros(X.shape[1])
            if self.itcp: beta0[0] = np.quantile(self.Y, tau)
            res = self.Y - beta0[0]

        phi, dev, count = self.opt['phi'], 1, 0
        while dev > self.opt['tol']*np.sum(beta0**2) and count < self.opt['max_iter']:
            
            if robust != None: trun = robust*self.mad(res)
            else: trun = robust
            
            grad0 = X.T.dot(self.grad_weight(res, tau, trun))
            loss_eval0 = self.retire_loss(res, tau, trun)
            beta1 = beta0 - grad0/phi
            beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi)
            diff_beta = beta1 - beta0
            dev = diff_beta.dot(diff_beta)
            
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*dev
            loss_eval1 = self.retire_loss(res, tau, trun)
            
            while loss_proxy < loss_eval1:
                phi *= self.opt['gamma']
                beta1 = beta0 - grad0/phi
                beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi)
                diff_beta = beta1 - beta0
                dev = diff_beta.dot(diff_beta)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*dev
                loss_eval1 = self.retire_loss(res, tau, trun)
            
            beta0, phi = beta1, (self.opt['phi'] + phi)/2
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return {'beta': beta1, 'res': res, 'intercept': self.itcp, \
                'niter': count, 'lambda': Lambda, 'robust': trun}



    def irw(self, Lambda=np.array([]), tau=0.5, robust=None, beta0=np.array([]), res=np.array([]), \
            penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Iteratively Reweighted L1-Penalized (Huberized) Expectile Regression
            
        Arguments
        ---------
        Lambda : regularization parameter. If unspecified, 
                 it will be computed by self.lambda_seq().
        
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        robust : the robustification constant in the Huberization parameter. 
                 If robust = 0, the function computes penalized expectile regression estimator;
                 if robust > 0, the function computes penalized Huberized expectile regression estimator.
        
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : number of iterations/steps of the IRW algorithm; default is 5.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.        
        
        Returns
        -------
        'beta' : a numpy array of estimated coefficients.
        
        'res' : a numpy array of fitted residuals.

        'intercept' : logical flag for fitting an intercept to the model.

        'lambda' : lambda value.

        'robust' : output robustification parameter.
        '''
        
        if not np.array(Lambda).any(): Lambda = 2*max(tau, 1-tau)*np.median(self.lambda_seq)

        if len(beta0)==0:
            model = self.l1(Lambda, tau, robust, standardize=standardize, adjust=False)
        else:
            model = self.l1(Lambda, tau, robust, beta0, res, standardize, adjust=False)
        beta0, res = model['beta'], model['res']

        rel_dev, step = 1, 1
        while rel_dev > self.opt['irw_tol'] and step <= nstep:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            model = self.l1(rw_lambda, tau, robust, model['beta'], model['res'], standardize, adjust=False)
            rel_dev = np.sum((model['beta'] - beta0)**2)/np.sum(beta0**2)
            beta0, res, step = model['beta'], model['res'], step+1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
    
        return {'beta': beta0, 'res': res, 'intercept': self.itcp, \
                'lambda': Lambda, 'robust': model['robust']}


    def l1_path(self, lambda_seq, tau=0.5, robust=None, standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        robust : the robustification constant in the Huberization parameter. 
                 If robust = 0, the function computes penalized expectile regression estimator;
                 if robust > 0, the function computes penalized Huberized expectile regression estimator.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.     

        Returns
        -------
        'beta_seq' : a sequence of l1-retire estimates. Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of fitted residual vectors.

        'size_seq' : a sequence of selected model sizes. 

        'lambda_seq' : a sequence of lambda values in descending order.
        '''
        lambda_seq, nlambda = np.sort(lambda_seq)[::-1], len(lambda_seq)
        beta_seq = np.zeros(shape=(self.X.shape[1], nlambda))
        res_seq = np.zeros(shape=(self.n, nlambda))
        
        model = self.l1(lambda_seq[0], tau, robust, standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        
        for l in range(1, nlambda):
            model = self.l1(lambda_seq[l], tau, robust, model['beta'], model['res'], standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']

        size_seq = np.sum(beta_seq!=0, axis=0)
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta_seq': beta_seq, 'res_seq': res_seq, \
                'size_seq': size_seq, 'lambda_seq': lambda_seq}


    def irw_path(self, lambda_seq, tau=0.5, robust=None, \
                 penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Solution Path of IRW-L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        robust : the robustification constant in the Huberization parameter. 
                 If robust = 0, the function computes penalized expectile regression estimator;
                 if robust > 0, the function computes penalized Huberized expectile regression estimator.
                
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : number of iterations/steps of the IRW algorithm; default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.

        Returns
        -------
        'beta_seq' : a sequence of irw-l1-retire estimates. Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of fitted residual vectors.

        'size_seq' : a sequence of selected model sizes. 

        'lambda_seq' : a sequence of lambda values in descending order.
        '''
        lambda_seq, nlambda = np.sort(lambda_seq)[::-1], len(lambda_seq)
        beta_seq = np.empty(shape=(self.X.shape[1], nlambda))
        res_seq = np.empty(shape=(self.n, nlambda))
        model = self.irw(lambda_seq[0], tau, robust, \
                                      penalty=penalty, a=a, nstep=nstep, standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']

        for l in range(1, nlambda):
            model = self.irw(lambda_seq[l], tau, robust, model['beta'], model['res'], \
                             penalty, a, nstep, standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
        
        size_seq = np.sum(beta_seq!=0, axis=0)
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
    
        return {'beta_seq': beta_seq, 'res_seq': res_seq, \
                'size_seq': size_seq, 'lambda_seq': lambda_seq}



class cv(ilamm):
    '''
        Cross-Validated Penalized Expectile Regression
    '''
    penalties = ["L1", "SCAD", "MCP"]
    opt = {'phi': 0.5, 'gamma': 1.25, 'max_iter': 1e3, \
           'tol': 1e-5, 'irw_tol': 1e-4, 'nboot': 200}

    def __init__(self, X, Y, intercept=True, options={}):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y
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

    def fit(self, lambda_seq=np.array([]), nlambda=50, tau=0.5, robust=None, nfolds=5, \
            penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):

        rgs = ilamm(self.X, self.Y, self.itcp, self.opt)

        if len(lambda_seq)==0:
            lambda_seq = rgs.lambda_seq(nlambda, standardize=standardize)
        else:
            nlambda = len(lambda_seq)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD or MCP")

        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))

        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx,folds[v]),:], self.Y[np.setdiff1d(idx,folds[v])]
            X_val, Y_val = self.X[folds[v],:], self.Y[folds[v]]
            train = ilamm(X_train, Y_train, self.itcp, self.opt)

            if penalty == "L1":
                model = train.l1_path(lambda_seq, tau, robust, standardize, adjust)
            else:
                model = train.irw_path(lambda_seq, tau, robust, penalty, a, nstep, standardize, adjust)
                       
            val_err[v,:] = np.array([self.retire_loss(Y_val - model['beta_seq'][0,l]*self.itcp \
                                     - X_val.dot(model['beta_seq'][self.itcp:,l]), tau) for l in range(nlambda)])
        
        cv_err = np.mean(val_err, axis=0)
        cv_min = min(cv_err)
        l_min = np.where(cv_err == cv_min)[0][0]
        lambda_min = model['lambda_seq'][l_min]
        
        if penalty == "L1":
            cv_model = rgs.l1(lambda_min, tau, robust, standardize=standardize, adjust=adjust)
        else:
            cv_model = rgs.irw(lambda_min, tau, robust, penalty=penalty, a=a, nstep=nstep, \
                               standardize=standardize, adjust=adjust)

        return {'cv_beta': cv_model['beta'], 'cv_res': cv_model['res'], \
                'lambda_min': lambda_min, 'lambda_seq': model['lambda_seq'], \
                'min_cv_err': cv_min, 'cv_err': cv_err, 'robust': cv_model['robust']}




