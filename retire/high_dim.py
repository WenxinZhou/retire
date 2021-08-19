import numpy as np

class ilamm():
    '''
        Regularized/Penalized Expectile Regression via Iterative Local Adaptive Majorization-Minimization (ILAMM)


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
    
    def __init__(self, X, Y, intercept=True, phi=0.5, gamma=1.5, max_iter=500, tol=1e-5):

        '''
        Internal Optimization Parameters
        --------------------------------
        phi : initial quadratic coefficient parameter in the ILAMM algorithm; default is 0.5.
        
        gamma : adaptive search parameter that is larger than 1; default is 1.5.
        
        max_iter : maximum numder of iterations in the ILAMM algorithm; default is 500.
        
        tol : minimum change in (squared) Euclidean distance for stopping LAMM iterations; default is 1e-5.

        '''
        self.n, self.p = X.shape
        self.Y = Y
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.concatenate([np.ones((self.n,1)), X], axis=1)
            self.X1 = np.concatenate([np.ones((self.n,1)), (X - self.mX)/self.sdX], axis=1)
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt_para = [phi, gamma, max_iter, tol]

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
        if standardize: X = self.X1
        else: X = self.X
        lambda_max = np.max(np.abs(X.T.dot(self.Y - np.mean(self.Y))))/(self.n)
        return np.exp(np.linspace(np.log(eps*lambda_max), np.log(lambda_max), num=nlambda))
    
    def grad_weight(self, x, tau=0.5, c=0):
        '''
            Gradient Weight
        '''
        if not c:
            return -2*np.where(x>=0, tau*x, (1-tau)*x)/len(x)
        else:
            tmp1 = tau*c*(x>c) - (1-tau)*c*(x<-c)
            tmp2 = tau*x*(x>=0)*(x<=c) + (1-tau)*x*(x<0)*(x>=-c)   
            return -2*(tmp1 + tmp2)/len(x)
    
    def retire_loss(self, x, tau=0.5, c=0):
        '''
            Asymmetric Quadratic/Huber Loss 
        '''
        if not c:
            return np.mean( abs(tau - (x<0))* x**2 )
        else:
            out = (abs(x)<=c)* x**2 + (2*c*abs(x)-c**2)*(abs(x)>c)
            return np.mean( abs(tau - (x<0))*out )
    
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
    
    def l1(self, Lambda=np.array([]), tau=0.5, tune=0, beta0=np.array([]), res=np.array([]), standardize=True, adjust=True):   
        '''
                L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        Lambda : regularization parameter. This should be either a scalar, or 
                 a vector of length equal to the column dimension of X. If unspecified, 
                 it will be computed by self.lambda_seq().

        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        tune : the tuning constant in the Huberization parameter. 
               If tune = 0, the function computes penalized expectile regression estimator;
               if tune > 0, the function computes penalized Huberized expectile regression estimator.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.

        Returns
        -------

        beta0 : a numpy array of estimated coefficients.
        
        list : a list of residual vector, number of iterations and lambda value.
        '''
        if not np.array(Lambda).any(): Lambda = 2*max(tau, 1-tau)*np.median(self.lambda_seq)

        if standardize: X = self.X1
        else: X = self.X
        
        if not beta0.any():
            beta0 = np.zeros(X.shape[1])
            if self.itcp: beta0[0] = np.quantile(self.Y, tau)
            res = self.Y - beta0[0]

        phi, gamma, max_iter, tol = self.opt_para[0], self.opt_para[1], self.opt_para[2], self.opt_para[3]
        phi0, dev, count = phi, 1, 0
        while dev > tol and count <= max_iter:
            c = tune*self.mad(res)
            grad0 = X.T.dot(self.grad_weight(res, tau, c))
            loss_eval0 = self.retire_loss(res, tau, c)
            beta1 = beta0 - grad0/phi0
            beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi0)
            diff_beta = beta1 - beta0
            dev = diff_beta.dot(diff_beta)
            
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi0*dev
            loss_eval1 = self.retire_loss(res, tau, c)
            
            while loss_proxy < loss_eval1:
                phi0 *= gamma
                beta1 = beta0 - grad0/phi0
                beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi0)
                diff_beta = beta1 - beta0
                dev = diff_beta.dot(diff_beta)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi0*dev
                loss_eval1 = self.retire_loss(res, tau, c)
                
            beta0, phi0 = beta1, phi
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return beta1, [res, count, Lambda]


    def irw(self, Lambda=np.array([]), tau=0.5, tune=0, beta0=np.array([]), res=np.array([]), penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True, tol=1e-5):
        '''
            Iteratively Reweighted L1-Penalized (Huberized) Expectile Regression
            
        Arguments
        ---------
        Lambda : regularization parameter. If unspecified, 
                 it will be computed by self.lambda_seq().
        
        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        tune : the tuning constant in the Huberization parameter. 
               If tune = 0, the function computes penalized expectile regression estimator;
               if tune > 0, the function computes penalized Huberized expectile regression estimator.
        
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : the number of iterations/steps of the IRW algorithm; default is 5.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.        

        Returns
        -------
        beta0 : a numpy array of estimated coefficients.
        
        list : a list of residual vector, number of iterations and lambda value.
        '''
        
        if not np.array(Lambda).any(): Lambda = 2*max(tau, 1-tau)*np.median(self.lambda_seq)

        if not beta0.any():
            beta0, fit = self.l1(Lambda, tau, tune, standardize=standardize, adjust=False)
        else:
            beta0, fit = self.l1(Lambda, tau, tune, beta0, res, standardize, adjust=False)
        res = fit[0]

        err, count = 1, 1
        while err > tol and count <= nstep:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            beta1, fit = self.l1(rw_lambda, tau, tune, beta0, res, standardize, adjust=False)
            err = max(abs(beta1-beta0))
            beta0, res = beta1, fit[0]
            count += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
    
        return beta0, [res, count, Lambda]



    def l1_path(self, lambda_seq, tau=0.5, tune=0, standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        tune : the tuning constant in the Huberization parameter. 
               If tune = 0, the function computes penalized expectile regression estimator;
               if tune > 0, the function computes penalized Huberized expectile regression estimator.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.     

        Returns
        -------
        beta_seq : a sequence of lasso estimates. Each column corresponds to an estiamte for a lambda value.

        list : a list of redisual sequence, a sequence of model sizes, and a sequence of lambda values in ascending order.
        '''
        lambda_seq, nlambda = np.sort(lambda_seq), len(lambda_seq)
        beta_seq = np.zeros(shape=(self.X.shape[1], nlambda))
        res_seq = np.zeros(shape=(self.n, nlambda))
        beta_seq[:,0], fit = self.l1(lambda_seq[0], tau, tune, standardize=standardize, adjust=False)
        res_seq[:,0] = fit[0]
        
        for l in range(1, nlambda):
            beta_seq[:,l], fit = self.l1(lambda_seq[l], tau, tune, beta_seq[:,l-1], fit[0], standardize, adjust=False)
            res_seq[:,l] = fit[0]

        model_size = np.sum(beta_seq!=0, axis=0)
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return beta_seq, [res_seq, model_size, lambda_seq]



    def irw_path(self, lambda_seq, tau=0.5, tune=0, penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Solution Path of IRW-L1-Penalized (Huberized) Expectile Regression

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        tau : location parameter between 0 and 1 for expectile regression; default is 0.5.
        
        tune : the tuning constant in the Huberization parameter. 
               If tune = 0, the function computes penalized expectile regression estimator;
               if tune > 0, the function computes penalized Huberized expectile regression estimator.
                
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : the number of iterations/steps of the IRW algorithm; default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.

        Returns
        -------
        beta_seq : a sequence of irw-lasso estimates. Each column corresponds to an estimate for a lambda value.

        list : a list of redisual sequence, a sequence of model sizes, and a sequence of lambda values in ascending order.
        '''
        lambda_seq, nlambda = np.sort(lambda_seq), len(lambda_seq)
        beta_seq = np.empty(shape=(self.X.shape[1], nlambda))
        res_seq = np.empty(shape=(self.n, nlambda))
        beta_seq[:,0], fit = self.irw(lambda_seq[0], tau, tune, penalty=penalty, a=a, nstep=nstep, standardize=standardize, adjust=False)
        res_seq[:,0] = fit[0]
        for l in range(1, nlambda):
            beta_seq[:,l], fit = self.irw(lambda_seq[l], tau, tune, beta_seq[:,l-1], fit[0], penalty, a, nstep, standardize, adjust=False)
            res_seq[:,l] = fit[0]
        
        model_size = np.sum(beta_seq!=0, axis=0)
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
    
        return beta_seq, [res_seq, model_size, lambda_seq]


class cv(ilamm):
    '''
        Cross-Validated Penalized Expectile Regression
    '''
    penalties = ["L1", "SCAD", "MCP"]

    def __init__(self, X, Y, intercept=True):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y
        self.itcp = intercept

    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds

    def fit(self, lambda_seq=np.array([]), nlambda=40, tau=0.5, tune=0, nfolds=5, penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):

        rgs = ilamm(self.X, self.Y, self.itcp)

        if not lambda_seq.any():
            lambda_seq = rgs.lambda_seq(nlambda, standardize=standardize)
        else:
            nlambda = len(lambda_seq)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD or MCP")

        idx, folds = self.divide_sample(nfolds)
        val_error = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx,folds[v]),:], self.Y[np.setdiff1d(idx,folds[v])]
            X_val, Y_val = self.X[folds[v],:], self.Y[folds[v]]
            train = ilamm(X_train, Y_train, intercept=self.itcp)

            if penalty == "L1":
                train_beta, train_fit = train.l1_path(lambda_seq, tau, tune, standardize, adjust)
            else:
                train_beta, train_fit = train.irw_path(lambda_seq, tau, tune, penalty, a, nstep, standardize, adjust)
                       
            for l in range(nlambda):
                val_error[v,l] = self.retire_loss(Y_val - train_beta[0,l]*self.itcp - X_val.dot(train_beta[self.itcp:,l]), tau)
        
        cv_error = np.mean(val_error, axis=0)

        cv_min = min(cv_error)
        l_min = np.where(cv_error == cv_min)[0][0]
        lambda_seq, lambda_min = train_fit[2], train_fit[2][l_min]
        if penalty == "L1":
            cv_beta, cv_fit = rgs.l1(lambda_min, tau, tune, standardize=standardize, adjust=adjust)
        else:
            cv_beta, cv_fit = rgs.irw(lambda_min, tau, tune, penalty=penalty, a=a, nstep=nstep, standardize=standardize, adjust=adjust)

        return cv_beta, [cv_fit[0], lambda_min, lambda_seq, cv_min, cv_error]