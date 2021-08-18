# retire (Robustified/Huberized Expectile Regression)

This package exploits the iterative local adaptive majorize-minimize (ILAMM) algorithm for fitting sparse linear regression models via *L<sub>1</sub>*-penalized and iteratively reweighted *L<sub>1</sub>*-penalized (IRW-*L<sub>1</sub>*) methods. It includes penalized least squares regression, penalized expectile (asymmetric least squares) regression, and penalized robustified/Huberized expectile regression (including Huber's *M*-estimator as a spectial case). The IRW method is motivated by the local linear approximation (LLA) algorithm proposed by [Zou & Li (2008)](https://projecteuclid.org/journals/annals-of-statistics/volume-36/issue-4/One-step-sparse-estimates-in-nonconcave-penalized-likelihood-models/10.1214/009053607000000802.full) for folded concave penalized estimation, typified by the SCAD penalty ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)) and the minimax concave penalty (MCP) ([Zhang, 2010](https://projecteuclid.org/journals/annals-of-statistics/volume-38/issue-2/Nearly-unbiased-variable-selection-under-minimax-concave-penalty/10.1214/09-AOS729.full)). 



## Dependencies

```
python >=3, numpy
```


## Examples

```
import numpy as np
import pandas as pd
import numpy.random as rgt
from retire.high_dim import ilamm
from sklearn.linear_model import Lasso
```
Generate data from a sparse linear model with high-dimensional covariates. The dimension of the feature/covariate space is `p`, and the sample size is `n`. The errors are generated from (i) the standard normal distribution, and (ii) the *t<sub>2</sub>*-distribution (*t*-distribution with 2 degrees of freedom). Compare `retire.high_dim.l1` with `sklearn.linear_model.Lasso` in terms of estimation error and runtime; the latter uses the coordinate descent (cd) solver.

```
n = 5000
p = 20000
itcp, beta = 2, np.zeros(p)
beta[:19] = [1.8, 0, 1.6, 0, 1.4, 0, 1.2, 0, 1, 0, -1, 0, -1.2, 0, -1.4, 0, -1.6, 0, -1.8]
sse = np.zeros([2,2])

rgt.seed(0)
X = rgt.normal(0, 1, size=(n, p))
# Normal error
Y_norm = itcp + X.dot(beta) + rgt.normal(0,1,n)
# t error 
Y_t = itcp + X.dot(beta) + rgt.standard_t(2,n)

rgs1 = Lasso(alpha=0.3)
print('sklearn.linear_model.Lasso (normal error):')
%time rgs1.fit(X, Y_norm)

rgs2 = ilamm(X, Y_norm, phi=0.5, gamma=1.5, tol=1e-4)
print('\nretire.high_dim.ilamm (normal error):')
%time l1_beta, l1_fit = rgs2.l1(Lambda=0.3)
sse[0,:] = np.array([np.sum((rgs1.coef_ - beta)**2), np.sum((l1_beta[1:] - beta)**2)])


print('\nsklearn.linear_model.Lasso (t error):')
rgs1 = Lasso(alpha=0.3)
%time rgs1.fit(X, Y_t)

rgs2 = ilamm(X, Y_t, phi=0.5, gamma=1.5, tol=1e-4)
print('\nretire.high_dim.ilamm (t error):')
%time l1_beta, l1_fit = rgs2.l1(Lambda=0.3)
sse[1,:] = np.array([np.sum((rgs1.coef_ - beta)**2), np.sum((l1_beta[1:] - beta)**2)])

pd.DataFrame(sse, columns=['cd', 'ilamm'], index=['normal', 't_2'])
```


## References

Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. *Ann. Statist.* **46**(2) 814-841. [Paper](https://www.tandfonline.com/doi/abs/10.1080/07350015.2019.1660177?journalCode=ubes20)

Gu, Y. and Zou, H. (2016). High-dimensional generalizations of asymmetric least squares regression and their application. *Ann. Statist.* **44**(6) 2661-2694. [Paper](https://projecteuclid.org/journals/annals-of-statistics/volume-44/issue-6/High-dimensional-generalizations-of-asymmetric-least-squares-regression-and-their/10.1214/15-AOS1431.full)

Newey, W. K. and Powell, J. L. (1987). Asymmetric least squares estimation and testing. *Econometrica*. **55**(4) 819-847. [Paper](https://www.jstor.org/stable/1911031?seq=1#metadata_info_tab_contents)

Pan, X., Sun, Q. and Zhou, W.-X. (2021). Iteratively reweighted *l<sub>1</sub>*-penalized robust regression. *Electron. J. Stat.* **15**(1) 3287-3348. [Paper](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-15/issue-1/Iteratively-reweighted-%E2%84%931-penalized-robust-regression/10.1214/21-EJS1862.full)

Tibshirani. R. (1996). Regression shrinkage and selection via the lasso. *J. Roy. Statist. Soc. Ser. B* **58**(1) 267-288. [Paper](https://www.jstor.org/stable/2346178?seq=1#metadata_info_tab_contents)


## License 

This package is released under the GPL-3.0 license.
