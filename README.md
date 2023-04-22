# pyexpectreg (Robust Expectile and Quantile Regression)

Estimation and inference methods for expectile (asymmetric least squares) regression and its robust/Huberized variants ([Man et al. (2022)](https://drive.google.com/file/d/1ldm9DhtB-yd3drPZfRFQ7JjUV51ocUHz/view)). The iterative local adaptive majorize-minimize (ILAMM) algorithm is employed for computing *L<sub>1</sub>*-penalized and iteratively reweighted *L<sub>1</sub>*-penalized (IRW-*L<sub>1</sub>*) (robust) expectile regression estimates. Special cases include penalized least squares and Huber regressions. The IRW method is motivated by the local linear approximation (LLA) algorithm proposed by [Zou & Li (2008)](https://doi.org/10.1214/009053607000000802) for folded concave penalized estimation, typified by the SCAD penalty ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)) and the minimax concave penalty (MCP) ([Zhang, 2010](https://doi.org/10.1214/09-AOS729)).



## Dependencies

```
python >=3, numpy, scipy
```


## Examples

```
import numpy as np
import pandas as pd
import numpy.random as rgt
from pyexpectreg.retire import high_dim
from sklearn.linear_model import Lasso
```
Generate data from a sparse linear model with high-dimensional covariates. The dimension of the feature/covariate space is `p`, and the sample size is `n`. The errors are generated from (i) the standard normal distribution, and (ii) the *t<sub>2</sub>*-distribution (*t*-distribution with 2 degrees of freedom). Compare `retire.high_dim.l1` with `sklearn.linear_model.Lasso` in terms of estimation error and runtime; the latter uses the coordinate descent (cd) solver.

```
n, p = 5000, 20000
itcp, beta = 2, np.zeros(p)
beta[:19] = [1.8, 0, 1.6, 0, 1.4, 0, 1.2, 0, 1, 0, -1, 0, -1.2, 0, -1.4, 0, -1.6, 0, -1.8]
sse = np.zeros([2,2])

rgt.seed(0)
X = rgt.normal(0, 1, size=(n, p))
Y_norm = itcp + X.dot(beta) + rgt.normal(0,1,n) # normal error
Y_t = itcp + X.dot(beta) + rgt.standard_t(2,n)	# t_2 error

rgs1 = Lasso(alpha=0.1)
print('sklearn.linear_model.Lasso (normal error):')
%time rgs1.fit(X, Y_norm)

model1 = high_dim(X, Y_norm)
print('\nretire.high_dim.l1 (normal error):')
%time model1 = model1.l1(Lambda=0.1)
sse[0,:] = np.array([np.sum((rgs1.coef_ - beta)**2), np.sum((model1['beta'][1:] - beta)**2)])

print('\nsklearn.linear_model.Lasso (t error):')
rgs2 = Lasso(alpha=0.1)
%time rgs2.fit(X, Y_t)

model2 = high_dim(X, Y_t)
print('\nretire.high_dim.l1 (t error):')
%time model2 = model2.l1(Lambda=0.1)
sse[1,:] = np.array([np.sum((rgs2.coef_ - beta)**2), np.sum((model2['beta'][1:] - beta)**2)])

pd.DataFrame(sse, columns=['cd', 'ilamm'], index=['normal', 't_2'])
```


## References

Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. *Ann. Statist.* **46**(2) 814-841. [Paper](https://doi.org/10.1214/17-AOS1568)

Gu, Y. and Zou, H. (2016). High-dimensional generalizations of asymmetric least squares regression and their application. *Ann. Statist.* **44**(6) 2661-2694. [Paper](https://doi.org/10.1214/15-AOS1431)

Man, R., Tan, K. M., Wang, Z. and Zhou, W.-X. (2022). Retire: Robustified expectile regression in high dimensions. [Paper](https://drive.google.com/file/d/1ldm9DhtB-yd3drPZfRFQ7JjUV51ocUHz/view)


Newey, W. K. and Powell, J. L. (1987). Asymmetric least squares estimation and testing. *Econometrica*. **55**(4) 819-847. [Paper](https://doi.org/10.2307/1911031)

Pan, X., Sun, Q. and Zhou, W.-X. (2021). Iteratively reweighted *l<sub>1</sub>*-penalized robust regression. *Electron. J. Stat.* **15**(1) 3287-3348. [Paper](https://doi.org/10.1214/21-EJS1862)

Tibshirani. R. (1996). Regression shrinkage and selection via the lasso. *J. Roy. Statist. Soc. Ser. B* **58**(1) 267-288. [Paper](https://www.jstor.org/stable/2346178)


## License 

This package is released under the GPL-3.0 license.
