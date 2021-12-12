[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
# ebnmpy: Python package for fitting the empirical Bayes normal means (EBNM) model
ebnmpy is a Python package that implements functionalities provided the R package [ebnm](https://github.com/stephenslab/ebnm).
For more details, refer to the original paper [Willwerscheid, J., & Stephens, M. (2021). ebnm: An R Package for Solving the Empirical Bayes Normal Means Problem Using a Variety of Prior Families.](https://arxiv.org/abs/2110.00152)

Code for reproducing results in [Willwerscheid and Stephens, 2021](https://arxiv.org/abs/2110.00152) are located in the [paper](https://github.com/kclamar/ebnmpy/tree/master/paper) folder.

## Installation

```bash
git clone https://github.com/kclamar/ebnmpy.git
cd ebnmpy
pip install -e .[paper]
```

## Example
```python
import numpy as np
from ebnmpy.estimators import PointNormalEBNM

# simulate data
np.random.seed(0)
n = 10000
s = np.random.rand(n) + 1
x = np.concatenate((np.zeros(n // 2), np.random.normal(0, 10, n // 2)))\
    + np.random.normal(0, s)

# run ebnm
estimator = PointNormalEBNM(mode="estimate").fit(x=x, s=s)
```
