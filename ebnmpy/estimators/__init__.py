from .normal import NormalEBNM
from .point_exponential import PointExponentialEBNM
from .point_laplace import PointLaplaceEBNM
from .point_normal import PointNormalEBNM

estimators = dict(
    normal=NormalEBNM,
    point_normal=PointNormalEBNM,
    point_laplace=PointLaplaceEBNM,
    point_exponential=PointExponentialEBNM,
)
