from .point_normal import PointNormalEBNM


class NormalEBNM(PointNormalEBNM):
    @property
    def _pointmass(self) -> bool:
        return False
