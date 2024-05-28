from .heuristic import Heuristic
from .euclidean import Euclidean

# 20240527, making a version to not use transonic
# was this
# from .euclidean_transonic import EuclideanTransonic
from .euclidean import Euclidean as EuclideanTransonic

DO_TRANSONIC = False
if DO_TRANSONIC:
    from .reciprocal_transonic import EuclideanTransonic
else:
    from .euclidean import Euclidean as EuclideanTransonic
