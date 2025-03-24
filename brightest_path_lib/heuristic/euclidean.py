from brightest_path_lib.heuristic import Heuristic
import math
import numpy as np
from typing import Tuple
from numba import njit, float64, int64, boolean

# Standalone Numba-optimized function for 2D distance calculation
@njit(fastmath=True)
def _euclidean_distance_2d(current_y, current_x, goal_y, goal_x, scale_x, scale_y):
    """Calculate Euclidean distance between two 2D points with scaling"""
    x_diff = (goal_x - current_x) * scale_x
    y_diff = (goal_y - current_y) * scale_y
    
    return math.sqrt(x_diff * x_diff + y_diff * y_diff)

# Standalone Numba-optimized function for 3D distance calculation
@njit(fastmath=True)
def _euclidean_distance_3d(current_z, current_y, current_x, goal_z, goal_y, goal_x, 
                          scale_x, scale_y, scale_z):
    """Calculate Euclidean distance between two 3D points with scaling"""
    x_diff = (goal_x - current_x) * scale_x
    y_diff = (goal_y - current_y) * scale_y
    z_diff = (goal_z - current_z) * scale_z
    
    return math.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)

# Optimized function that dispatches to either 2D or 3D implementation
@njit(fastmath=True)
def _estimate_cost(current_point, goal_point, scale_x, scale_y, scale_z):
    """Optimized cost estimation function"""
    # Fast path based on dimension
    if len(current_point) == 2:  # 2D case
        return _euclidean_distance_2d(
            current_point[0], current_point[1],  # y, x for current
            goal_point[0], goal_point[1],        # y, x for goal
            scale_x, scale_y
        )
    else:  # 3D case
        return _euclidean_distance_3d(
            current_point[0], current_point[1], current_point[2],  # z, y, x for current
            goal_point[0], goal_point[1], goal_point[2],          # z, y, x for goal
            scale_x, scale_y, scale_z
        )

class Euclidean(Heuristic):
    """heuristic cost estimation using Euclidean distance from current point to goal point

    Parameters
    ----------
    scale : Tuple
        the scale of the image's axes. For example (1.0 1.0) for a 2D image.
        - for 2D points, the order of scale is: (x, y)
        - for 3D points, the order of scale is: (x, y, z)
    
    Attributes
    ----------
    scale_x : float
        the scale of the image's X-axis
    scale_y : float
        the scale of the image's Y-axis
    scale_z : float
        the scale of the image's Z-axis

    """
    def __init__(self, scale: Tuple):
        if scale is None:
            raise TypeError
        if len(scale) == 0:
            raise ValueError

        self.scale_x = scale[0]
        self.scale_y = scale[1]
        self.scale_z = 1.0
        if len(scale) == 3:
            self.scale_z = scale[2]

    def estimate_cost_to_goal(self, current_point: np.ndarray, goal_point: np.ndarray) -> float:
        """calculates the estimated cost from current point to the goal
    
        Parameters
        ----------
        current_point : numpy ndarray
            the coordinates of the current point
        goal_point : numpy ndarray
            the coordinates of the current point
        
        Returns
        -------
        float
            the estimated cost to goal in the form of Euclidean distance
        
        Notes
        -----
        If the image is zoomed in or out, then the scale of one of more
        axes will be more or less than 1.0. For example, if the image is zoomed
        in to twice its size then the scale of X and Y axes will be 2.0.
        
        By including the scale in the calculation of distance to the goal we
        can get an accurate cost.

        - for 2D points, the order of coordinates is: (y, x)
        - for 3D points, the order of coordinates is: (z, x, y)
        """
        if current_point is None or goal_point is None:
            raise TypeError
        if (len(current_point) == 0 or len(goal_point) == 0) or (len(current_point) != len(goal_point)):
            raise ValueError

        # Use the Numba-optimized function for calculation
        return _estimate_cost(
            current_point, 
            goal_point, 
            self.scale_x, 
            self.scale_y, 
            self.scale_z
        )