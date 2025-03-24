# algorithm/astar.py

"""This implementation of the A* search algorithm finds the brightest path
in a graph. Each node in the graph represents a point in space, and the
weight of each edge represents the brightness of the path between the two nodes. 
The goal is to find the path that maximizes the total brightness.
     
To use the A* search algorithm for brightest path finding, a heuristic 
function is needed that estimates the maximum brightness that can be 
achieved from the current node to the goal node. One way to do this 
is to use a greedy heuristic that always selects the edge with the
highest brightness from the current node. 

The A* search algorithm starts at the start node and explores
neighboring nodes, selecting the node with the highest brightness
at each step. The algorithm uses the heuristic function to estimate
the maximum brightness that can be achieved from the current node to
the goal node. If the algorithm reaches the goal node, it terminates
and returns the brightest path. If not, it continues searching until
all nodes have been explored.

To search for the brightest path between two points in an image:

1. Initialize the AStarSearch class with the 2D/3D image,
   start point and the goal point: `astar = AStarSearch(image, start_point, goal_point)`
2. Call the search method: `path = astar.search()`
"""

from collections import defaultdict
import math
import numpy as np
from queue import PriorityQueue, Queue
from typing import List, Tuple, Dict, Set, Any
import numba as nb
from numba import types

from brightest_path_lib.cost import ReciprocalTransonic
from brightest_path_lib.heuristic import EuclideanTransonic
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node


# Numba helper functions
@nb.njit
def array_equal(arr1, arr2):
    """Numba-compatible implementation of np.array_equal"""
    if arr1.shape != arr2.shape:
        return False
    return np.all(arr1 == arr2)


@nb.njit(fastmath=True)
def euclidean_distance_scaled(current_point, goal_point, scale_x, scale_y, scale_z=1.0):
    """Calculate scaled Euclidean distance between two points"""
    if len(current_point) == 2:  # 2D case
        current_x, current_y = current_point[1], current_point[0]
        goal_x, goal_y = goal_point[1], goal_point[0]
        
        x_diff = (goal_x - current_x) * scale_x
        y_diff = (goal_y - current_y) * scale_y
        
        return math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
    else:  # 3D case
        current_z, current_y, current_x = current_point[0], current_point[1], current_point[2]
        goal_z, goal_y, goal_x = goal_point[0], goal_point[1], goal_point[2]
        
        x_diff = (goal_x - current_x) * scale_x
        y_diff = (goal_y - current_y) * scale_y
        z_diff = (goal_z - current_z) * scale_z
        
        return math.sqrt((x_diff * x_diff) + (y_diff * y_diff) + (z_diff * z_diff))


@nb.njit
def find_2D_neighbors(node_point, g_score, image, x_min, x_max, y_min, y_max, 
                     min_intensity, max_intensity, reciprocal_min, reciprocal_max, 
                     min_step_cost, scale_x, scale_y, goal_point):
    """Find 2D neighbors of a node (optimized with Numba)"""
    neighbors = []
    steps = np.array([-1, 0, 1])
    
    for xdiff in steps:
        new_x = node_point[1] + xdiff
        if new_x < x_min or new_x > x_max:
            continue

        for ydiff in steps:
            if xdiff == 0 and ydiff == 0:
                continue

            new_y = node_point[0] + ydiff
            if new_y < y_min or new_y > y_max:
                continue

            new_point = np.array([new_y, new_x], dtype=np.int64)
            
            # Calculate h_score
            h_score = min_step_cost * euclidean_distance_scaled(
                new_point, goal_point, scale_x, scale_y)
            
            # Calculate cost of moving
            intensity = float(image[new_y, new_x])
            
            # Normalize intensity
            norm_intensity = reciprocal_max * (intensity - min_intensity) / (max_intensity - min_intensity)
            
            # Ensure minimum value
            if norm_intensity < reciprocal_min:
                norm_intensity = reciprocal_min
            
            cost = 1.0 / norm_intensity
            
            # Ensure minimum cost
            if cost < min_step_cost:
                cost = min_step_cost
                
            # Calculate g_score
            distance = math.sqrt(xdiff*xdiff + ydiff*ydiff)
            new_g_score = g_score + distance * cost
            
            # Create neighbor data
            neighbors.append((new_point, new_g_score, h_score))
    
    return neighbors


@nb.njit
def find_3D_neighbors(node_point, g_score, image, x_min, x_max, y_min, y_max, z_min, z_max,
                     min_intensity, max_intensity, reciprocal_min, reciprocal_max, 
                     min_step_cost, scale_x, scale_y, scale_z, goal_point):
    """Find 3D neighbors of a node (optimized with Numba)"""
    neighbors = []
    steps = np.array([-1, 0, 1])
    
    for xdiff in steps:
        new_x = node_point[2] + xdiff
        if new_x < x_min or new_x > x_max:
            continue

        for ydiff in steps:
            new_y = node_point[1] + ydiff
            if new_y < y_min or new_y > y_max:
                continue

            for zdiff in steps:
                if xdiff == 0 and ydiff == 0 and zdiff == 0:
                    continue

                new_z = node_point[0] + zdiff
                if new_z < z_min or new_z > z_max:
                    continue

                new_point = np.array([new_z, new_y, new_x], dtype=np.int64)
                
                # Calculate h_score
                h_score = min_step_cost * euclidean_distance_scaled(
                    new_point, goal_point, scale_x, scale_y, scale_z)
                
                # Calculate cost of moving
                intensity = float(image[new_z, new_y, new_x])
                
                # Normalize intensity
                norm_intensity = reciprocal_max * (intensity - min_intensity) / (max_intensity - min_intensity)
                
                # Ensure minimum value
                if norm_intensity < reciprocal_min:
                    norm_intensity = reciprocal_min
                
                cost = 1.0 / norm_intensity
                
                # Ensure minimum cost
                if cost < min_step_cost:
                    cost = min_step_cost
                    
                # Calculate g_score
                distance = math.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)
                new_g_score = g_score + distance * cost
                
                # Create neighbor data
                neighbors.append((new_point, new_g_score, h_score))
    
    return neighbors


class AStarSearch:
    """A* Search Implementation with Numba optimization

    Parameters
    ----------
    image : numpy ndarray
        the 2D/3D image on which we will run an A star search
    start_point : numpy ndarray
        the 2D/3D coordinates of the starting point (could be a pixel or a voxel)
        For 2D images, the coordinates are of the form (y, x)
        For 3D images, the coordinates are of the form (z, x, y)
    goal_point : numpy ndarray
        the 2D/3D coordinates of the goal point (could be a pixel or a voxel)
        For 2D images, the coordinates are of the form (y, x)
        For 3D images, the coordinates are of the form (z, x, y)
    scale : Tuple
        the scale of the image; defaults to (1.0, 1.0), i.e. image is not zoomed in/out
        For 2D images, the scale is of the form (x, y)
        For 3D images, the scale is of the form (x, y, z)
    cost_function : Enum CostFunction
        this enum value specifies the cost function to be used for computing 
        the cost of moving to a new point
        Default type is CostFunction.RECIPROCAL to use the reciprocal function
    heuristic_function : Enum HeuristicFunction
        this enum value specifies the heuristic function to be used to compute
        the estimated cost of moving from a point to the goal
        Default type is HeuristicFunction.EUCLIDEAN to use the 
        euclidean function for cost estimation
    open_nodes : Queue
        contains a list of points that are in the open set;
        can be used by the calling application to show a visualization
        of where the algorithm is searching currently
        Default value is None
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        scale: Tuple = (1.0, 1.0),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN,
        open_nodes: Queue = None
    ):

        self._validate_inputs(image, start_point, goal_point)

        self.image = image
        self.image_stats = ImageStats(image)
        self.start_point = np.round(start_point).astype(np.int64)
        self.goal_point = np.round(goal_point).astype(np.int64)
        self.scale = scale
        self.open_nodes = open_nodes

        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = ReciprocalTransonic(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = EuclideanTransonic(scale=self.scale)
        
        self.is_canceled = False
        self.found_path = False
        self.evaluated_nodes = 0
        self.result = []

    def _validate_inputs(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
    ):
        """Checks for a non-empty image, start point and goal point before
        the A* search
        """
        if image is None or start_point is None or goal_point is None:
            raise TypeError
        if len(image) == 0 or len(start_point) == 0 or len(goal_point) == 0:
            raise ValueError

    @property
    def found_path(self) -> bool:
        return self._found_path

    @found_path.setter
    def found_path(self, value: bool):
        if value is None:
            raise TypeError
        self._found_path = value

    @property
    def is_canceled(self) -> bool:
        return self._is_canceled

    @is_canceled.setter
    def is_canceled(self, value: bool):
        if value is None:
            raise TypeError
        self._is_canceled = value

    def search(self, verbose: bool = False) -> List[np.ndarray]:
        """Performs A star search to find the brightest path

        Parameters
        ----------
        verbose (bool)
            If True, will print `Found goal!` when goal is found.

        Returns
        -------
        List[np.ndarray]
            the list containing the 2D/3D point coordinates
            that constitute the brightest path between the
            start_point and the goal_point
        """
        count = 0
        open_set = PriorityQueue()
        start_node = Node(
            point=self.start_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.start_point), 
            predecessor=None
            )
        open_set.put((0, count, start_node))  # f_score, count: priority of occurence, current node
        open_set_hash = {tuple(self.start_point)}  # hashset contains tuple of node coordinates to be visited
        close_set_hash = set()  # hashset contains tuple of node coordinates already been visited
        f_scores = defaultdict(self._default_value)  # key: tuple of node coordinates, value: f_score
        f_scores[tuple(self.start_point)] = start_node.f_score
        
        # Extract values needed for Numba-optimized neighbor finding
        scale_x, scale_y = self.scale[0], self.scale[1]
        scale_z = 1.0
        if len(self.scale) == 3:
            scale_z = self.scale[2]
            
        min_intensity = self.image_stats.min_intensity
        max_intensity = self.image_stats.max_intensity
        x_min, x_max = self.image_stats.x_min, self.image_stats.x_max
        y_min, y_max = self.image_stats.y_min, self.image_stats.y_max
        z_min, z_max = self.image_stats.z_min, self.image_stats.z_max
        
        # Extract cost function parameters
        reciprocal_min = self.cost_function.RECIPROCAL_MIN
        reciprocal_max = self.cost_function.RECIPROCAL_MAX
        min_step_cost = self.cost_function.minimum_step_cost()
        
        while not open_set.empty():
            if self.is_canceled:
                break
                
            current_node = open_set.get()[2]
            current_coordinates = tuple(current_node.point)
            
            if current_coordinates in close_set_hash:
                continue
            
            open_set_hash.remove(current_coordinates)

            if array_equal(current_node.point, self.goal_point):
                if verbose:
                    print("Found goal!")
                self._construct_path_from(current_node)
                self.found_path = True
                break

            # Use Numba-optimized neighbor finding
            if len(current_node.point) == 2:  # 2D case
                neighbor_data = find_2D_neighbors(
                    current_node.point, current_node.g_score, self.image,
                    x_min, x_max, y_min, y_max, 
                    min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                    min_step_cost, scale_x, scale_y, self.goal_point
                )
            else:  # 3D case
                neighbor_data = find_3D_neighbors(
                    current_node.point, current_node.g_score, self.image,
                    x_min, x_max, y_min, y_max, z_min, z_max,
                    min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                    min_step_cost, scale_x, scale_y, scale_z, self.goal_point
                )
            
            # Process neighbor data and create Node objects
            for new_point, g_score, h_score in neighbor_data:
                neighbor_coordinates = tuple(new_point)
                
                if neighbor_coordinates in close_set_hash:
                    # this neighbor has already been visited
                    continue
                    
                # Create a Node object for the neighbor
                neighbor = Node(
                    point=new_point,
                    g_score=g_score,
                    h_score=h_score,
                    predecessor=current_node
                )
                
                if neighbor_coordinates not in open_set_hash:
                    count += 1
                    open_set.put((neighbor.f_score, count, neighbor))
                    open_set_hash.add(neighbor_coordinates)
                    if self.open_nodes is not None:
                        # add to our queue
                        # can be monitored from caller to update plots
                        self.open_nodes.put(neighbor_coordinates)
                else:
                    if neighbor.f_score < f_scores[neighbor_coordinates]:
                        f_scores[neighbor_coordinates] = neighbor.f_score
                        count += 1
                        open_set.put((neighbor.f_score, count, neighbor))
            
            close_set_hash.add(current_coordinates)

        self.evaluated_nodes = count
        return self.result
    
    def _default_value(self) -> float:
        """the default value f_score of all nodes in the image

        Returns
        -------
        float
            returns infinity as the default f_score
        """
        return float("inf")
    
    def _find_neighbors_of(self, node: Node) -> List[Node]:
        """Finds the neighbors of a node (2D/3D)

        Parameters
        ----------
        node : Node
            the node whose neighbors we are interested in
        
        Returns
        -------
        List[Node]
            a list of nodes that are the neighbors of the given node
        """
        if len(node.point) == 2:
            return self._find_2D_neighbors_of(node)
        else:
            return self._find_3D_neighbors_of(node)
    
    def _find_2D_neighbors_of(self, node: Node) -> List[Node]:
        """Finds the neighbors of a 2D node (Legacy method, not used in optimized search)"""
        neighbors = []
        steps = [-1, 0, 1]
        for xdiff in steps:
            new_x = node.point[1] + xdiff
            if new_x < self.image_stats.x_min or new_x > self.image_stats.x_max:
                continue

            for ydiff in steps:
                if xdiff == ydiff == 0:
                    continue

                new_y = node.point[0] + ydiff
                if new_y < self.image_stats.y_min or new_y > self.image_stats.y_max:
                    continue

                new_point = np.array([new_y, new_x])

                h_for_new_point = self._estimate_cost_to_goal(new_point)

                intensity_at_new_point = self.image[new_y, new_x]

                cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(float(intensity_at_new_point))
                if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                    cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                g_for_new_point = node.g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff)) * cost_of_moving_to_new_point
                neighbor = Node(
                    point=new_point,
                    g_score=g_for_new_point,
                    h_score=h_for_new_point,
                    predecessor=node
                )

                neighbors.append(neighbor)

        return neighbors

    def _find_3D_neighbors_of(self, node: Node) -> List[Node]:
        """Finds the neighbors of a 3D node (Legacy method, not used in optimized search)"""
        neighbors = []
        steps = [-1, 0, 1]
        
        for xdiff in steps:
            new_x = node.point[2] + xdiff
            if new_x < self.image_stats.x_min or new_x > self.image_stats.x_max:
                continue

            for ydiff in steps:
                new_y = node.point[1] + ydiff
                if new_y < self.image_stats.y_min or new_y > self.image_stats.y_max:
                    continue

                for zdiff in steps:
                    if xdiff == ydiff == zdiff == 0:
                        continue

                    new_z = node.point[0] + zdiff
                    if new_z < self.image_stats.z_min or new_z > self.image_stats.z_max:
                        continue

                    new_point = np.array([new_z, new_y, new_x])

                    h_for_new_point = self._estimate_cost_to_goal(new_point)

                    intensity_at_new_point = self.image[new_z, new_y, new_x]
                    cost_of_moving_to_new_point = self.cost_function.cost_of_moving_to(float(intensity_at_new_point))
                    if cost_of_moving_to_new_point < self.cost_function.minimum_step_cost():
                        cost_of_moving_to_new_point = self.cost_function.minimum_step_cost()

                    g_for_new_point = node.g_score + math.sqrt((xdiff*xdiff) + (ydiff*ydiff) + (zdiff*zdiff)) * cost_of_moving_to_new_point
                    neighbor = Node(
                        point=new_point,
                        g_score=g_for_new_point,
                        h_score=h_for_new_point,
                        predecessor=node
                    ) 

                    neighbors.append(neighbor)
                
        return neighbors
    
    def _found_goal(self, point: np.ndarray) -> bool:
        """Checks if the goal point is reached

        Parameters
        ----------
        point : numpy ndarray
            the point whose coordinates are to be compared to the goal
            point for equality

        Returns
        -------
        bool
            returns True if the goal is reached; False otherwise
        """
        return np.array_equal(point, self.goal_point)

    def _estimate_cost_to_goal(self, point: np.ndarray) -> float:
        """Estimates the heuristic cost (h_score)
        from a point to the goal point

        Parameters
        ----------
        point : numpy ndarray
            the point from which we have to estimate the heuristic cost to
            goal

        Returns
        -------
        float
            returns the heuristic cost between the point and goal point
        """
        return self.cost_function.minimum_step_cost() * self.heuristic_function.estimate_cost_to_goal(
            current_point=point, goal_point=self.goal_point
        )

    def _construct_path_from(self, node: Node):
        """constructs the brightest path upon reaching the goal_point by 
        backtracing steps from goal point to the start point
        
        Parameters
        ----------
        node : Node
            a node that lies on the brightest path
        """
        while node is not None:
            self.result.insert(0, node.point)
            node = node.predecessor