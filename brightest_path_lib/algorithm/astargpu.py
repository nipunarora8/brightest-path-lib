# algorithm/astar_cuda.py

"""GPU-accelerated implementation of the A* search algorithm for brightest path finding
using Numba's CUDA capabilities.

This implementation is designed to leverage Nvidia GPUs to accelerate the computationally
intensive parts of the A* algorithm, particularly neighbor finding and cost calculations.

Requirements:
- CUDA-compatible GPU (Nvidia)
- Numba with CUDA support
- cupy (optional, for additional GPU array operations)

To search for the brightest path between two points in an image:

1. Initialize the AStarCUDASearch class with the 2D/3D image,
   start point and the goal point: `astar = AStarCUDASearch(image, start_point, goal_point)`
2. Call the search method: `path = astar.search()`
"""

from collections import defaultdict
import math
import numpy as np
from queue import PriorityQueue, Queue
from typing import List, Tuple, Dict, Set, Any, Optional
import numba as nb
from numba import cuda, float64, int64, boolean

from brightest_path_lib.cost import ReciprocalTransonic
from brightest_path_lib.heuristic import EuclideanTransonic
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node


# ================ CUDA Kernels ================

@cuda.jit(device=True)
def device_array_equal(arr1, arr2):
    """CUDA device function to check if two arrays are equal"""
    if arr1.shape[0] != arr2.shape[0]:
        return False
    for i in range(arr1.shape[0]):
        if arr1[i] != arr2[i]:
            return False
    return True


@cuda.jit(device=True)
def device_euclidean_distance_2d(y1, x1, y2, x2, scale_x, scale_y):
    """CUDA device function to calculate 2D Euclidean distance with scaling"""
    x_diff = (x2 - x1) * scale_x
    y_diff = (y2 - y1) * scale_y
    return math.sqrt(x_diff * x_diff + y_diff * y_diff)


@cuda.jit(device=True)
def device_euclidean_distance_3d(z1, y1, x1, z2, y2, x2, scale_x, scale_y, scale_z):
    """CUDA device function to calculate 3D Euclidean distance with scaling"""
    x_diff = (x2 - x1) * scale_x
    y_diff = (y2 - y1) * scale_y
    z_diff = (z2 - z1) * scale_z
    return math.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)


@cuda.jit(device=True)
def device_calculate_cost(intensity, min_intensity, max_intensity, 
                          reciprocal_min, reciprocal_max, min_step_cost):
    """CUDA device function to calculate cost from intensity"""
    # Normalize intensity
    norm_intensity = reciprocal_max * (intensity - min_intensity) / (max_intensity - min_intensity)
    
    # Ensure minimum value
    if norm_intensity < reciprocal_min:
        norm_intensity = reciprocal_min
    
    # Calculate cost
    cost = 1.0 / norm_intensity
    
    # Ensure minimum cost
    if cost < min_step_cost:
        return min_step_cost
    return cost


@cuda.jit
def find_2d_neighbors_kernel(node_points, g_scores, image, 
                           x_min, x_max, y_min, y_max,
                           min_intensity, max_intensity, 
                           reciprocal_min, reciprocal_max, min_step_cost,
                           scale_x, scale_y, goal_point,
                           neighbor_points, neighbor_g_scores, neighbor_h_scores,
                           neighbor_counts, max_neighbors_per_node):
    """
    CUDA kernel to find 2D neighbors for multiple nodes in parallel
    
    Parameters
    ----------
    node_points : device array
        Array of node points to find neighbors for, shape (n_nodes, 2)
    g_scores : device array
        G-scores for each node, shape (n_nodes,)
    image : device array
        Image data
    x_min, x_max, y_min, y_max : int
        Image coordinate bounds
    min_intensity, max_intensity : float
        Image intensity bounds
    reciprocal_min, reciprocal_max : float
        Cost function parameters
    min_step_cost : float
        Minimum cost of moving to a neighbor
    scale_x, scale_y : float
        Scale factors for x and y dimensions
    goal_point : device array
        Coordinates of the goal point, shape (2,)
    neighbor_points : device array
        Output array for neighbor coordinates, shape (n_nodes, max_neighbors_per_node, 2)
    neighbor_g_scores : device array
        Output array for neighbor g-scores, shape (n_nodes, max_neighbors_per_node)
    neighbor_h_scores : device array
        Output array for neighbor h-scores, shape (n_nodes, max_neighbors_per_node)
    neighbor_counts : device array
        Output array for counting valid neighbors per node, shape (n_nodes,)
    max_neighbors_per_node : int
        Maximum number of neighbors to store per node
    """
    # Get thread index
    node_idx = cuda.grid(1)
    
    # Check if thread is within bounds
    if node_idx >= node_points.shape[0]:
        return
    
    # Get node point and g-score
    node_y = node_points[node_idx, 0]
    node_x = node_points[node_idx, 1]
    node_g_score = g_scores[node_idx]
    
    # Initialize neighbor count for this node
    count = 0
    
    # Direction steps
    steps = (-1, 0, 1)
    
    # Loop through all possible neighbors
    for i in range(3):
        xdiff = steps[i]
        new_x = node_x + xdiff
        
        if new_x < x_min or new_x > x_max:
            continue
            
        for j in range(3):
            ydiff = steps[j]
            
            # Skip the center point (current node)
            if xdiff == 0 and ydiff == 0:
                continue
                
            new_y = node_y + ydiff
            if new_y < y_min or new_y > y_max:
                continue
                
            # Make sure we don't exceed maximum neighbors
            if count >= max_neighbors_per_node:
                break
                
            # Calculate distance
            distance = math.sqrt(xdiff*xdiff + ydiff*ydiff)
            
            # Get intensity and calculate cost
            intensity = float(image[new_y, new_x])
            cost = device_calculate_cost(intensity, min_intensity, max_intensity,
                                        reciprocal_min, reciprocal_max, min_step_cost)
            
            # Calculate g-score
            new_g_score = node_g_score + distance * cost
            
            # Calculate h-score
            h_score = min_step_cost * device_euclidean_distance_2d(
                new_y, new_x, goal_point[0], goal_point[1], scale_x, scale_y)
            
            # Store neighbor data
            neighbor_points[node_idx, count, 0] = new_y
            neighbor_points[node_idx, count, 1] = new_x
            neighbor_g_scores[node_idx, count] = new_g_score
            neighbor_h_scores[node_idx, count] = h_score
            
            # Increment count
            count += 1
    
    # Store final count
    neighbor_counts[node_idx] = count


@cuda.jit
def find_3d_neighbors_kernel(node_points, g_scores, image, 
                           x_min, x_max, y_min, y_max, z_min, z_max,
                           min_intensity, max_intensity, 
                           reciprocal_min, reciprocal_max, min_step_cost,
                           scale_x, scale_y, scale_z, goal_point,
                           neighbor_points, neighbor_g_scores, neighbor_h_scores,
                           neighbor_counts, max_neighbors_per_node):
    """
    CUDA kernel to find 3D neighbors for multiple nodes in parallel
    
    Parameters
    ----------
    node_points : device array
        Array of node points to find neighbors for, shape (n_nodes, 3)
    g_scores : device array
        G-scores for each node, shape (n_nodes,)
    image : device array
        Image data
    x_min, x_max, y_min, y_max, z_min, z_max : int
        Image coordinate bounds
    min_intensity, max_intensity : float
        Image intensity bounds
    reciprocal_min, reciprocal_max : float
        Cost function parameters
    min_step_cost : float
        Minimum cost of moving to a neighbor
    scale_x, scale_y, scale_z : float
        Scale factors for x, y, and z dimensions
    goal_point : device array
        Coordinates of the goal point, shape (3,)
    neighbor_points : device array
        Output array for neighbor coordinates, shape (n_nodes, max_neighbors_per_node, 3)
    neighbor_g_scores : device array
        Output array for neighbor g-scores, shape (n_nodes, max_neighbors_per_node)
    neighbor_h_scores : device array
        Output array for neighbor h-scores, shape (n_nodes, max_neighbors_per_node)
    neighbor_counts : device array
        Output array for counting valid neighbors per node, shape (n_nodes,)
    max_neighbors_per_node : int
        Maximum number of neighbors to store per node
    """
    # Get thread index
    node_idx = cuda.grid(1)
    
    # Check if thread is within bounds
    if node_idx >= node_points.shape[0]:
        return
    
    # Get node point and g-score
    node_z = node_points[node_idx, 0]
    node_y = node_points[node_idx, 1]
    node_x = node_points[node_idx, 2]
    node_g_score = g_scores[node_idx]
    
    # Initialize neighbor count for this node
    count = 0
    
    # Direction steps
    steps = (-1, 0, 1)
    
    # Loop through all possible neighbors
    for i in range(3):
        xdiff = steps[i]
        new_x = node_x + xdiff
        
        if new_x < x_min or new_x > x_max:
            continue
            
        for j in range(3):
            ydiff = steps[j]
            new_y = node_y + ydiff
            
            if new_y < y_min or new_y > y_max:
                continue
                
            for k in range(3):
                zdiff = steps[k]
                
                # Skip the center point (current node)
                if xdiff == 0 and ydiff == 0 and zdiff == 0:
                    continue
                    
                new_z = node_z + zdiff
                if new_z < z_min or new_z > z_max:
                    continue
                    
                # Make sure we don't exceed maximum neighbors
                if count >= max_neighbors_per_node:
                    break
                    
                # Calculate distance
                distance = math.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)
                
                # Get intensity and calculate cost
                intensity = float(image[new_z, new_y, new_x])
                cost = device_calculate_cost(intensity, min_intensity, max_intensity,
                                          reciprocal_min, reciprocal_max, min_step_cost)
                
                # Calculate g-score
                new_g_score = node_g_score + distance * cost
                
                # Calculate h-score
                h_score = min_step_cost * device_euclidean_distance_3d(
                    new_z, new_y, new_x, 
                    goal_point[0], goal_point[1], goal_point[2], 
                    scale_x, scale_y, scale_z)
                
                # Store neighbor data
                neighbor_points[node_idx, count, 0] = new_z
                neighbor_points[node_idx, count, 1] = new_y
                neighbor_points[node_idx, count, 2] = new_x
                neighbor_g_scores[node_idx, count] = new_g_score
                neighbor_h_scores[node_idx, count] = h_score
                
                # Increment count
                count += 1
    
    # Store final count
    neighbor_counts[node_idx] = count


class AStarCUDASearch:
    """A* Search Implementation with CUDA acceleration
    
    This class implements the A* search algorithm with CUDA acceleration for
    finding the brightest path in an image. The most computationally intensive
    parts of the algorithm, such as neighbor finding and cost calculations,
    are offloaded to the GPU.

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
    batch_size : int
        The number of nodes to process in parallel on the GPU
        Default value is 128
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        scale: Tuple = (1.0, 1.0),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN,
        open_nodes: Queue = None,
        batch_size: int = 128
    ):
        self._validate_inputs(image, start_point, goal_point)

        # Basic parameters
        self.image = image
        self.image_stats = ImageStats(image)
        self.start_point = np.round(start_point).astype(np.int64)
        self.goal_point = np.round(goal_point).astype(np.int64)
        self.scale = scale
        self.open_nodes = open_nodes
        self.batch_size = batch_size
        
        # Determine if 2D or 3D
        self.is_2d = len(self.start_point) == 2
        
        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = ReciprocalTransonic(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = EuclideanTransonic(scale=self.scale)
        
        # Initialize GPU variables
        self._init_gpu_data()
        
        # Algorithm state
        self.is_canceled = False
        self.found_path = False
        self.evaluated_nodes = 0
        self.result = []
        
        # Maximum neighbors per node
        self.max_neighbors_per_node = 8 if self.is_2d else 26
    
    def _init_gpu_data(self):
        """Initialize GPU data and transfer constant data to the device"""
        # Transfer image to GPU
        self.d_image = cuda.to_device(self.image)
        
        # Transfer goal point to GPU
        self.d_goal_point = cuda.to_device(self.goal_point)
        
        # Extract and prepare constants needed for kernels
        self.min_intensity = self.image_stats.min_intensity
        self.max_intensity = self.image_stats.max_intensity
        self.x_min = self.image_stats.x_min
        self.x_max = self.image_stats.x_max
        self.y_min = self.image_stats.y_min
        self.y_max = self.image_stats.y_max
        self.z_min = self.image_stats.z_min
        self.z_max = self.image_stats.z_max
        
        # Scale factors
        self.scale_x = self.scale[0]
        self.scale_y = self.scale[1]
        self.scale_z = 1.0 if len(self.scale) < 3 else self.scale[2]
        
        # Cost function parameters
        self.reciprocal_min = self.cost_function.RECIPROCAL_MIN
        self.reciprocal_max = self.cost_function.RECIPROCAL_MAX
        self.min_step_cost = self.cost_function.minimum_step_cost()
    
    def _validate_inputs(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
    ):
        """Validate input parameters"""
        if image is None or start_point is None or goal_point is None:
            raise TypeError("Image, start_point, and goal_point must not be None")
        if len(image) == 0 or len(start_point) == 0 or len(goal_point) == 0:
            raise ValueError("Image, start_point, and goal_point must not be empty")
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available. Make sure you have a compatible GPU and drivers installed.")

    @property
    def found_path(self) -> bool:
        return self._found_path

    @found_path.setter
    def found_path(self, value: bool):
        if value is None:
            raise TypeError("found_path value cannot be None")
        self._found_path = value

    @property
    def is_canceled(self) -> bool:
        return self._is_canceled

    @is_canceled.setter
    def is_canceled(self, value: bool):
        if value is None:
            raise TypeError("is_canceled value cannot be None")
        self._is_canceled = value

    def search(self, verbose: bool = False) -> List[np.ndarray]:
        """Performs A* search for the brightest path using CUDA acceleration
        
        Parameters
        ----------
        verbose : bool
            Whether to print progress information
            
        Returns
        -------
        List[np.ndarray]
            List of coordinates forming the brightest path
        """
        # Clear previous results
        self.result = []
        self.evaluated_nodes = 0
        self._found_path = False
        
        # Basic initialization
        count = 0
        open_set = PriorityQueue()
        
        # Create start node
        start_node = Node(
            point=self.start_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.start_point), 
            predecessor=None
        )
        
        # Add start node to open set
        open_set.put((0, count, start_node))
        
        # Keep track of nodes in open set and closed set
        open_set_hash = {tuple(self.start_point)}
        close_set_hash = set()
        
        # Track best f_scores for each point
        f_scores = {}
        f_scores[tuple(self.start_point)] = start_node.f_score
        
        # Map coordinates to Node objects for quick lookup
        coord_to_node = {}
        coord_to_node[tuple(self.start_point)] = start_node
        
        # Main search loop
        while not open_set.empty():
            if self.is_canceled:
                break
            
            # Get current node with lowest f_score
            current_node = open_set.get()[2]
            current_coordinates = tuple(current_node.point)
            
            # Skip if already processed
            if current_coordinates in close_set_hash:
                continue
            
            # Remove from open set
            open_set_hash.remove(current_coordinates)
            
            # Check if goal reached
            if np.array_equal(current_node.point, self.goal_point):
                if verbose:
                    print("Found goal!")
                self._construct_path_from(current_node)
                self.found_path = True
                break
            
            # Process neighbors using GPU
            # Instead of processing one node at a time, we collect a batch for GPU processing
            batch = [(current_node, current_coordinates)]
            batch_points = [current_node.point]
            batch_g_scores = [current_node.g_score]
            
            # Try to collect a batch of nodes from the open set
            while len(batch) < self.batch_size and not open_set.empty():
                _, _, next_node = open_set.get()
                next_coordinates = tuple(next_node.point)
                if next_coordinates not in close_set_hash:
                    batch.append((next_node, next_coordinates))
                    batch_points.append(next_node.point)
                    batch_g_scores.append(next_node.g_score)
                    open_set_hash.remove(next_coordinates)
            
            # Find neighbors for all nodes in the batch using GPU
            neighbors_data = self._find_neighbors_cuda(
                np.array(batch_points, dtype=np.int64),
                np.array(batch_g_scores, dtype=np.float64)
            )
            
            # Process results from GPU
            for i, (node, node_coords) in enumerate(batch):
                # Add node to closed set
                close_set_hash.add(node_coords)
                
                # Get neighbors for this node
                node_neighbors = neighbors_data[i]
                
                # Process each neighbor
                for new_point, g_score, h_score in node_neighbors:
                    neighbor_coordinates = tuple(new_point)
                    
                    if neighbor_coordinates in close_set_hash:
                        # Already processed this neighbor
                        continue
                    
                    # Calculate f_score
                    f_score = g_score + h_score
                    
                    if neighbor_coordinates not in open_set_hash:
                        # New node, add to open set
                        neighbor = Node(
                            point=new_point,
                            g_score=g_score,
                            h_score=h_score,
                            predecessor=node
                        )
                        
                        count += 1
                        open_set.put((f_score, count, neighbor))
                        open_set_hash.add(neighbor_coordinates)
                        f_scores[neighbor_coordinates] = f_score
                        coord_to_node[neighbor_coordinates] = neighbor
                        
                        if self.open_nodes is not None:
                            self.open_nodes.put(neighbor_coordinates)
                    elif neighbor_coordinates in f_scores and f_score < f_scores[neighbor_coordinates]:
                        # Better path found, update
                        neighbor = Node(
                            point=new_point,
                            g_score=g_score,
                            h_score=h_score,
                            predecessor=node
                        )
                        
                        f_scores[neighbor_coordinates] = f_score
                        coord_to_node[neighbor_coordinates] = neighbor
                        count += 1
                        open_set.put((f_score, count, neighbor))
            
            # Update evaluated nodes count
            self.evaluated_nodes += len(batch)
        
        return self.result
    
    def _find_neighbors_cuda(self, node_points, g_scores):
        """Find neighbors for a batch of nodes using CUDA
        
        Parameters
        ----------
        node_points : numpy.ndarray
            Array of node points, shape (batch_size, 2) for 2D or (batch_size, 3) for 3D
        g_scores : numpy.ndarray
            Array of g-scores, shape (batch_size,)
            
        Returns
        -------
        list
            List of lists of (point, g_score, h_score) tuples for each node in the batch
        """
        # Number of nodes in the batch
        n_nodes = node_points.shape[0]
        
        # Define CUDA grid and block dimensions
        threads_per_block = 128
        blocks_per_grid = (n_nodes + threads_per_block - 1) // threads_per_block
        
        # Transfer node data to GPU
        d_node_points = cuda.to_device(node_points)
        d_g_scores = cuda.to_device(g_scores)
        
        # Allocate output arrays on GPU
        point_dim = 2 if self.is_2d else 3
        d_neighbor_points = cuda.device_array((n_nodes, self.max_neighbors_per_node, point_dim), dtype=np.int64)
        d_neighbor_g_scores = cuda.device_array((n_nodes, self.max_neighbors_per_node), dtype=np.float64)
        d_neighbor_h_scores = cuda.device_array((n_nodes, self.max_neighbors_per_node), dtype=np.float64)
        d_neighbor_counts = cuda.device_array(n_nodes, dtype=np.int64)
        
        # Launch kernel based on dimension
        if self.is_2d:
            find_2d_neighbors_kernel[blocks_per_grid, threads_per_block](
                d_node_points, d_g_scores, self.d_image,
                self.x_min, self.x_max, self.y_min, self.y_max,
                self.min_intensity, self.max_intensity,
                self.reciprocal_min, self.reciprocal_max, self.min_step_cost,
                self.scale_x, self.scale_y, self.d_goal_point,
                d_neighbor_points, d_neighbor_g_scores, d_neighbor_h_scores,
                d_neighbor_counts, self.max_neighbors_per_node
            )
        else:
            find_3d_neighbors_kernel[blocks_per_grid, threads_per_block](
                d_node_points, d_g_scores, self.d_image,
                self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max,
                self.min_intensity, self.max_intensity,
                self.reciprocal_min, self.reciprocal_max, self.min_step_cost,
                self.scale_x, self.scale_y, self.scale_z, self.d_goal_point,
                d_neighbor_points, d_neighbor_g_scores, d_neighbor_h_scores,
                d_neighbor_counts, self.max_neighbors_per_node
            )
        
        # Synchronize to ensure kernel execution is complete
        cuda.synchronize()
        
        # Copy results back to host
        neighbor_points = d_neighbor_points.copy_to_host()
        neighbor_g_scores = d_neighbor_g_scores.copy_to_host()
        neighbor_h_scores = d_neighbor_h_scores.copy_to_host()
        neighbor_counts = d_neighbor_counts.copy_to_host()
        
        # Process results into the expected format
        result = []
        for i in range(n_nodes):
            node_neighbors = []
            for j in range(neighbor_counts[i]):
                point = neighbor_points[i, j].copy()
                g_score = neighbor_g_scores[i, j]
                h_score = neighbor_h_scores[i, j]
                node_neighbors.append((point, g_score, h_score))
            result.append(node_neighbors)
        
        return result
    
    def _estimate_cost_to_goal(self, point: np.ndarray) -> float:
        """Estimate heuristic cost from a point to the goal"""
        return self.min_step_cost * self.heuristic_function.estimate_cost_to_goal(
            current_point=point, goal_point=self.goal_point
        )

    def _construct_path_from(self, node: Node):
        """Construct path by tracing back from goal to start"""
        while node is not None:
            self.result.insert(0, node.point)
            node = node.predecessor