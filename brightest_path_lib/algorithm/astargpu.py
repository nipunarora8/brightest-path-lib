# algorithm/astar_gpu.py

import math
import numpy as np
from queue import PriorityQueue, Queue
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any

# Import Numba CUDA modules
import numba as nb
from numba import cuda, float32, int32, int64, float64
from numba import jit, njit

from brightest_path_lib.cost import ReciprocalTransonic
from brightest_path_lib.heuristic import EuclideanTransonic
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node

# GPU helper functions
@cuda.jit(device=True)
def array_equal_device(arr1, arr2):
    """Device function to check array equality"""
    if len(arr1) != len(arr2):
        return False
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True

@cuda.jit(device=True)
def euclidean_distance_scaled_device(current_point, goal_point, scale_x, scale_y, scale_z=1.0):
    """Device function to calculate scaled Euclidean distance between two points"""
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

@cuda.jit
def compute_neighbors_costs_2d(image, node_points, node_g_scores, node_count,
                             results, result_count, x_min, x_max, y_min, y_max, 
                             min_intensity, max_intensity, reciprocal_min, reciprocal_max, 
                             min_step_cost, scale_x, scale_y, goal_point):
    """
    CUDA kernel to compute costs for all potential neighbors of multiple nodes in parallel
    
    Parameters:
    -----------
    image: 2D array - the image data
    node_points: 2D array - points of current nodes (batch of nodes to process)
    node_g_scores: 1D array - g scores of current nodes
    node_count: int - number of nodes in this batch
    results: 3D array - will store [point_y, point_x, g_score, h_score, predecessor_idx]
    result_count: 1D array - will store count of neighbors found for each node
    """
    # Get thread index
    node_idx = cuda.grid(1)
    
    # Check if this thread should process a node
    if node_idx >= node_count:
        return
    
    # Get the current node information
    node_y = node_points[node_idx, 0]
    node_x = node_points[node_idx, 1]
    g_score = node_g_scores[node_idx]
    
    # Steps for finding neighbors
    steps = (-1, 0, 1)
    
    # Counter for neighbors found for this node
    neighbor_idx = 0
    
    # Check all 8 neighbors
    for xdiff in steps:
        new_x = node_x + xdiff
        
        if new_x < x_min or new_x > x_max:
            continue
            
        for ydiff in steps:
            if xdiff == 0 and ydiff == 0:
                continue
                
            new_y = node_y + ydiff
            if new_y < y_min or new_y > y_max:
                continue
                
            # Calculate cost and scores
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
                
            # Calculate distance and g_score
            distance = math.sqrt(xdiff*xdiff + ydiff*ydiff)
            new_g_score = g_score + distance * cost
            
            # Calculate h_score using Euclidean distance
            new_point = cuda.local.array(2, dtype=int64)
            new_point[0] = new_y
            new_point[1] = new_x
            
            h_score = min_step_cost * euclidean_distance_scaled_device(
                new_point, goal_point, scale_x, scale_y)
            
            # Store results
            if neighbor_idx < 8:  # Maximum 8 neighbors for 2D
                results[node_idx, neighbor_idx, 0] = new_y
                results[node_idx, neighbor_idx, 1] = new_x
                results[node_idx, neighbor_idx, 2] = new_g_score
                results[node_idx, neighbor_idx, 3] = h_score
                results[node_idx, neighbor_idx, 4] = node_idx
                neighbor_idx += 1
    
    # Store the count of neighbors found
    result_count[node_idx] = neighbor_idx

@cuda.jit
def compute_neighbors_costs_3d(image, node_points, node_g_scores, node_count,
                             results, result_count, x_min, x_max, y_min, y_max, z_min, z_max,
                             min_intensity, max_intensity, reciprocal_min, reciprocal_max, 
                             min_step_cost, scale_x, scale_y, scale_z, goal_point):
    """
    CUDA kernel to compute costs for all potential neighbors of multiple 3D nodes in parallel
    """
    # Get thread index
    node_idx = cuda.grid(1)
    
    # Check if this thread should process a node
    if node_idx >= node_count:
        return
    
    # Get the current node information
    node_z = node_points[node_idx, 0]
    node_y = node_points[node_idx, 1]
    node_x = node_points[node_idx, 2]
    g_score = node_g_scores[node_idx]
    
    # Steps for finding neighbors
    steps = (-1, 0, 1)
    
    # Counter for neighbors found for this node
    neighbor_idx = 0
    
    # Check all 26 neighbors
    for xdiff in steps:
        new_x = node_x + xdiff
        
        if new_x < x_min or new_x > x_max:
            continue
            
        for ydiff in steps:
            new_y = node_y + ydiff
            
            if new_y < y_min or new_y > y_max:
                continue
                
            for zdiff in steps:
                if xdiff == 0 and ydiff == 0 and zdiff == 0:
                    continue
                    
                new_z = node_z + zdiff
                if new_z < z_min or new_z > z_max:
                    continue
                    
                # Calculate cost and scores
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
                    
                # Calculate distance and g_score
                distance = math.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)
                new_g_score = g_score + distance * cost
                
                # Calculate h_score using Euclidean distance
                new_point = cuda.local.array(3, dtype=int64)
                new_point[0] = new_z
                new_point[1] = new_y
                new_point[2] = new_x
                
                h_score = min_step_cost * euclidean_distance_scaled_device(
                    new_point, goal_point, scale_x, scale_y, scale_z)
                
                # Store results
                if neighbor_idx < 26:  # Maximum 26 neighbors for 3D
                    results[node_idx, neighbor_idx, 0] = new_z
                    results[node_idx, neighbor_idx, 1] = new_y
                    results[node_idx, neighbor_idx, 2] = new_x
                    results[node_idx, neighbor_idx, 3] = new_g_score
                    results[node_idx, neighbor_idx, 4] = h_score
                    results[node_idx, neighbor_idx, 5] = node_idx
                    neighbor_idx += 1
    
    # Store the count of neighbors found
    result_count[node_idx] = neighbor_idx

# Fallback CPU implementations for when GPU is not available
@njit
def array_equal(arr1, arr2):
    """Numba-compatible CPU implementation of np.array_equal"""
    if arr1.shape != arr2.shape:
        return False
    return np.all(arr1 == arr2)

@njit
def find_2D_neighbors(node_point, g_score, image, x_min, x_max, y_min, y_max, 
                     min_intensity, max_intensity, reciprocal_min, reciprocal_max, 
                     min_step_cost, scale_x, scale_y, goal_point):
    """CPU fallback for finding 2D neighbors"""
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

@njit
def euclidean_distance_scaled(current_point, goal_point, scale_x, scale_y, scale_z=1.0):
    """CPU implementation of distance calculation"""
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

class AStarGPUSearch:
    """A* Search Implementation with GPU acceleration

    Parameters
    ----------
    image : numpy ndarray
        the 2D/3D image on which we will run an A star search
    start_point : numpy ndarray
        the 2D/3D coordinates of the starting point
    goal_point : numpy ndarray
        the 2D/3D coordinates of the goal point
    scale : Tuple
        the scale of the image; defaults to (1.0, 1.0)
    cost_function : Enum CostFunction
        cost function to be used
    heuristic_function : Enum HeuristicFunction
        heuristic function to be used
    open_nodes : Queue
        visualization queue (optional)
    use_gpu : bool
        whether to use GPU acceleration (default: True)
    batch_size : int
        number of nodes to process in parallel (default: 128)
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
        use_gpu: bool = True,
        batch_size: int = 128
    ):
        self._validate_inputs(image, start_point, goal_point)

        # Check if CUDA is available
        self.has_cuda = cuda.is_available() and use_gpu
        if self.has_cuda:
            print("Using GPU acceleration")
        else:
            print("GPU acceleration not available, falling back to CPU")

        self.image = image
        self.image_stats = ImageStats(image)
        self.start_point = np.round(start_point).astype(np.int64)
        self.goal_point = np.round(goal_point).astype(np.int64)
        self.scale = scale
        self.open_nodes = open_nodes
        self.batch_size = batch_size

        # Copy image to device if using GPU
        if self.has_cuda:
            self.d_image = cuda.to_device(self.image)

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
        """Checks for a non-empty image, start point and goal point"""
        if image is None or start_point is None or goal_point is None:
            raise TypeError("Image, start_point, and goal_point must not be None")
        if len(image) == 0 or len(start_point) == 0 or len(goal_point) == 0:
            raise ValueError("Image, start_point, and goal_point must not be empty")

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
        """Performs A* search using GPU acceleration when available
        
        For 2D and 3D images, uses specialized GPU kernels to compute 
        neighbor costs in parallel batches.
        """
        if not self.has_cuda:
            # Fall back to CPU implementation
            return self._search_cpu(verbose)
            
        # Use specialized GPU implementation for 2D/3D
        is_2d = len(self.start_point) == 2
        if is_2d:
            return self._search_gpu_2d(verbose)
        else:
            return self._search_gpu_3d(verbose)
    
    def _search_cpu(self, verbose: bool = False) -> List[np.ndarray]:
        """CPU fallback implementation"""
        count = 0
        open_set = PriorityQueue()
        start_node = Node(
            point=self.start_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.start_point), 
            predecessor=None
            )
        open_set.put((0, count, start_node))
        open_set_hash = {tuple(self.start_point)}
        close_set_hash = set()
        f_scores = defaultdict(lambda: float("inf"))
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
        
        # Main search loop
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

            # Find neighbors using Numba optimized code
            if len(current_node.point) == 2:
                neighbor_data = find_2D_neighbors(
                    current_node.point, current_node.g_score, self.image,
                    x_min, x_max, y_min, y_max, 
                    min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                    min_step_cost, scale_x, scale_y, self.goal_point
                )
            else:
                # Implementation for 3D would be similar
                pass
            
            # Process neighbors
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
                        self.open_nodes.put(neighbor_coordinates)
                else:
                    if neighbor.f_score < f_scores[neighbor_coordinates]:
                        f_scores[neighbor_coordinates] = neighbor.f_score
                        count += 1
                        open_set.put((neighbor.f_score, count, neighbor))
            
            close_set_hash.add(current_coordinates)

        self.evaluated_nodes = count
        return self.result
    
    def _search_gpu_2d(self, verbose: bool = False) -> List[np.ndarray]:
        """GPU-accelerated A* search for 2D images"""
        count = 0
        open_set = PriorityQueue()
        start_node = Node(
            point=self.start_point, 
            g_score=0, 
            h_score=self._estimate_cost_to_goal(self.start_point), 
            predecessor=None
            )
        open_set.put((0, count, start_node))
        open_set_hash = {tuple(self.start_point)}
        close_set_hash = set()
        f_scores = defaultdict(lambda: float("inf"))
        f_scores[tuple(self.start_point)] = start_node.f_score
        
        # Map to store node objects by coordinates
        coord_to_node = {}
        coord_to_node[tuple(self.start_point)] = start_node
        
        # Extract parameters
        scale_x, scale_y = self.scale[0], self.scale[1]
        min_intensity = self.image_stats.min_intensity
        max_intensity = self.image_stats.max_intensity
        x_min, x_max = self.image_stats.x_min, self.image_stats.x_max
        y_min, y_max = self.image_stats.y_min, self.image_stats.y_max
        reciprocal_min = self.cost_function.RECIPROCAL_MIN
        reciprocal_max = self.cost_function.RECIPROCAL_MAX
        min_step_cost = self.cost_function.minimum_step_cost()
        
        # Transfer goal point to device
        d_goal_point = cuda.to_device(self.goal_point)
        
        # Main search loop
        while not open_set.empty():
            if self.is_canceled:
                break
                
            # Get a batch of nodes to process in parallel
            batch_nodes = []
            batch_size = min(self.batch_size, open_set.qsize())
            
            for _ in range(batch_size):
                if open_set.empty():
                    break
                    
                _, _, node = open_set.get()
                coord = tuple(node.point)
                
                if coord in close_set_hash:
                    continue
                    
                batch_nodes.append(node)
                open_set_hash.remove(coord)
                
                # Check if we've found the goal
                if array_equal(node.point, self.goal_point):
                    if verbose:
                        print("Found goal!")
                    self._construct_path_from(node)
                    self.found_path = True
                    break
            
            # If we found the goal, break the loop
            if self.found_path:
                break
                
            # If no valid nodes in batch, continue
            if not batch_nodes:
                continue
                
            # Prepare data for GPU
            node_points = np.array([node.point for node in batch_nodes], dtype=np.int64)
            node_g_scores = np.array([node.g_score for node in batch_nodes], dtype=np.float64)
            
            # Allocate memory for results
            results = np.zeros((len(batch_nodes), 8, 5), dtype=np.float64)  # [node_idx, neighbor_idx, (y, x, g_score, h_score, predecessor_idx)]
            result_count = np.zeros(len(batch_nodes), dtype=np.int32)
            
            # Transfer data to device
            d_node_points = cuda.to_device(node_points)
            d_node_g_scores = cuda.to_device(node_g_scores)
            d_results = cuda.to_device(results)
            d_result_count = cuda.to_device(result_count)
            
            # Configure kernel
            threads_per_block = 128
            blocks_per_grid = (len(batch_nodes) + threads_per_block - 1) // threads_per_block
            
            # Launch kernel
            compute_neighbors_costs_2d[blocks_per_grid, threads_per_block](
                self.d_image, d_node_points, d_node_g_scores, len(batch_nodes),
                d_results, d_result_count, x_min, x_max, y_min, y_max,
                min_intensity, max_intensity, reciprocal_min, reciprocal_max,
                min_step_cost, scale_x, scale_y, d_goal_point
            )
            
            # Copy results back
            d_results.copy_to_host(results)
            d_result_count.copy_to_host(result_count)
            
            # Process results
            for node_idx, node in enumerate(batch_nodes):
                close_set_hash.add(tuple(node.point))
                
                for neighbor_idx in range(int(result_count[node_idx])):
                    y = int(results[node_idx, neighbor_idx, 0])
                    x = int(results[node_idx, neighbor_idx, 1])
                    g_score = results[node_idx, neighbor_idx, 2]
                    h_score = results[node_idx, neighbor_idx, 3]
                    
                    new_point = np.array([y, x], dtype=np.int64)
                    neighbor_coordinates = tuple(new_point)
                    
                    if neighbor_coordinates in close_set_hash:
                        continue
                        
                    # Create neighbor node
                    neighbor = Node(
                        point=new_point,
                        g_score=g_score,
                        h_score=h_score,
                        predecessor=node
                    )
                    
                    if neighbor_coordinates not in open_set_hash:
                        count += 1
                        open_set.put((neighbor.f_score, count, neighbor))
                        open_set_hash.add(neighbor_coordinates)
                        coord_to_node[neighbor_coordinates] = neighbor
                        
                        if self.open_nodes is not None:
                            self.open_nodes.put(neighbor_coordinates)
                    else:
                        if neighbor.f_score < f_scores[neighbor_coordinates]:
                            f_scores[neighbor_coordinates] = neighbor.f_score
                            coord_to_node[neighbor_coordinates] = neighbor
                            count += 1
                            open_set.put((neighbor.f_score, count, neighbor))
        
        self.evaluated_nodes = count
        return self.result
    
    def _search_gpu_3d(self, verbose: bool = False) -> List[np.ndarray]:
        """GPU-accelerated A* search for 3D images - implementation would be similar to 2D"""
        # Implementation similar to _search_gpu_2d but with 3D operations
        # Would use compute_neighbors_costs_3d kernel
        pass
    
    def _estimate_cost_to_goal(self, point: np.ndarray) -> float:
        """Estimates the heuristic cost (h_score) from a point to the goal point"""
        return self.cost_function.minimum_step_cost() * self.heuristic_function.estimate_cost_to_goal(
            current_point=point, goal_point=self.goal_point
        )

    def _construct_path_from(self, node: Node):
        """Constructs the brightest path by backtracing from goal to start"""
        while node is not None:
            self.result.insert(0, node.point)
            node = node.predecessor