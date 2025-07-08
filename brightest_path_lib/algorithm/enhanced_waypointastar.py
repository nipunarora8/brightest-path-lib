"""Optimized Enhanced A* search implementation with intelligent waypoint processing and automatic z-range detection.

This extends the WaypointBidirectionalAStarSearch to support:
1. Automatic z-range detection based on intensity thresholds
2. Intelligent waypoint z-positioning 
3. Automatic start/end point detection from a list of points
4. User-friendly single-frame point selection workflow
5. Performance optimizations for speed without sacrificing accuracy
"""

import heapq
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any, Optional
import numba as nb
from numba import njit, prange, jit

from brightest_path_lib.algorithm.astar import (
    BidirectionalAStarSearch, array_equal, euclidean_distance_scaled,
    find_2D_neighbors_optimized, find_3D_neighbors_optimized
)
from brightest_path_lib.cost import Reciprocal
from brightest_path_lib.heuristic import Euclidean
from brightest_path_lib.image import ImageStats
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node


# Optimized Numba functions with parallel processing
@nb.njit(cache=True, parallel=True)
def find_intensity_transitions_at_point_optimized(image, y, x, intensity_threshold=0.3):
    """Find z-frames where intensity transitions occur (appearance/disappearance) - OPTIMIZED"""
    z_size = image.shape[0]
    intensities = np.zeros(z_size)
    
    # Extract intensity profile along z-axis with parallel processing
    for z in nb.prange(z_size):
        intensities[z] = image[z, y, x]
    
    # Find peak intensity and threshold
    max_intensity = np.max(intensities)
    threshold = max_intensity * intensity_threshold
    
    # Vectorized transition detection
    above_threshold = intensities >= threshold
    
    # Find first frame above threshold (appearance)
    start_z = -1
    for z in range(z_size):
        if above_threshold[z]:
            start_z = z
            break
    
    # Find last frame above threshold (disappearance)
    end_z = -1
    for z in range(z_size - 1, -1, -1):
        if above_threshold[z]:
            end_z = z
            break
    
    # Handle edge cases
    if start_z == -1:
        start_z = 0
    if end_z == -1:
        end_z = z_size - 1
    
    return start_z, end_z, max_intensity


@nb.njit(cache=True, parallel=True)
def batch_find_optimal_z_for_waypoints(image, waypoint_coords, z_min, z_max):
    """Find optimal z-coordinates for multiple waypoints in parallel - OPTIMIZED"""
    num_waypoints = waypoint_coords.shape[0]
    optimal_z_values = np.zeros(num_waypoints, dtype=np.int32)
    
    for i in nb.prange(num_waypoints):
        y, x = waypoint_coords[i, 0], waypoint_coords[i, 1]
        best_z = z_min
        best_intensity = 0.0
        
        for z in range(z_min, min(z_max + 1, image.shape[0])):
            intensity = image[z, y, x]
            if intensity > best_intensity:
                best_intensity = intensity
                best_z = z
        
        optimal_z_values[i] = best_z
    
    return optimal_z_values


@nb.njit(cache=True)
def distribute_waypoints_across_z_range_optimized(num_waypoints, start_z, end_z):
    """Distribute waypoints evenly across the z-range - OPTIMIZED"""
    if num_waypoints == 0:
        return np.empty(0, dtype=np.int32)
    
    if num_waypoints == 1:
        return np.array([(start_z + end_z) // 2], dtype=np.int32)
    
    # For multiple waypoints, distribute evenly
    z_positions = np.zeros(num_waypoints, dtype=np.int32)
    if end_z > start_z:
        step = (end_z - start_z) / (num_waypoints + 1)
        for i in range(num_waypoints):
            z_positions[i] = int(start_z + (i + 1) * step)
    else:
        # If start_z == end_z, all waypoints get the same z
        z_positions.fill(start_z)
    
    return z_positions


@nb.njit(cache=True)
def filter_close_waypoints_optimized(waypoints, min_distance=3):
    """Remove waypoints that are too close to each other - OPTIMIZED"""
    if len(waypoints) <= 1:
        return waypoints
    
    # Always keep first waypoint
    filtered_indices = [0]
    
    for i in range(1, len(waypoints)):
        # Check distance to all previously kept waypoints
        keep_waypoint = True
        for j in filtered_indices:
            distance_sq = 0.0
            for k in range(waypoints.shape[1]):
                diff = waypoints[i, k] - waypoints[j, k]
                distance_sq += diff * diff
            
            if distance_sq < min_distance * min_distance:
                keep_waypoint = False
                break
        
        if keep_waypoint:
            filtered_indices.append(i)
    
    # Create filtered array
    filtered = np.zeros((len(filtered_indices), waypoints.shape[1]), dtype=waypoints.dtype)
    for i, idx in enumerate(filtered_indices):
        for j in range(waypoints.shape[1]):
            filtered[i, j] = waypoints[idx, j]
    
    return filtered


@nb.njit(cache=True)
def calculate_euclidean_distance_fast(point1, point2):
    """Fast euclidean distance calculation - OPTIMIZED"""
    distance_sq = 0.0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        distance_sq += diff * diff
    return math.sqrt(distance_sq)


def find_start_end_points_optimized(points_list):
    """
    Find the two points with maximum distance as start and end points - OPTIMIZED
    Returns remaining points as waypoints.
    """
    if len(points_list) < 2:
        raise ValueError("Need at least 2 points to find start and end")
    
    # Convert to numpy array for faster computation
    points_array = np.array(points_list, dtype=np.float64)
    
    max_distance = 0.0
    start_idx = 0
    end_idx = 1
    
    # Find the two points with maximum distance
    for i in range(len(points_array)):
        for j in range(i + 1, len(points_array)):
            distance = calculate_euclidean_distance_fast(points_array[i], points_array[j])
            if distance > max_distance:
                max_distance = distance
                start_idx = i
                end_idx = j
    
    # Extract start and end points
    start_point = points_array[start_idx].copy()
    end_point = points_array[end_idx].copy()
    
    # Create waypoints array excluding start and end points
    waypoint_indices = []
    for i in range(len(points_array)):
        if i != start_idx and i != end_idx:
            waypoint_indices.append(i)
    
    if waypoint_indices:
        waypoints = points_array[waypoint_indices]
    else:
        waypoints = np.empty((0, points_array.shape[1]))
    
    return waypoints.tolist(), start_point, end_point


def should_use_hierarchical_search(image, num_waypoints):
    """Determine if hierarchical search should be used - OPTIMIZED"""
    image_size = np.prod(image.shape)
    complexity_factor = image_size * (num_waypoints + 1)
    
    # Use hierarchical search for very large images
    return complexity_factor > 100_000_000  # 100M operations threshold


class ZRangeCache:
    """Cache z-range calculations to avoid recomputation - OPTIMIZED"""
    def __init__(self):
        self.cache = {}
    
    def get_z_range(self, image, y, x, intensity_threshold):
        key = (y, x, intensity_threshold)
        if key not in self.cache:
            self.cache[key] = find_intensity_transitions_at_point_optimized(image, y, x, intensity_threshold)
        return self.cache[key]


class EnhancedWaypointAStarSearch:
    """Enhanced waypoint A* search with intelligent z-range detection and waypoint positioning - OPTIMIZED
    
    This implementation provides:
    1. Automatic z-range detection based on intensity thresholds
    2. Intelligent waypoint z-positioning for optimal path finding
    3. Automatic start/end point detection from point lists
    4. User-friendly single-frame workflow
    5. Performance optimizations for speed without sacrificing accuracy
    """

    def __init__(
        self,
        image: np.ndarray,
        points_list: List[List] = None,
        start_point: np.ndarray = None,
        goal_point: np.ndarray = None,
        waypoints: List[np.ndarray] = None,
        scale: Tuple = (1.0, 1.0),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN,
        open_nodes=None,
        use_hierarchical: bool = None,  # Auto-determine if None
        weight_heuristic: float = 1.0,  # Always use optimal weight
        intensity_threshold: float = 0.3,
        auto_z_detection: bool = True,
        waypoint_z_optimization: bool = True,
        filter_close_waypoints: bool = True,
        min_waypoint_distance: float = 3.0,
        verbose: bool = False
    ):
        """Initialize enhanced waypoint A* search - OPTIMIZED
        
        Parameters
        ----------
        image : numpy ndarray
            The 3D image to search
        points_list : List[List], optional
            List of points [[z,y,x], ...] for automatic start/end/waypoint detection
        start_point, goal_point : numpy ndarray, optional
            Explicit start and goal coordinates (if not using points_list)
        waypoints : List[numpy ndarray], optional
            Explicit waypoints (if not using points_list)
        scale : tuple
            Image scale factors
        cost_function, heuristic_function : Enum
            Functions to use for cost and heuristic
        open_nodes : Queue, optional
            Queue for visualization
        use_hierarchical : bool, optional
            Whether to use hierarchical search (auto-determined if None)
        weight_heuristic : float
            Weight for heuristic (kept at 1.0 for optimality)
        intensity_threshold : float
            Threshold for z-range detection (fraction of peak intensity)
        auto_z_detection : bool
            Whether to automatically detect optimal z-range
        waypoint_z_optimization : bool
            Whether to optimize waypoint z-positions
        filter_close_waypoints : bool
            Whether to filter out waypoints that are too close together
        min_waypoint_distance : float
            Minimum distance between waypoints (if filtering enabled)
        verbose : bool
            Whether to print detailed information
        """
        self.image = image
        self.scale = scale
        self.open_nodes = open_nodes
        self.weight_heuristic = weight_heuristic
        self.intensity_threshold = intensity_threshold
        self.auto_z_detection = auto_z_detection
        self.waypoint_z_optimization = waypoint_z_optimization
        self.filter_close_waypoints = filter_close_waypoints
        self.min_waypoint_distance = min_waypoint_distance
        self.verbose = verbose
        
        # Initialize z-range cache for performance
        self.z_range_cache = ZRangeCache()
        
        # Auto-determine hierarchical search if not specified
        if use_hierarchical is None:
            num_wp = len(waypoints) if waypoints else len(points_list) - 2 if points_list else 0
            self.use_hierarchical = should_use_hierarchical_search(image, num_wp)
        else:
            self.use_hierarchical = use_hierarchical
        
        # Process input points
        if points_list is not None:
            self._process_points_list_optimized(points_list)
        else:
            if start_point is None or goal_point is None:
                raise ValueError("Either points_list or both start_point and goal_point must be provided")
            self.start_point = np.round(start_point).astype(np.int32)
            self.goal_point = np.round(goal_point).astype(np.int32)
            self.waypoints = np.array([np.round(wp).astype(np.int32) for wp in waypoints]) if waypoints else np.empty((0, 3), dtype=np.int32)
            
            # Filter close waypoints if enabled
            if self.filter_close_waypoints and len(self.waypoints) > 1:
                original_count = len(self.waypoints)
                self.waypoints = filter_close_waypoints_optimized(self.waypoints, self.min_waypoint_distance)
                if self.verbose and len(self.waypoints) != original_count:
                    print(f"Filtered waypoints from {original_count} to {len(self.waypoints)} (removed close duplicates)")
            
            # Create processing_info for explicit points case
            self.processing_info = {
                'original_points': None,
                'detected_z_range': (min(self.start_point[0], self.goal_point[0]), max(self.start_point[0], self.goal_point[0])),
                'auto_z_detection': False,
                'waypoint_optimization': False,
                'num_waypoints': len(self.waypoints)
            }
        
        # Initialize other components
        self.image_stats = ImageStats(image)
        
        if cost_function == CostFunction.RECIPROCAL:
            self.cost_function = Reciprocal(
                min_intensity=self.image_stats.min_intensity, 
                max_intensity=self.image_stats.max_intensity)
        
        if heuristic_function == HeuristicFunction.EUCLIDEAN:
            self.heuristic_function = Euclidean(scale=self.scale)
        
        # State variables
        self.is_canceled = False
        self.found_path = False
        self.evaluated_nodes = 0
        self.result = []
        self.segment_results = []
        self.segment_evaluated_nodes = []

    def _process_points_list_optimized(self, points_list):
        """Process a list of points to extract start, end, and waypoints with intelligent positioning - OPTIMIZED"""
        if len(points_list) < 2:
            raise ValueError("Need at least 2 points")
        
        if self.verbose:
            print(f"Processing {len(points_list)} input points...")
        
        # Find start and end points (farthest apart) - OPTIMIZED
        waypoint_coords, start_coords, end_coords = find_start_end_points_optimized(points_list)
        
        if self.verbose:
            print(f"Auto-detected start: {start_coords}, end: {end_coords}")
            print(f"Remaining waypoints: {len(waypoint_coords)}")
        
        # Apply z-range detection if enabled
        if self.auto_z_detection and len(self.image.shape) == 3:
            # Find optimal z-ranges for start and end points using transition detection
            start_z_min, start_z_max, start_max_intensity = self.z_range_cache.get_z_range(
                self.image, int(start_coords[1]), int(start_coords[2]), self.intensity_threshold)
            
            end_z_min, end_z_max, end_max_intensity = self.z_range_cache.get_z_range(
                self.image, int(end_coords[1]), int(end_coords[2]), self.intensity_threshold)
            
            if self.verbose:
                print(f"Start point transition frames: appears at {start_z_min}, disappears at {start_z_max} (max intensity: {start_max_intensity:.3f})")
                print(f"End point transition frames: appears at {end_z_min}, disappears at {end_z_max} (max intensity: {end_max_intensity:.3f})")
            
            # Update start and end points with transition-based z-coordinates
            if start_z_min >= 0:
                start_coords[0] = start_z_min  # Use appearance frame for start
                
            if end_z_max >= 0:
                end_coords[0] = end_z_max  # Use disappearance frame for end
            
            # Store the detected z-range for waypoint processing
            detected_z_min = int(start_coords[0])
            detected_z_max = int(end_coords[0])
            
            if self.verbose:
                print(f"Using transition-based coordinates - Start: {start_coords}, End: {end_coords}")
                print(f"Path z-range: {detected_z_min} to {detected_z_max}")
        else:
            detected_z_min = min(int(start_coords[0]), int(end_coords[0]))
            detected_z_max = max(int(start_coords[0]), int(end_coords[0]))
        
        # Process waypoints with intelligent z-positioning - OPTIMIZED
        processed_waypoints = []
        if waypoint_coords and self.waypoint_z_optimization:
            if self.verbose:
                print("Optimizing waypoint z-positions...")
            
            # Convert waypoint coordinates to numpy array for batch processing
            waypoint_coords_array = np.array(waypoint_coords)
            
            # Distribute waypoints across the z-range
            optimal_z_positions = distribute_waypoints_across_z_range_optimized(
                len(waypoint_coords), detected_z_min, detected_z_max)
            
            # Extract Y,X coordinates for batch processing
            waypoint_yx = waypoint_coords_array[:, 1:3].astype(np.int32)
            
            # Batch process all waypoints for optimal z-coordinates
            search_range = 2  # Search ±2 frames around target
            z_min_search = max(detected_z_min, min(optimal_z_positions) - search_range)
            z_max_search = min(detected_z_max, max(optimal_z_positions) + search_range)
            
            optimal_z_values = batch_find_optimal_z_for_waypoints(
                self.image, waypoint_yx, z_min_search, z_max_search)
            
            # Create processed waypoints
            for i, waypoint in enumerate(waypoint_coords):
                processed_waypoint = np.array([optimal_z_values[i], waypoint[1], waypoint[2]], dtype=np.int32)
                processed_waypoints.append(processed_waypoint)
                
                if self.verbose:
                    print(f"Waypoint {i+1}: {waypoint} -> {processed_waypoint}")
        else:
            # Use waypoints as-is
            processed_waypoints = [np.array(wp, dtype=np.int32) for wp in waypoint_coords]
        
        # Store results
        self.start_point = np.array(start_coords, dtype=np.int32)
        self.goal_point = np.array(end_coords, dtype=np.int32)
        self.waypoints = np.array(processed_waypoints) if processed_waypoints else np.empty((0, 3), dtype=np.int32)
        
        # Filter close waypoints if enabled - OPTIMIZED
        if self.filter_close_waypoints and len(self.waypoints) > 1:
            original_count = len(self.waypoints)
            self.waypoints = filter_close_waypoints_optimized(self.waypoints, self.min_waypoint_distance)
            if self.verbose and len(self.waypoints) != original_count:
                print(f"Filtered waypoints from {original_count} to {len(self.waypoints)} (removed close duplicates)")
        
        # Store processing information
        self.processing_info = {
            'original_points': points_list,
            'detected_z_range': (detected_z_min, detected_z_max),
            'auto_z_detection': self.auto_z_detection,
            'waypoint_optimization': self.waypoint_z_optimization,
            'num_waypoints': len(processed_waypoints)
        }

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

    def search(self, verbose: bool = None) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray]]:
        """Perform enhanced A* search with intelligent waypoint processing - OPTIMIZED
        
        Parameters
        ----------
        verbose : bool, optional
            Override the instance verbose setting
            
        Returns
        -------
        tuple
            (path, start_point, goal_point, waypoints) where:
            - path: List[np.ndarray] - Complete path from start to goal through waypoints
            - start_point: np.ndarray - The processed start point [z, y, x]
            - goal_point: np.ndarray - The processed goal point [z, y, x]  
            - waypoints: List[np.ndarray] - The processed waypoint coordinates
        """
        if verbose is None:
            verbose = self.verbose
            
        # Reset state
        self.result = []
        self.segment_results = []
        self.segment_evaluated_nodes = []
        self.evaluated_nodes = 0
        
        if verbose:
            print(f"\nStarting enhanced waypoint A* search:")
            print(f"Start: {self.start_point}")
            print(f"Goal: {self.goal_point}")
            print(f"Waypoints: {len(self.waypoints)}")
            if self.use_hierarchical:
                print("Using hierarchical search for large image")
        
        # Create a list of all points in order
        wp_list = [self.waypoints[i] for i in range(len(self.waypoints))]
        all_points = [self.start_point] + wp_list + [self.goal_point]
        
        # Track overall success
        overall_success = True
        
        # Process each segment with optimized BidirectionalAStarSearch
        for i in range(len(all_points) - 1):
            if self.is_canceled:
                return [], self.start_point, self.goal_point, []
                
            point_a = all_points[i]
            point_b = all_points[i+1]
            
            if verbose:
                segment_type = self._get_segment_type(i, len(all_points), len(wp_list))
                distance = np.linalg.norm(point_b - point_a)
                print(f"Segment {i+1}: {segment_type} (distance: {distance:.1f})")
                print(f"  From: {point_a} to {point_b}")
            
            # Create optimized A* search for this segment using BidirectionalAStarSearch
            segment_search = BidirectionalAStarSearch(
                image=self.image,
                start_point=point_a,
                goal_point=point_b,
                scale=self.scale,
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                open_nodes=self.open_nodes,
                use_hierarchical=self.use_hierarchical,
                weight_heuristic=self.weight_heuristic  # Always 1.0 for optimality
            )
            
            # Run the search for this segment
            segment_path = segment_search.search(verbose=False)
            
            # Track stats and results
            self.segment_evaluated_nodes.append(segment_search.evaluated_nodes)
            self.evaluated_nodes += segment_search.evaluated_nodes
            
            # Check if segment search was successful
            if segment_search.found_path and len(segment_path) > 0:
                self.segment_results.append(segment_path)
                
                if verbose:
                    print(f"  Success: {len(segment_path)} points, {segment_search.evaluated_nodes} nodes evaluated")
            else:
                if verbose:
                    print(f"  Failed to find path for segment {i+1}")
                overall_success = False
                break
        
        # If all segments were successful, combine the paths
        if overall_success:
            self._construct_complete_path()
            self.found_path = True
            
            if verbose:
                print(f"\nComplete path found:")
                print(f"  Total points: {len(self.result)}")
                print(f"  Total nodes evaluated: {self.evaluated_nodes}")
                print(f"  Segments: {len(self.segment_results)}")
                print(f"  Used hierarchical search: {self.use_hierarchical}")
        else:
            if verbose:
                print("Failed to find a complete path through all waypoints")
        
        # Convert waypoints to list for return
        waypoints_list = [self.waypoints[i] for i in range(len(self.waypoints))] if len(self.waypoints) > 0 else []
        
        return self.result, self.start_point, self.goal_point, waypoints_list
    
    def _get_segment_type(self, i, total_points, num_waypoints):
        """Helper to get segment type description"""
        if i == 0 and num_waypoints > 0:
            return "start to waypoint"
        elif i + 1 == total_points - 1 and num_waypoints > 0:
            return "waypoint to goal"
        elif num_waypoints > 0:
            return "waypoint to waypoint"
        else:
            return "start to goal"
    
    def _construct_complete_path(self):
        """Combine segment paths into a complete path, removing duplicate points at boundaries"""
        if not self.segment_results:
            return
            
        # Start with the first segment
        self.result = self.segment_results[0].copy()
        
        # Add each subsequent segment (skipping the first point to avoid duplication)
        for i in range(1, len(self.segment_results)):
            segment = self.segment_results[i]
            
            # Skip the first point of each subsequent segment as it should 
            # be the same as the last point of the previous segment
            if len(segment) > 1 and len(self.result) > 0 and np.array_equal(self.result[-1], segment[0]):
                self.result.extend(segment[1:])
            else:
                # Something's wrong - just append everything
                self.result.extend(segment)
    
    def get_processing_info(self):
        """Get information about the point processing and path finding
        
        Returns
        -------
        dict
            Dictionary with detailed information about the processing
        """
        info = {
            'start_point': self.start_point.tolist(),
            'goal_point': self.goal_point.tolist(),
            'waypoints': self.waypoints.tolist() if len(self.waypoints) > 0 else [],
            'num_segments': len(self.segment_results),
            'segment_lengths': [len(path) for path in self.segment_results],
            'segment_evaluated_nodes': self.segment_evaluated_nodes,
            'total_path_length': len(self.result),
            'total_evaluated_nodes': self.evaluated_nodes,
            'found_path': self.found_path,
            'used_hierarchical_search': self.use_hierarchical,
            'optimizations_enabled': {
                'bidirectional_search': True,
                'parallel_processing': True,
                'waypoint_filtering': self.filter_close_waypoints,
                'z_range_caching': True,
                'auto_hierarchical': True
            }
        }
        
        # Add processing info if available
        if hasattr(self, 'processing_info'):
            info.update(self.processing_info)
        
        return info

    def visualize_points(self, show_original=True, show_processed=True):
        """Create a visualization showing original and processed points
        
        Parameters
        ----------
        show_original : bool
            Whether to show original input points
        show_processed : bool
            Whether to show processed start/end/waypoints
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine which z-slice to show
        if hasattr(self, 'processing_info') and 'detected_z_range' in self.processing_info:
            z_min, z_max = self.processing_info['detected_z_range']
            if z_min == z_max:
                display_image = self.image[z_min]
                title_z = f"Z-slice {z_min}"
            else:
                display_image = self.image[z_min:z_max+1].max(axis=0)
                title_z = f"Max projection Z{z_min}-{z_max}"
        else:
            # Fallback to max projection
            display_image = self.image.max(axis=0)
            title_z = "Max projection"
        
        # Show the image
        ax.imshow(display_image, cmap='gray')
        
        # Show original points if requested
        if show_original and hasattr(self, 'processing_info'):
            original_points = self.processing_info['original_points']
            if original_points:
                x_coords = [p[2] for p in original_points]
                y_coords = [p[1] for p in original_points]
                ax.scatter(x_coords, y_coords, c='lightblue', s=100, marker='o', 
                          label='Original points', alpha=0.7, edgecolors='blue')
        
        # Show processed points if requested
        if show_processed:
            # Start point
            ax.scatter(self.start_point[2], self.start_point[1], c='green', s=150, 
                      marker='o', label=f'Start (Z={self.start_point[0]})', edgecolors='darkgreen')
            
            # End point
            ax.scatter(self.goal_point[2], self.goal_point[1], c='red', s=150, 
                      marker='o', label=f'Goal (Z={self.goal_point[0]})', edgecolors='darkred')
            
            # Waypoints
            if len(self.waypoints) > 0:
                waypoint_x = [wp[2] for wp in self.waypoints]
                waypoint_y = [wp[1] for wp in self.waypoints]
                waypoint_z = [wp[0] for wp in self.waypoints]
                
                for i, (x, y, z) in enumerate(zip(waypoint_x, waypoint_y, waypoint_z)):
                    ax.scatter(x, y, c='yellow', s=120, marker='s', 
                              edgecolors='orange', label=f'Waypoint {i+1} (Z={z})' if i == 0 else "")
        
        ax.set_title(f'Optimized Point Processing Visualization - {title_z}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


# Helper functions for creating optimized search instances
def create_enhanced_waypoint_search(image, points_list=None, start_point=None, goal_point=None, 
                                   waypoints=None, **kwargs):
    """
    Factory function to create an optimized EnhancedWaypointAStarSearch with best performance settings
    
    Parameters
    ----------
    image : numpy.ndarray
        3D image data
    points_list : list, optional
        List of points for automatic start/end/waypoint detection
    start_point : numpy.ndarray, optional
        Explicit start point
    goal_point : numpy.ndarray, optional
        Explicit goal point
    waypoints : list, optional
        Explicit waypoints
    **kwargs
        Additional parameters to pass to EnhancedWaypointAStarSearch
        
    Returns
    -------
    EnhancedWaypointAStarSearch
        Optimized search instance
    """
    # Set optimal default parameters
    default_params = {
        'scale': (1.0, 1.0, 1.0),
        'cost_function': CostFunction.RECIPROCAL,
        'heuristic_function': HeuristicFunction.EUCLIDEAN,
        'weight_heuristic': 1.0,  # Always optimal
        'intensity_threshold': 0.3,
        'auto_z_detection': True,
        'waypoint_z_optimization': True,
        'filter_close_waypoints': True,
        'min_waypoint_distance': 3.0,
        'verbose': False
    }
    
    # Update with user-provided parameters
    default_params.update(kwargs)
    
    return EnhancedWaypointAStarSearch(
        image=image,
        points_list=points_list,
        start_point=start_point,
        goal_point=goal_point,
        waypoints=waypoints,
        **default_params
    )


def quick_enhanced_search(image, points_list, verbose=True):
    """
    Quick function to run enhanced waypoint search with optimal settings
    
    Parameters
    ----------
    image : numpy.ndarray
        3D image data
    points_list : list
        List of points [[z,y,x], ...] for automatic processing
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    tuple
        (path, start_point, goal_point, waypoints, processing_info)
    """
    # Create optimized search
    search = create_enhanced_waypoint_search(
        image=image,
        points_list=points_list,
        verbose=verbose
    )
    
    # Run search
    path, start_point, goal_point, waypoints = search.search(verbose=verbose)
    
    # Get processing info
    processing_info = search.get_processing_info()
    
    if verbose:
        print(f"\nSearch completed successfully: {search.found_path}")
        print(f"Performance stats:")
        print(f"  - Bidirectional search: {processing_info['optimizations_enabled']['bidirectional_search']}")
        print(f"  - Parallel processing: {processing_info['optimizations_enabled']['parallel_processing']}")
        print(f"  - Hierarchical search: {processing_info['used_hierarchical_search']}")
        print(f"  - Waypoint filtering: {processing_info['optimizations_enabled']['waypoint_filtering']}")
        print(f"  - Total nodes evaluated: {processing_info['total_evaluated_nodes']}")
    
    return path, start_point, goal_point, waypoints, processing_info


# Performance benchmarking function
def benchmark_enhanced_search(image, points_list, num_runs=3):
    """
    Benchmark the enhanced waypoint search performance
    
    Parameters
    ----------
    image : numpy.ndarray
        3D image data
    points_list : list
        List of points for testing
    num_runs : int
        Number of runs for averaging
        
    Returns
    -------
    dict
        Benchmark results
    """
    import time
    
    times = []
    node_counts = []
    
    for run in range(num_runs):
        print(f"Benchmark run {run + 1}/{num_runs}")
        
        start_time = time.time()
        
        # Create and run search
        search = create_enhanced_waypoint_search(
            image=image,
            points_list=points_list,
            verbose=False
        )
        
        path, _, _, _ = search.search(verbose=False)
        
        end_time = time.time()
        
        times.append(end_time - start_time)
        node_counts.append(search.evaluated_nodes)
    
    avg_time = np.mean(times)
    avg_nodes = np.mean(node_counts)
    
    results = {
        'average_time_seconds': avg_time,
        'average_nodes_evaluated': avg_nodes,
        'nodes_per_second': avg_nodes / avg_time if avg_time > 0 else 0,
        'all_times': times,
        'all_node_counts': node_counts,
        'image_shape': image.shape,
        'num_points': len(points_list)
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Average time: {avg_time:.2f} seconds")
    print(f"  Average nodes evaluated: {avg_nodes:.0f}")
    print(f"  Processing rate: {results['nodes_per_second']:.0f} nodes/second")
    
    return results