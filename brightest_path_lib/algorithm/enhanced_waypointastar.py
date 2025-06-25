"""Enhanced A* search implementation with intelligent waypoint processing and automatic z-range detection.

This extends the WaypointBidirectionalAStarSearch to support:
1. Automatic z-range detection based on intensity thresholds
2. Intelligent waypoint z-positioning 
3. Automatic start/end point detection from a list of points
4. User-friendly single-frame point selection workflow
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


# Numba-optimized functions for z-range detection
@nb.njit(cache=True)
def find_intensity_transitions_at_point(image, y, x, intensity_threshold=0.3):
    """Find z-frames where intensity transitions occur (appearance/disappearance)"""
    z_size = image.shape[0]
    intensities = np.zeros(z_size)
    
    # Extract intensity profile along z-axis
    for z in range(z_size):
        intensities[z] = image[z, y, x]
    
    # Find peak intensity and threshold
    max_intensity = np.max(intensities)
    threshold = max_intensity * intensity_threshold
    
    # Find the frame where dendrite appears (black to bright transition)
    start_z = -1
    for z in range(1, z_size):  # Start from 1 to compare with previous
        prev_intensity = intensities[z-1]
        curr_intensity = intensities[z]
        
        # Look for transition from below threshold to above threshold
        if prev_intensity < threshold and curr_intensity >= threshold:
            start_z = z
            break
    
    # If no transition found, use first frame above threshold
    if start_z == -1:
        for z in range(z_size):
            if intensities[z] >= threshold:
                start_z = z
                break
    
    # Find the frame where dendrite disappears (bright to black transition)
    end_z = -1
    for z in range(z_size - 2, -1, -1):  # Go backwards, stop at second-to-last
        curr_intensity = intensities[z]
        next_intensity = intensities[z+1]
        
        # Look for transition from above threshold to below threshold
        if curr_intensity >= threshold and next_intensity < threshold:
            end_z = z
            break
    
    # If no transition found, use last frame above threshold
    if end_z == -1:
        for z in range(z_size - 1, -1, -1):
            if intensities[z] >= threshold:
                end_z = z
                break
    
    return start_z, end_z, max_intensity


@nb.njit(cache=True)
def find_optimal_z_for_waypoint(image, y, x, z_min, z_max):
    """Find the z-coordinate where a waypoint has maximum intensity within range"""
    best_z = z_min
    best_intensity = 0.0
    
    for z in range(z_min, z_max + 1):
        if z < image.shape[0]:
            intensity = image[z, y, x]
            if intensity > best_intensity:
                best_intensity = intensity
                best_z = z
    
    return best_z


@nb.njit(cache=True)
def distribute_waypoints_across_z_range(num_waypoints, start_z, end_z):
    """Distribute waypoints evenly across the z-range"""
    if num_waypoints == 0:
        return np.empty(0, dtype=np.int32)
    
    if num_waypoints == 1:
        return np.array([(start_z + end_z) // 2], dtype=np.int32)
    
    # For multiple waypoints, distribute evenly
    z_positions = np.zeros(num_waypoints, dtype=np.int32)
    step = (end_z - start_z) / (num_waypoints + 1)
    
    for i in range(num_waypoints):
        z_positions[i] = int(start_z + (i + 1) * step)
    
    return z_positions


def find_distance(start_point, end_point):
    """Calculate Euclidean distance between two 3D points"""
    return np.sqrt(np.sum((start_point - end_point) ** 2))


def find_start_end_points(points_list):
    """
    Find the two points with maximum distance as start and end points.
    Returns remaining points as waypoints.
    
    Parameters:
    -----------
    points_list : list
        List of points in format [[z, y, x], ...]
    
    Returns:
    --------
    waypoints : list
        Remaining points after removing start and end
    start_point : numpy.ndarray
        Point that will be the start
    end_point : numpy.ndarray
        Point that will be the end
    """
    if len(points_list) < 2:
        raise ValueError("Need at least 2 points to find start and end")
    
    # Create a copy to avoid modifying the original
    points_copy = [list(p) for p in points_list]
    points_array = [np.array(p) for p in points_copy]
    
    max_distance = 0
    start_idx = 0
    end_idx = 1
    
    # Find the two points with maximum distance
    for i in range(len(points_array)):
        for j in range(i + 1, len(points_array)):
            dist = find_distance(points_array[i], points_array[j])
            if dist > max_distance:
                max_distance = dist
                start_idx = i
                end_idx = j
    
    # Extract start and end points
    start_point = np.array(points_copy[start_idx])
    end_point = np.array(points_copy[end_idx])
    
    # Remove start and end points from the list to get waypoints
    # Remove in reverse order to maintain indices
    if end_idx > start_idx:
        points_copy.pop(end_idx)
        points_copy.pop(start_idx)
    else:
        points_copy.pop(start_idx)
        points_copy.pop(end_idx)
    
    waypoints = points_copy
    
    return waypoints, start_point, end_point


class EnhancedWaypointAStarSearch:
    """Enhanced waypoint A* search with intelligent z-range detection and waypoint positioning
    
    This implementation provides:
    1. Automatic z-range detection based on intensity thresholds
    2. Intelligent waypoint z-positioning for optimal path finding
    3. Automatic start/end point detection from point lists
    4. User-friendly single-frame workflow
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
        use_hierarchical: bool = False,
        weight_heuristic: float = 1.0,
        intensity_threshold: float = 0.3,
        auto_z_detection: bool = True,
        waypoint_z_optimization: bool = True,
        verbose: bool = False
    ):
        """Initialize enhanced waypoint A* search
        
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
        use_hierarchical : bool
            Whether to use hierarchical search for large images
        weight_heuristic : float
            Weight for heuristic (> 1.0 makes search faster but less optimal)
        intensity_threshold : float
            Threshold for z-range detection (fraction of peak intensity)
        auto_z_detection : bool
            Whether to automatically detect optimal z-range
        waypoint_z_optimization : bool
            Whether to optimize waypoint z-positions
        verbose : bool
            Whether to print detailed information
        """
        self.image = image
        self.scale = scale
        self.open_nodes = open_nodes
        self.use_hierarchical = use_hierarchical
        self.weight_heuristic = weight_heuristic
        self.intensity_threshold = intensity_threshold
        self.auto_z_detection = auto_z_detection
        self.waypoint_z_optimization = waypoint_z_optimization
        self.verbose = verbose
        
        # Process input points
        if points_list is not None:
            self._process_points_list(points_list)
        else:
            if start_point is None or goal_point is None:
                raise ValueError("Either points_list or both start_point and goal_point must be provided")
            self.start_point = np.round(start_point).astype(np.int32)
            self.goal_point = np.round(goal_point).astype(np.int32)
            self.waypoints = np.array([np.round(wp).astype(np.int32) for wp in waypoints]) if waypoints else np.empty((0, 3), dtype=np.int32)
            
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
        
        # Store processing information
        self.processing_info = {}

    def _process_points_list(self, points_list):
        """Process a list of points to extract start, end, and waypoints with intelligent positioning"""
        if len(points_list) < 2:
            raise ValueError("Need at least 2 points")
        
        if self.verbose:
            print(f"Processing {len(points_list)} input points...")
        
        # Find start and end points (farthest apart)
        waypoint_coords, start_coords, end_coords = find_start_end_points(points_list)
        
        if self.verbose:
            print(f"Auto-detected start: {start_coords}, end: {end_coords}")
            print(f"Remaining waypoints: {len(waypoint_coords)}")
        
        # Apply z-range detection if enabled
        if self.auto_z_detection and len(self.image.shape) == 3:
            # Find optimal z-ranges for start and end points using transition detection
            start_z_min, start_z_max, start_max_intensity = find_intensity_transitions_at_point(
                self.image, start_coords[1], start_coords[2], self.intensity_threshold)
            
            end_z_min, end_z_max, end_max_intensity = find_intensity_transitions_at_point(
                self.image, end_coords[1], end_coords[2], self.intensity_threshold)
            
            if self.verbose:
                print(f"Start point transition frames: appears at {start_z_min}, disappears at {start_z_max} (max intensity: {start_max_intensity:.3f})")
                print(f"End point transition frames: appears at {end_z_min}, disappears at {end_z_max} (max intensity: {end_max_intensity:.3f})")
            
            # Update start and end points with transition-based z-coordinates
            if start_z_min >= 0:
                start_coords[0] = start_z_min  # Use appearance frame for start
                
            if end_z_max >= 0:
                end_coords[0] = end_z_max  # Use disappearance frame for end
            
            # Store the detected z-range for waypoint processing
            detected_z_min = start_coords[0]
            detected_z_max = end_coords[0]
            
            if self.verbose:
                print(f"Using transition-based coordinates - Start: {start_coords}, End: {end_coords}")
                print(f"Path z-range: {detected_z_min} to {detected_z_max}")
        else:
            detected_z_min = min(start_coords[0], end_coords[0])
            detected_z_max = max(start_coords[0], end_coords[0])
        
        # Process waypoints with intelligent z-positioning
        processed_waypoints = []
        if waypoint_coords and self.waypoint_z_optimization:
            if self.verbose:
                print("Optimizing waypoint z-positions...")
            
            # Distribute waypoints across the z-range
            optimal_z_positions = distribute_waypoints_across_z_range(
                len(waypoint_coords), detected_z_min, detected_z_max)
            
            for i, waypoint in enumerate(waypoint_coords):
                # Start with the distributed z-position
                target_z = optimal_z_positions[i]
                
                # Fine-tune based on local intensity maximum
                search_range = 2  # Search ±2 frames around target
                z_min_search = max(detected_z_min, target_z - search_range)
                z_max_search = min(detected_z_max, target_z + search_range)
                
                optimal_z = find_optimal_z_for_waypoint(
                    self.image, waypoint[1], waypoint[2], z_min_search, z_max_search)
                
                processed_waypoint = np.array([optimal_z, waypoint[1], waypoint[2]], dtype=np.int32)
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
        """Perform enhanced A* search with intelligent waypoint processing
        
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
        
        # Create a list of all points in order
        wp_list = [self.waypoints[i] for i in range(len(self.waypoints))]
        all_points = [self.start_point] + wp_list + [self.goal_point]
        
        # Track overall success
        overall_success = True
        
        # Process each segment
        for i in range(len(all_points) - 1):
            if self.is_canceled:
                return []
                
            point_a = all_points[i]
            point_b = all_points[i+1]
            
            if verbose:
                segment_type = "start to waypoint" if i == 0 and len(wp_list) > 0 else \
                             "waypoint to goal" if i+1 == len(all_points)-1 and len(wp_list) > 0 else \
                             "waypoint to waypoint" if len(wp_list) > 0 else \
                             "start to goal"
                print(f"Segment {i+1}: {segment_type}")
                print(f"  From: {point_a} to {point_b}")
            
            # Create A* search for this segment
            segment_search = BidirectionalAStarSearch(
                image=self.image,
                start_point=point_a,
                goal_point=point_b,
                scale=self.scale,
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                open_nodes=self.open_nodes,
                use_hierarchical=self.use_hierarchical,
                weight_heuristic=self.weight_heuristic
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
        else:
            if verbose:
                print("Failed to find a complete path through all waypoints")
        
        # Convert waypoints to list for return
        waypoints_list = [self.waypoints[i] for i in range(len(self.waypoints))] if len(self.waypoints) > 0 else []
        
        return self.result, self.start_point, self.goal_point, waypoints_list
    
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
            'found_path': self.found_path
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
        
        ax.set_title(f'Point Processing Visualization - {title_z}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig