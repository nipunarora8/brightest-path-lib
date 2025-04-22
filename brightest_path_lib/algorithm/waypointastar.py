"""Advanced optimized A* search implementation with waypoint support for finding brightest paths.
This extends the BidirectionalAStarSearch to support user-defined auxiliary waypoints.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from collections import deque

from brightest_path_lib.algorithm.astar import BidirectionalAStarSearch
from brightest_path_lib.input import CostFunction, HeuristicFunction
from brightest_path_lib.node import Node

class WaypointBidirectionalAStarSearch:
    """A* search implementation that supports auxiliary waypoints
    
    This implementation allows users to specify intermediate points that
    the path should pass through, breaking down a complex search into
    multiple simpler searches between consecutive points.
    """

    def __init__(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        waypoints: List[np.ndarray] = None,
        scale: Tuple = (1.0, 1.0),
        cost_function: CostFunction = CostFunction.RECIPROCAL,
        heuristic_function: HeuristicFunction = HeuristicFunction.EUCLIDEAN,
        open_nodes=None,
        use_hierarchical: bool = False,
        weight_heuristic: float = 1.0
    ):
        """Initialize waypoint-enabled A* search
        
        Parameters
        ----------
        image : numpy ndarray
            The image to search
        start_point, goal_point : numpy ndarray
            Start and goal coordinates
        waypoints : List[numpy ndarray], optional
            List of intermediate points that the path must pass through,
            in the order they should be visited
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
        """
        self._validate_inputs(image, start_point, goal_point, waypoints)
        
        self.image = image
        self.start_point = np.round(start_point).astype(np.int32)
        self.goal_point = np.round(goal_point).astype(np.int32)
        self.waypoints = []
        
        # Process and validate waypoints if provided
        if waypoints and len(waypoints) > 0:
            self.waypoints = [np.round(wp).astype(np.int32) for wp in waypoints]
        
        self.scale = scale
        self.cost_function = cost_function
        self.heuristic_function = heuristic_function
        self.open_nodes = open_nodes
        self.use_hierarchical = use_hierarchical
        self.weight_heuristic = weight_heuristic
        
        # State variables
        self.is_canceled = False
        self.found_path = False
        self.evaluated_nodes = 0
        self.result = []
        self.segment_results = []  # Store individual segment paths
        self.segment_evaluated_nodes = []  # Track nodes evaluated per segment
        
    def _validate_inputs(
        self,
        image: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        waypoints: Optional[List[np.ndarray]] = None
    ):
        """Validate input parameters"""
        if image is None or start_point is None or goal_point is None:
            raise TypeError("Image, start_point, and goal_point cannot be None")
        if len(image) == 0 or len(start_point) == 0 or len(goal_point) == 0:
            raise ValueError("Image, start_point, and goal_point cannot be empty")
            
        # Validate waypoints if provided
        if waypoints:
            for i, wp in enumerate(waypoints):
                if wp is None:
                    raise TypeError(f"Waypoint {i} cannot be None")
                if len(wp) == 0:
                    raise ValueError(f"Waypoint {i} cannot be empty")
                if len(wp) != len(start_point):
                    raise ValueError(f"Waypoint {i} dimensions must match start_point dimensions")

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
        """Perform A* search with waypoints
        
        This method breaks down the search into multiple segments:
        start→waypoint₁, waypoint₁→waypoint₂, ..., waypointₙ→goal
        
        Returns
        -------
        List[np.ndarray]
            Complete path from start to goal through all waypoints
        """
        # Reset state
        self.result = []
        self.segment_results = []
        self.segment_evaluated_nodes = []
        self.evaluated_nodes = 0
        
        # Create a list of all points in order, including start and goal
        all_points = [self.start_point] + self.waypoints + [self.goal_point]
        
        # Track overall success
        overall_success = True
        
        if verbose:
            if self.waypoints:
                print(f"Starting search with {len(self.waypoints)} waypoints")
            else:
                print("Starting search (no waypoints)")
        
        # Process each segment (start to first waypoint, waypoint to waypoint, last waypoint to goal)
        for i in range(len(all_points) - 1):
            if self.is_canceled:
                return []
                
            point_a = all_points[i]
            point_b = all_points[i+1]
            
            if verbose:
                if i == 0:
                    print(f"Searching from start to {'waypoint' if i+1 < len(all_points)-1 else 'goal'} {i+1}")
                elif i+1 == len(all_points)-1:
                    print(f"Searching from waypoint {i} to goal")
                else:
                    print(f"Searching from waypoint {i} to waypoint {i+1}")
            
            # Create A* search for this segment
            segment_search = BidirectionalAStarSearch(
                image=self.image,
                start_point=point_a,
                goal_point=point_b,
                scale=self.scale,
                cost_function=self.cost_function,
                heuristic_function=self.heuristic_function,
                open_nodes=self.open_nodes,
                use_hierarchical=self.use_hierarchical,
                weight_heuristic=self.weight_heuristic
            )
            
            # Run the search for this segment
            segment_path = segment_search.search(verbose=verbose)
            
            # Track stats and results
            self.segment_evaluated_nodes.append(segment_search.evaluated_nodes)
            self.evaluated_nodes += segment_search.evaluated_nodes
            
            # Check if segment search was successful
            if segment_search.found_path and len(segment_path) > 0:
                self.segment_results.append(segment_path)
                
                if verbose:
                    print(f"Found path for segment {i+1} ({len(segment_path)} points, {segment_search.evaluated_nodes} nodes evaluated)")
            else:
                if verbose:
                    print(f"Failed to find path for segment {i+1}")
                overall_success = False
                break
        
        # If all segments were successful, combine the paths
        if overall_success:
            self._construct_complete_path()
            self.found_path = True
            
            if verbose:
                print(f"Complete path found: {len(self.result)} points, {self.evaluated_nodes} total nodes evaluated")
        else:
            if verbose:
                print("Failed to find a complete path through all waypoints")
        
        return self.result
    
    def _construct_complete_path(self):
        """Combine segment paths into a complete path
        
        This handles removing duplicate points at segment boundaries.
        """
        if not self.segment_results:
            return
            
        # Start with the first segment
        self.result = self.segment_results[0].copy()
        
        # Add each subsequent segment (skipping the first point to avoid duplication)
        for i in range(1, len(self.segment_results)):
            segment = self.segment_results[i]
            
            # Skip the first point of each subsequent segment as it should 
            # be the same as the last point of the previous segment
            # But verify they're actually the same
            if len(segment) > 1 and np.array_equal(self.result[-1], segment[0]):
                self.result.extend(segment[1:])
            else:
                # Something's wrong - just append everything
                self.result.extend(segment)
    
    def get_segment_info(self):
        """Get information about each segment of the path
        
        Returns
        -------
        dict
            Dictionary with information about each path segment
        """
        if not self.segment_results:
            return None
            
        info = {
            'num_segments': len(self.segment_results),
            'segment_lengths': [len(path) for path in self.segment_results],
            'segment_evaluated_nodes': self.segment_evaluated_nodes,
            'total_path_length': len(self.result),
            'total_evaluated_nodes': self.evaluated_nodes
        }
        return info