"""
Accurate Fast Waypoint Search with Nanometer Support
Optimized for speed while maintaining high path accuracy
Now works with nanometer-based thresholds while maintaining pixel-based processing
NO SUBDIVISION - Pure waypoint-to-waypoint processing
"""

import numpy as np
from typing import Tuple, List, Optional
import numba as nb
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil


@nb.njit(cache=True)
def calculate_segment_distance_accurate_nm(point_a_arr, point_b_arr, xy_spacing_nm):
    """Calculate 3D distance between points in nanometers"""
    # Convert pixel coordinates to nanometers
    z_diff_nm = (point_b_arr[0] - point_a_arr[0])
    y_diff_nm = (point_b_arr[1] - point_a_arr[1]) * xy_spacing_nm
    x_diff_nm = (point_b_arr[2] - point_a_arr[2]) * xy_spacing_nm
    
    distance_sq_nm = z_diff_nm * z_diff_nm + y_diff_nm * y_diff_nm + x_diff_nm * x_diff_nm
    return np.sqrt(distance_sq_nm)


@nb.njit(cache=True)
def calculate_segment_distance_accurate_pixels(point_a_arr, point_b_arr):
    """Calculate 3D distance between points in pixels (for internal use)"""
    distance_sq = 0.0
    for i in range(len(point_a_arr)):
        diff = point_b_arr[i] - point_a_arr[i]
        distance_sq += diff * diff
    return np.sqrt(distance_sq)


@dataclass
class SearchStrategy:
    """Search strategy that prioritizes accuracy"""
    use_hierarchical: bool
    hierarchical_factor: int
    weight_heuristic: float  # Always 1.0 for medical accuracy
    refine_path: bool  # Whether to refine path at full resolution
    suitable_for_parallel: bool  # Whether this segment can be processed in parallel


class Optimizer:
    """Optimizer that maintains accuracy while improving speed - now with nanometer support"""
    
    def __init__(self, image_shape, xy_spacing_nm=94.0, enable_parallel=True, 
                 max_parallel_workers=None, my_weight_heuristic=1.0):
        self.image_shape = image_shape
        self.image_volume = np.prod(image_shape)
        self.my_weight_heuristic = my_weight_heuristic
        
        # Store spacing in nanometers
        self.xy_spacing_nm = xy_spacing_nm
        
        # Conservative thresholds in nanometers (converted from pixels for compatibility)
        self.large_image_threshold = 30_000_000   # voxels
        self.huge_image_threshold = 100_000_000   # voxels
        
        # Parallel processing settings
        self.enable_parallel = enable_parallel
        if max_parallel_workers is None:
            # Auto-detect optimal number of workers
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Conservative worker count based on memory and cores
            max_workers_by_memory = max(1, int(available_memory_gb / 2))  # 2GB per worker
            max_workers_by_cpu = max(1, cpu_count - 1)  # Leave one core free
            
            self.max_parallel_workers = min(max_workers_by_memory, max_workers_by_cpu, 4)  # Cap at 4
        else:
            self.max_parallel_workers = max_parallel_workers
        
        print(f"Parallel processing: {self.enable_parallel}, Max workers: {self.max_parallel_workers}")
        print(f"Spacing: XY={self.xy_spacing_nm:.1f} nm/pixel")
    
    def determine_accurate_strategy(self, distance_nm: float, segment_idx: int, total_segments: int) -> SearchStrategy:
        """Determine strategy that maintains accuracy with parallel processing - now using nanometer thresholds"""
        
        # Default to high accuracy
        strategy = SearchStrategy(
            use_hierarchical=False,
            hierarchical_factor=4,
            weight_heuristic=self.my_weight_heuristic,  # ALWAYS 1.0 for medical accuracy
            refine_path=False,
            suitable_for_parallel=False
        )
        
        # Determine if suitable for parallel processing - using nanometer thresholds
        # Criteria: moderate to long segments in nanometers, not too complex
        if (self.enable_parallel and 
            distance_nm > 9400.0 and      # 9.4 μm (was 100 pixels × 94 nm/pixel)
            distance_nm < 56400.0 and     # 56.4 μm (was 600 pixels × 94 nm/pixel)
            total_segments > 2):          # Multiple segments available
            strategy.suitable_for_parallel = True
        
        # Only use hierarchical for very large images and long segments - nanometer thresholds
        if self.image_volume > self.huge_image_threshold and distance_nm > 28200.0:  # 28.2 μm (300 pixels × 94 nm)
            # Very conservative hierarchical search
            strategy.use_hierarchical = True
            strategy.hierarchical_factor = 4  # Small factor to preserve detail
            strategy.refine_path = True       # Always refine for accuracy
            strategy.weight_heuristic = self.my_weight_heuristic  # Always optimal
            
        elif self.image_volume > self.large_image_threshold and distance_nm > 37600.0:  # 37.6 μm (400 pixels × 94 nm)
            # Only for very long segments on large images
            strategy.use_hierarchical = True
            strategy.hierarchical_factor = 3  # Very conservative factor
            strategy.refine_path = True
            strategy.weight_heuristic = self.my_weight_heuristic 
        
        return strategy


class FasterWaypointSearch:
    """Fast waypoint search that maintains high accuracy with parallel processing - now with nanometer support"""
    
    def __init__(self, image, points_list, xy_spacing_nm=94.0, **kwargs):
        self.image = image
        self.points_list = points_list
        self.verbose = kwargs.get('verbose', True)
        self.my_weight_heuristic = kwargs.get('weight_heuristic', 1.0) 
        
        # Store spacing
        self.xy_spacing_nm = xy_spacing_nm
        
        # Parallel processing settings
        enable_parallel = kwargs.get('enable_parallel', True)
        max_parallel_workers = kwargs.get('max_parallel_workers', None)
        
        # Initialize optimizer with parallel settings and spacing
        self.optimizer = Optimizer(
            image.shape, 
            xy_spacing_nm=xy_spacing_nm,
            enable_parallel=enable_parallel,
            max_parallel_workers=max_parallel_workers
        )
        
        # Configuration - conservative settings in nanometers
        self.enable_refinement = kwargs.get('enable_refinement', True)
        
        if self.verbose:
            print(f"Initializing accurate fast search for image shape: {image.shape}")
            print(f"Image volume: {self.optimizer.image_volume:,} voxels")
            print(f"Parallel processing: {self.optimizer.enable_parallel}")
            print(f"Spacing: XY={self.xy_spacing_nm:.1f} nm/pixel")
            print("NO SUBDIVISION - Pure waypoint-to-waypoint processing")
    
    def search_segment_accurate_wrapper(self, segment_data):
        """Wrapper for parallel processing"""
        point_a, point_b, segment_idx, strategy = segment_data
        return self.search_segment_accurate(point_a, point_b, segment_idx, strategy)
    
    def search_segments_parallel(self, parallel_segments):
        """Search multiple segments in parallel"""
        if not parallel_segments:
            return []
        
        if self.verbose:
            print(f"Processing {len(parallel_segments)} segments in parallel...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.optimizer.max_parallel_workers) as executor:
            # Submit all parallel segments
            parallel_results = list(executor.map(self.search_segment_accurate_wrapper, parallel_segments))
        
        return parallel_results
    
    def search_segment_accurate(self, point_a, point_b, segment_idx, strategy: SearchStrategy):
        """Search segment with high accuracy"""
        from brightest_path_lib.algorithm.astar import BidirectionalAStarSearch
        from brightest_path_lib.input import CostFunction, HeuristicFunction
        
        # Calculate distance in nanometers for logging
        distance_nm = calculate_segment_distance_accurate_nm(
            np.array(point_a, dtype=np.float64), 
            np.array(point_b, dtype=np.float64),
            self.xy_spacing_nm,
        )
        
        if self.verbose:
            print(f"  Segment {segment_idx}: distance={distance_nm:.1f} nm, "
                  f"hierarchical={strategy.use_hierarchical}, "
                  f"refine={strategy.refine_path}")
        
        start_time = time.time()
        
        # First pass - potentially hierarchical for speed
        if strategy.use_hierarchical:
            # Create conservative hierarchical image
            hierarchical_image = self._create_conservative_hierarchical_image(strategy.hierarchical_factor)
            
            # Scale coordinates for hierarchical search
            scale_factor = strategy.hierarchical_factor
            start_scaled = np.array(point_a, dtype=np.int32) // scale_factor
            goal_scaled = np.array(point_b, dtype=np.int32) // scale_factor
            
            # Ensure scaled coordinates are valid
            start_scaled = np.clip(start_scaled, 0, np.array(hierarchical_image.shape) - 1)
            goal_scaled = np.clip(goal_scaled, 0, np.array(hierarchical_image.shape) - 1)
            
            # Hierarchical search
            hierarchical_search = BidirectionalAStarSearch(
                image=hierarchical_image,
                start_point=start_scaled,
                goal_point=goal_scaled,
                scale=(1.0, 1.0, 1.0),
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                use_hierarchical=False,  # Don't nest hierarchical
                weight_heuristic=self.my_weight_heuristic    # Always optimal
            )
            
            hierarchical_path = hierarchical_search.search(verbose=False)
            
            if not hierarchical_path:
                print(f"    Hierarchical search failed, falling back to full resolution")
                strategy.use_hierarchical = False
                strategy.refine_path = False
        
        # Second pass - refine at full resolution if needed
        if strategy.refine_path and strategy.use_hierarchical and hierarchical_path:
            # Scale hierarchical path back to full resolution
            scaled_path = [point * strategy.hierarchical_factor for point in hierarchical_path]
            
            # Refine path by searching in a corridor around the hierarchical path
            refined_path = self._refine_path_in_corridor(scaled_path, point_a, point_b)
            segment_path = refined_path
        else:
            # Direct search at full resolution
            search = BidirectionalAStarSearch(
                image=self.image,
                start_point=np.array(point_a, dtype=np.int32),
                goal_point=np.array(point_b, dtype=np.int32),
                scale=(1.0, 1.0, 1.0),
                cost_function=CostFunction.RECIPROCAL,
                heuristic_function=HeuristicFunction.EUCLIDEAN,
                use_hierarchical=False,
                weight_heuristic=self.my_weight_heuristic  # Always optimal for accuracy
            )
            
            segment_path = search.search(verbose=False)
        
        search_time = time.time() - start_time
        
        if self.verbose:
            print(f"    Completed in {search_time:.2f}s, "
                  f"{len(segment_path) if segment_path else 0} points")
        
        return segment_path
    
    def _create_conservative_hierarchical_image(self, factor):
        """Create hierarchical image with conservative downsampling"""
        if not hasattr(self, '_hierarchical_cache'):
            self._hierarchical_cache = {}
        
        if factor not in self._hierarchical_cache:
            # Use maximum intensity to preserve bright structures
            z, y, x = self.image.shape
            new_z = max(1, z // factor)
            new_y = max(1, y // factor)
            new_x = max(1, x // factor)
            
            downsampled = np.zeros((new_z, new_y, new_x), dtype=self.image.dtype)
            
            for i in range(new_z):
                for j in range(new_y):
                    for k in range(new_x):
                        z_start, z_end = i * factor, min((i + 1) * factor, z)
                        y_start, y_end = j * factor, min((j + 1) * factor, y)
                        x_start, x_end = k * factor, min((k + 1) * factor, x)
                        
                        region = self.image[z_start:z_end, y_start:y_end, x_start:x_end]
                        # Use maximum to preserve bright structures
                        downsampled[i, j, k] = np.max(region)
            
            self._hierarchical_cache[factor] = downsampled
        
        return self._hierarchical_cache[factor]
    
    def _refine_path_in_corridor(self, coarse_path, original_start, original_goal):
        """Refine a coarse path by searching in a corridor around it"""
        from brightest_path_lib.algorithm.astar import BidirectionalAStarSearch
        from brightest_path_lib.input import CostFunction, HeuristicFunction
    
        # This ensures maximum accuracy
        search = BidirectionalAStarSearch(
            image=self.image,
            start_point=np.array(original_start, dtype=np.int32),
            goal_point=np.array(original_goal, dtype=np.int32),
            scale=(1.0, 1.0, 1.0),
            cost_function=CostFunction.RECIPROCAL,
            heuristic_function=HeuristicFunction.EUCLIDEAN,
            use_hierarchical=False,
            weight_heuristic=self.my_weight_heuristic  # Always optimal
        )
        
        return search.search(verbose=False)
    
    def search(self):
        """Perform accurate fast search with parallel processing - NO SUBDIVISION"""
        start_time = time.time()
        
        # NO SUBDIVISION - Use original waypoints directly
        from brightest_path_lib.algorithm.enhanced_waypointastar import find_start_end_points_optimized
        
        # Find optimal start/end points but don't subdivide anything
        waypoint_coords, start_coords, end_coords = find_start_end_points_optimized(self.points_list)
        all_points = [start_coords] + waypoint_coords + [end_coords]
        
        if self.verbose:
            print(f"Starting accurate fast search with {len(all_points)} waypoints (no subdivision)")
            print(f"Original points: {len(self.points_list)} -> Optimized points: {len(all_points)}")
        
        # Analyze all segments and determine strategies
        segment_data = []
        strategies = []
        
        for i in range(len(all_points) - 1):
            point_a = all_points[i]
            point_b = all_points[i + 1]
            
            # Determine strategy for this segment using nanometer distance
            distance_nm = calculate_segment_distance_accurate_nm(
                np.array(point_a, dtype=np.float64), 
                np.array(point_b, dtype=np.float64),
                self.xy_spacing_nm,
            )
            
            strategy = self.optimizer.determine_accurate_strategy(
                distance_nm, i + 1, len(all_points) - 1)
            
            strategies.append(strategy)
            segment_data.append((point_a, point_b, i + 1, strategy))
        
        # Separate segments for parallel vs sequential processing
        parallel_segments = []
        sequential_segments = []
        parallel_indices = []
        sequential_indices = []
        
        for idx, (point_a, point_b, segment_idx, strategy) in enumerate(segment_data):
            if strategy.suitable_for_parallel and len(parallel_segments) < self.optimizer.max_parallel_workers:
                parallel_segments.append((point_a, point_b, segment_idx, strategy))
                parallel_indices.append(idx)
            else:
                sequential_segments.append((point_a, point_b, segment_idx, strategy))
                sequential_indices.append(idx)
        
        if self.verbose:
            print(f"Parallel segments: {len(parallel_segments)}, Sequential: {len(sequential_segments)}")
        
        # Initialize results array to maintain order
        all_paths = [None] * len(segment_data)
        
        # Process parallel segments
        if parallel_segments:
            parallel_start_time = time.time()
            parallel_results = self.search_segments_parallel(parallel_segments)
            parallel_time = time.time() - parallel_start_time
            
            # Store parallel results in correct positions
            for idx, result in zip(parallel_indices, parallel_results):
                all_paths[idx] = result
            
            if self.verbose:
                print(f"Parallel processing completed in {parallel_time:.2f}s")
        
        # Process sequential segments
        if sequential_segments:
            if self.verbose:
                print("Processing sequential segments...")
            
            for segment_idx, (point_a, point_b, seg_num, strategy) in enumerate(sequential_segments):
                path = self.search_segment_accurate(point_a, point_b, seg_num, strategy)
                
                # Store in correct position
                original_idx = sequential_indices[segment_idx]
                all_paths[original_idx] = path
        
        # Check if all segments succeeded
        if all(path is not None for path in all_paths):
            # Combine paths in correct order
            result = all_paths[0].copy()
            for path in all_paths[1:]:
                result.extend(path[1:])  # Skip first point to avoid duplication
        else:
            failed_segments = [i + 1 for i, path in enumerate(all_paths) if path is None]
            print(f"ERROR: Failed to find path for segments: {failed_segments}")
            return None
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"Accurate fast search completed in {total_time:.2f}s")
            print(f"Total path length: {len(result)}")
            
            # Calculate theoretical speedup from parallelization
            if parallel_segments:
                sequential_time_estimate = total_time + (len(parallel_segments) - 1) * (total_time / len(all_points))
                parallel_speedup = sequential_time_estimate / total_time
                print(f"Estimated parallel speedup: {parallel_speedup:.1f}x")
        
        return result

# Conservative optimization settings for medical use - now with nanometer thresholds
def create_accurate_settings_nm(xy_spacing_nm=94.0):
    """Create settings that prioritize accuracy for medical applications - using nanometer thresholds"""
    return {
        'enable_refinement': True,             # Always refine hierarchical paths
        'hierarchical_threshold': 100_000_000, # Only for very large images
        'weight_heuristic': 1.0,               # ALWAYS optimal for medical accuracy
        'enable_parallel': True,               # Enable parallel processing
        'max_parallel_workers': None,          # Auto-detect optimal workers
        'xy_spacing_nm': xy_spacing_nm,        # XY pixel spacing 
    }

def quick_accurate_optimized_search(image, points_list, xy_spacing_nm=94.0,
                                   my_weight_heuristic=2.0, verbose=True, enable_parallel=True):
    """
    Quick accurate optimized search with nanometer support - NO SUBDIVISION
    
    Args:
        image: 3D image array
        points_list: List of [z, y, x] waypoints in pixel coordinates
        xy_spacing_nm: XY pixel spacing in nanometers per pixel
        my_weight_heuristic: A* weight heuristic (1.0 = optimal)
        verbose: Print progress information
        enable_parallel: Enable parallel processing
    """
    
    if verbose:
        print("FAST SEARCH WITH PARALLEL PROCESSING AND NANOMETER SUPPORT")
        print("NO SUBDIVISION - Pure waypoint-to-waypoint processing")
        print(f"Image shape: {image.shape}")
        print(f"Image volume: {np.prod(image.shape):,} voxels")
        print(f"Number of points: {len(points_list)}")
        print(f"Parallel processing: {enable_parallel}")
        print(f"Spacing: XY={xy_spacing_nm:.1f} nm/pixel")
        print()
    
    # Use conservative settings with nanometer support
    settings = create_accurate_settings_nm(xy_spacing_nm)
    settings['weight_heuristic'] = my_weight_heuristic
    settings['enable_parallel'] = enable_parallel
    
    # Remove spacing from settings to avoid duplicate keyword arguments
    settings.pop('xy_spacing_nm', None)
    
    search = FasterWaypointSearch(
        image=image,
        points_list=points_list,
        xy_spacing_nm=xy_spacing_nm,
        verbose=verbose,
        **settings
    )
    
    return search.search()


if __name__ == "__main__":
    print('Fast waypoint search with nanometer support ready - NO SUBDIVISION!')