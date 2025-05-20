from brightest_path_lib.visualization import create_tube_data


def detect_spines_with_watershed(tube_data, shaft_radius=10, min_spine_intensity=1.5, min_spine_area=5, distance_threshold = 10,
                             min_spine_score=0.5, max_spine_distance=None, filter_dendrites=True):
    """
    Detect spines using watershed segmentation with cross-checking in 2D zoomed view
    to verify if detected spines are connected to the current dendrite shaft.
    
    Parameters:
    -----------
    tube_data : list of dict
        Output from create_tube_data function
    shaft_radius : int
        Approximate radius of the dendrite shaft
    min_spine_intensity : float
        Minimum intensity multiplier over threshold for spine detection
    min_spine_area : int
        Minimum area (in pixels) for a region to be considered a spine
    min_spine_score : float
        Minimum score for a spine to be included in results
    max_spine_distance : float or None
        Maximum distance from dendrite center for spine detection (None = no limit)
    filter_dendrites : bool
        Whether to apply parallel dendrite filtering
        
    Returns:
    --------
    spine_positions : list
        List of (z, y, x) coordinates for detected spine centroids
    dendrite_path : numpy.ndarray
        Array of (z, y, x) coordinates for the dendrite centerline
    frame_data : list
        List of dictionaries containing detection results for each frame
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    from skimage import filters, segmentation, morphology, measure, feature
    
    # Extract dendrite path
    dendrite_path = np.array([frame['current_point_in_path'] for frame in tube_data])
    
    # Create visualization figure
    num_frames = min(10, len(tube_data))
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    if num_frames == 1:
        axes = [axes]  # Make it iterable
    frame_indices = np.linspace(0, len(tube_data)-1, num_frames, dtype=int)
    
    # Initialize spine candidates and frame data
    spine_candidates = []
    frame_data = []
    
    # Process each frame
    for frame_idx, frame in enumerate(tube_data):
        # Get the tubular view
        normal_plane = frame['normal_plane']
        
        # Get current z-slice at this frame for cross-checking
        current_position = frame['current_point_in_path']
        z_slice_idx = int(round(current_position[0]))
        
        # Skip frames with low overall intensity (likely no dendrite)
        if np.max(normal_plane) < 0.1:
            frame_result = {
                'frame_idx': frame_idx,
                'position': current_position,
                'normal_plane': normal_plane,
                'dendrite_binary': np.zeros_like(normal_plane, dtype=bool),
                'spine_regions': np.zeros_like(normal_plane, dtype=bool),
                'segments': np.zeros_like(normal_plane, dtype=int),
                'spines': [],
                'basis_vectors': frame['basis_vectors'],
                'is_parallel_dendrite': []
            }
            frame_data.append(frame_result)
            continue
        
        # Get basis vectors
        up = frame['basis_vectors']['up']
        right = frame['basis_vectors']['right']
        forward = frame['basis_vectors']['forward']
        
        # Find center of the plane
        center_y, center_x = normal_plane.shape[0] // 2, normal_plane.shape[1] // 2
        
        # Apply Gaussian smoothing to reduce noise
        smoothed = filters.gaussian(normal_plane, sigma=1.5)
        
        # Create distance transform from center
        y_coords, x_coords = np.ogrid[:normal_plane.shape[0], :normal_plane.shape[1]]
        dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Limit max distance if specified
        if max_spine_distance is not None:
            mask_too_far = dist_from_center > max_spine_distance
            smoothed[mask_too_far] = 0
        
        # Create initial markers for watershed
        markers = np.zeros_like(smoothed, dtype=int)
        
        # Mark the shaft region (center)
        markers[dist_from_center < shaft_radius] = 1
        
        # Create a mask of the entire dendrite
        try:
            thresh_value = filters.threshold_otsu(smoothed)
        except:
            thresh_value = 0.1
            
        # More robust thresholding with hysteresis
        dendrite_binary = filters.apply_hysteresis_threshold(
            smoothed, 
            low=thresh_value * 0.5,
            high=thresh_value * 0.75
        )
        
        # Skip this frame if no dendrite is detected
        if np.sum(dendrite_binary) < 10:
            frame_result = {
                'frame_idx': frame_idx,
                'position': current_position,
                'normal_plane': normal_plane,
                'dendrite_binary': dendrite_binary,
                'spine_regions': np.zeros_like(normal_plane, dtype=bool),
                'segments': np.zeros_like(normal_plane, dtype=int),
                'spines': [],
                'basis_vectors': {
                    'up': up,
                    'right': right,
                    'forward': forward
                },
                'is_parallel_dendrite': []
            }
            frame_data.append(frame_result)
            continue
        
        # Clean up the binary mask
        dendrite_binary = morphology.remove_small_objects(dendrite_binary, min_size=min_spine_area)
        dendrite_binary = morphology.binary_closing(dendrite_binary, morphology.disk(2))
        
        # Connected component analysis to ensure we only have the main dendrite
        labeled_dendrite = measure.label(dendrite_binary)
        
        # Find the component that contains the center (main dendrite)
        if labeled_dendrite[center_y, center_x] > 0:
            main_label = labeled_dendrite[center_y, center_x]
            dendrite_binary = labeled_dendrite == main_label
        else:
            # If center doesn't have a label, find closest component
            props = measure.regionprops(labeled_dendrite)
            if props:
                min_dist = float('inf')
                closest_label = 0
                for prop in props:
                    y, x = prop.centroid
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_label = prop.label
                
                dendrite_binary = labeled_dendrite == closest_label
        
        # Mark strong spine candidates
        outer_mask = dendrite_binary & (dist_from_center > shaft_radius * 1.2)
        
        # Mark high-intensity areas in the outer region as potential spines
        spine_candidates_binary = (smoothed > thresh_value * min_spine_intensity) & outer_mask
        
        # Apply minimum size filter to candidate regions to eliminate noise spots
        spine_candidates_binary = morphology.remove_small_objects(spine_candidates_binary, min_size=min_spine_area)
        
        markers[spine_candidates_binary] = 2
        
        # Apply watershed
        elevation_map = -smoothed  # Use negative of intensity as elevation map
        segments = segmentation.watershed(elevation_map, markers, mask=dendrite_binary)
        
        # Spine regions are labeled as 2
        spine_regions = segments == 2
        
        # For visualization
        if frame_idx in frame_indices:
            ax_idx = np.where(frame_indices == frame_idx)[0][0]
            ax = axes[ax_idx]
            
            # Display the original image
            ax.imshow(normal_plane, cmap='gray')
            
            # Mark center point
            ax.plot(center_x, center_y, 'b+', markersize=10)
            
            # Show shaft boundary
            circle = plt.Circle((center_x, center_y), shaft_radius, 
                              color='blue', fill=False, linewidth=1)
            ax.add_patch(circle)
            
            # Overlay spine regions in red
            colored_overlay = np.zeros((*normal_plane.shape, 4))  # RGBA
            colored_overlay[..., 0] = 1  # Red channel
            colored_overlay[..., 3] = 0  # Alpha channel (transparent)
            colored_overlay[spine_regions, 3] = 0.5  # Make spine regions semi-transparent
            
            ax.imshow(colored_overlay)
        
        # Label and measure spine regions
        labeled_spines = measure.label(spine_regions)
        spine_props = measure.regionprops(labeled_spines, intensity_image=smoothed)
        
        # Calculate frame's median and max intensity for adaptive scoring
        med_intensity = np.median(smoothed[dendrite_binary])
        max_intensity = np.max(smoothed)
        
        # Store frame-specific results
        frame_spines = []
        is_parallel_dendrite = []  # Will store indices of spines classified as parallel dendrites
        
        # Get the zoomed patch data from the frame (if available)
        zoom_patch = frame.get('zoom_patch', None)
        zoom_coordinates = frame.get('zoom_coordinates', None)
        
        for prop_idx, prop in enumerate(spine_props):
            # Filter by minimum area
            if prop.area < min_spine_area:
                continue
                
            # Calculate center of mass (weighted by intensity)
            y_centroid, x_centroid = prop.weighted_centroid if hasattr(prop, 'weighted_centroid') else prop.centroid
            
            # Calculate 3D position
            dx_spine = x_centroid - center_x
            dy_spine = y_centroid - center_y
            spine_position = current_position + dx_spine * right + dy_spine * up
            
            # Calculate distance from center
            dist = np.sqrt((x_centroid - center_x)**2 + (y_centroid - center_y)**2)
            
            # Skip if beyond specified max distance
            if max_spine_distance is not None and dist > max_spine_distance:
                continue
            
            # PARALLEL DENDRITE DETECTION
            # Multiple methods to identify parallel dendrites:
            
            # 1. Basic Shape Analysis
            if hasattr(prop, 'axis_major_length') and hasattr(prop, 'axis_minor_length'):
                major_axis = prop.axis_major_length
                minor_axis = prop.axis_minor_length
                aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 10
            else:
                # Fallback to bounding box ratio
                minr, minc, maxr, maxc = prop.bbox
                height = maxr - minr
                width = maxc - minc
                aspect_ratio = max(height, width) / max(1, min(height, width))
            
            # 2. Connection Analysis
            # Get region mask
            region_mask = labeled_spines == prop.label
            
            # Create shaft mask
            shaft_mask = dist_from_center < shaft_radius * 1.1
            
            # Get region boundary
            boundary = morphology.binary_dilation(region_mask, morphology.disk(1)) & ~region_mask
            
            # Calculate perimeter length
            perimeter_length = np.sum(boundary)
            
            # Calculate length of boundary touching shaft
            touching_shaft = np.sum(boundary & shaft_mask)
            
            # Calculate percentage of perimeter touching shaft
            if perimeter_length > 0:
                touching_percentage = touching_shaft / perimeter_length
            else:
                touching_percentage = 0
            
            # 3. Check alignment with forward direction
            # Vector from center to centroid
            local_direction = np.array([0, dy_spine, dx_spine])
            local_direction_norm = np.linalg.norm(local_direction)
            if local_direction_norm > 0:
                local_direction = local_direction / local_direction_norm
                alignment_with_forward = np.abs(np.dot(local_direction, forward))
            else:
                alignment_with_forward = 0
            
            # 4. Calculate real-world coordinates of this spine
            spine_world_y = int(round(spine_position[1]))
            spine_world_x = int(round(spine_position[2]))
            
            # THE KEY NEW APPROACH: Cross-check in the 2D slice view
            is_connected_in_2d = False
            connection_confirmed = False
            
            if zoom_patch is not None and zoom_coordinates is not None:
                # Get zoom patch coordinates
                y_min, y_max, x_min, x_max = zoom_coordinates
                
                # Check if spine position is within zoom patch
                if (y_min <= spine_world_y < y_max and 
                    x_min <= spine_world_x < x_max):
                    
                    # Convert global position to zoom patch coordinates
                    patch_y = spine_world_y - y_min
                    patch_x = spine_world_x - x_min
                    
                    # Check if these coordinates are valid
                    if (0 <= patch_y < zoom_patch.shape[0] and 
                        0 <= patch_x < zoom_patch.shape[1]):
                        
                        # Get the value at this position
                        spine_intensity = zoom_patch[patch_y, patch_x]
                        
                        # Threshold the zoom patch to create a binary image
                        try:
                            zoom_thresh = filters.threshold_otsu(zoom_patch)
                        except:
                            zoom_thresh = 0.1
                        
                        zoom_binary = zoom_patch > zoom_thresh * 0.75
                        
                        # Clean up the binary image
                        zoom_binary = morphology.remove_small_objects(zoom_binary, min_size=5)
                        zoom_binary = morphology.binary_closing(zoom_binary, morphology.disk(1))
                        
                        # Label connected components
                        labeled_zoom = measure.label(zoom_binary)
                        
                        # Get the label at the spine position
                        if (0 <= patch_y < labeled_zoom.shape[0] and 
                            0 <= patch_x < labeled_zoom.shape[1]):
                            spine_label = labeled_zoom[patch_y, patch_x]
                            
                            # If the spine position has a label, check if it's connected to the dendrite path
                            if spine_label > 0:
                                # Get all points in this component
                                component = labeled_zoom == spine_label
                                
                                # Get the points along the dendrite path in this slice
                                dendrite_points = dendrite_path[
                                    (np.round(dendrite_path[:, 0]).astype(int) == z_slice_idx) & 
                                    (dendrite_path[:, 1] >= y_min) & (dendrite_path[:, 1] < y_max) &
                                    (dendrite_path[:, 2] >= x_min) & (dendrite_path[:, 2] < x_max)
                                ]
                                
                                # Convert dendrite points to patch coordinates
                                if len(dendrite_points) > 0:
                                    dend_patch_y = (dendrite_points[:, 1] - y_min).astype(int)
                                    dend_patch_x = (dendrite_points[:, 2] - x_min).astype(int)
                                    
                                    # Check if any dendrite points are in this component
                                    for dy, dx in zip(dend_patch_y, dend_patch_x):
                                        if (0 <= dy < labeled_zoom.shape[0] and 
                                            0 <= dx < labeled_zoom.shape[1] and 
                                            labeled_zoom[dy, dx] == spine_label):
                                            # The spine's component contains part of the dendrite path!
                                            is_connected_in_2d = True
                                            connection_confirmed = True
                                            break
                                            
                                # If we didn't find a direct connection, check for proximity
                                if not is_connected_in_2d and len(dendrite_points) > 0:
                                    # Convert component coordinates to array
                                    comp_coords = np.argwhere(component)
                                    
                                    # Convert dendrite points to array format
                                    dend_coords = np.column_stack((dend_patch_y, dend_patch_x))
                                    
                                    # For each point in the component, find the closest dendrite point
                                    min_distance = float('inf')
                                    for cy, cx in comp_coords:
                                        comp_pt = np.array([cy, cx])
                                        for dend_pt in dend_coords:
                                            dist = np.sqrt(np.sum((comp_pt - dend_pt)**2))
                                            min_distance = min(min_distance, dist)
                                    
                                    # If the closest dendrite point is within a threshold, consider it connected
                                    if min_distance < 5:  # 5 pixel proximity threshold
                                        is_connected_in_2d = True
                                        connection_confirmed = True
            
            # 5. SIMPLIFIED PARALLEL DENDRITE CRITERIA
            is_elongated = aspect_ratio > 3.0
            is_large = prop.area > 100
            has_small_connection = touching_percentage < 0.2
            is_aligned_with_forward = alignment_with_forward > 0.7
            
            # Combined check - Use connection check from 2D slice as primary filter
            is_likely_parallel = False
            if filter_dendrites:
                # If we confirmed it's NOT connected in the 2D view, it's likely a parallel dendrite
                if connection_confirmed and not is_connected_in_2d:
                    is_likely_parallel = True
                # Otherwise, use the traditional checks
                elif (is_elongated and is_large and has_small_connection) or (is_aligned_with_forward and is_large):
                    is_likely_parallel = True
                    
                if is_likely_parallel:
                    is_parallel_dendrite.append(len(frame_spines))
            
            # Standard spine scoring
            intensity_factor = (prop.mean_intensity - med_intensity) / (max_intensity - med_intensity) if max_intensity > med_intensity else 0
            size_factor = min(1.0, prop.area / 30)  # Cap size factor at 1.0
            distance_factor = min(1.0, dist / (shaft_radius * 2))  # Distance should be moderate
            
            # Add bonus for confirmed connection in 2D
            connection_factor = 0.2 if is_connected_in_2d else 0.0
            
            # Combined score with connection bonus
            spine_score = (intensity_factor * 0.3 + size_factor * 0.2 + distance_factor * 0.2 + connection_factor)
            
            # Skip spines with low scores
            if spine_score < min_spine_score:
                continue
            
            # Create spine info dictionary
            spine_info = {
                'position': spine_position,
                'frame_idx': frame_idx,
                'plane_x': x_centroid,
                'plane_y': y_centroid,
                'distance': dist,
                'intensity': prop.mean_intensity,
                'area': prop.area,
                'score': spine_score,
                'aspect_ratio': aspect_ratio,
                'touching_percentage': touching_percentage,
                'alignment_with_forward': alignment_with_forward,
                'is_connected_in_2d': is_connected_in_2d,
                'connection_confirmed': connection_confirmed,
                'is_likely_parallel': is_likely_parallel
            }
            
            # Add to candidates list if not likely a parallel dendrite
            if not is_likely_parallel:
                spine_candidates.append(spine_info)
            
            # Always add to frame-specific list
            frame_spines.append(spine_info)
            
            # Visualize
            if frame_idx in frame_indices:
                # Use different colors for different cases
                if is_likely_parallel:
                    marker_color = 'orange'  # Parallel dendrite
                elif is_connected_in_2d:
                    marker_color = 'green'   # Confirmed spine
                else:
                    marker_color = 'red'     # Potential spine
                
                ax.plot(x_centroid, y_centroid, 'o', color=marker_color, markersize=6)
                
                # Draw line from center to spine
                ax.plot([center_x, x_centroid], [center_y, y_centroid], 
                        color=marker_color, linewidth=0.5, alpha=0.5)
                
                # Add a small text label based on classification
                if is_likely_parallel:
                    label = "PD"
                elif is_connected_in_2d:
                    label = f"2D:{spine_score:.2f}"
                else:
                    label = f"{spine_score:.2f}"
                    
                ax.text(x_centroid + 2, y_centroid + 2, label, 
                       fontsize=8, color='yellow')
        
        # Create frame data dictionary
        frame_result = {
            'frame_idx': frame_idx,
            'position': current_position,
            'normal_plane': normal_plane,
            'dendrite_binary': dendrite_binary,
            'spine_regions': spine_regions,
            'segments': segments,
            'spines': frame_spines,
            'basis_vectors': {
                'up': up,
                'right': right,
                'forward': forward
            },
            'is_parallel_dendrite': is_parallel_dendrite
        }
        
        # Add to frame data list
        frame_data.append(frame_result)
        
        # Finish visualization for this frame
        if frame_idx in frame_indices:
            ax.set_title(f'Frame {frame_idx}')
            ax.axis('off')
    
    # Group nearby spine candidates
    spine_positions = []
    processed = set()

    
    # Sort candidates by score (highest first) to prioritize best detections
    sorted_candidates = sorted(spine_candidates, key=lambda x: x['score'], reverse=True)
    
    # Give preference to spines confirmed in 2D view
    sorted_candidates = sorted(sorted_candidates, 
                               key=lambda x: (x.get('is_connected_in_2d', False), x['score']), 
                               reverse=True)
    
    # Now cluster the candidates to get final spine positions
    for i, candidate in enumerate(sorted_candidates):
        if i in processed:
            continue
            
        pos_i = candidate['position']
        group = [candidate]
        processed.add(i)
        
        # Find nearby candidates
        for j, other in enumerate(sorted_candidates):
            if j in processed or i == j:
                continue
                
            pos_j = other['position']
            if np.linalg.norm(pos_i - pos_j) < distance_threshold:
                group.append(other)
                processed.add(j)
        
        # Prioritize 2D-confirmed spines within the group
        confirmed_spines = [s for s in group if s.get('is_connected_in_2d', False)]
        if confirmed_spines:
            # Use the highest-scoring confirmed spine
            best = max(confirmed_spines, key=lambda x: x['score'])
        else:
            # Otherwise use the highest-scoring spine
            best = max(group, key=lambda x: x['score'])
            
        spine_positions.append(best['position'])
    
    print(f"Detected {len(spine_positions)} spines after filtering parallel dendrites")
    
    return spine_positions, dendrite_path, frame_data

def find_spines_after_astar(image, start_point, end_point, pointsss, dendrite_mask):

    tube_data = create_tube_data(
        image=image,
        start_point=start_point,
        waypoints=pointsss,
        goal_point=end_point,
        reference_image=dendrite_mask,
        view_distance=0,
        field_of_view=50,
        zoom_size=50
    )

    spine_positions, dendrite_path, frame_data = detect_spines_with_watershed(
        tube_data,
        shaft_radius=8,              
        min_spine_intensity=0.8,      
        min_spine_area=5,            
        min_spine_score=0.1,          
        max_spine_distance=15,
        distance_threshold=15,         
        filter_dendrites=True  # Enable dendrite filtering
    )
    return spine_positions