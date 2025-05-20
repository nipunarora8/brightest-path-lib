import numpy as np
from brightest_path_lib.algorithm import WaypointBidirectionalAStarSearch
import matplotlib.cm as cm

def create_tube_data(image, start_point, goal_point, waypoints=None, 
                     view_distance=0, field_of_view=50, zoom_size=50, 
                     reference_image=None, reference_cmap='gray',
                     reference_alpha=0.7):
    """
    Generate data for a first-person fly-through visualization along the brightest path in a 3D image,
    returning full slice view, zoomed patch view, and tube view data.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The 3D image data (z, y, x)
    start_point : array-like
        Starting coordinates [z, y, x]
    goal_point : array-like
        Goal coordinates [z, y, x]
    waypoints : list of array-like, optional
        List of waypoints to include in the path
    view_distance : int
        How far ahead to look along the path
    field_of_view : int
        Width of the field of view in degrees
    zoom_size : int
        Size of the zoomed patch in pixels
    reference_image : numpy.ndarray, optional
        Optional reference image (can be 3D, same size as image, or 3D with 3 channels)
    reference_cmap : str, optional
        Colormap to use for the reference image 
    reference_alpha : float, optional
        Alpha (transparency) value for the reference image overlay
        
    Returns:
    --------
    list
        A list of dictionaries, each containing full slice view, zoomed patch view, and tube view data
    """
    # Validate reference image if provided
    is_multichannel = False
    if reference_image is not None:
        # Check dimensions
        if reference_image.shape[0] != image.shape[0]:
            raise ValueError("Reference image must have same number of z-slices as main image")
        
        # Handle multi-channel reference images
        if len(reference_image.shape) == 4:  # [z, y, x, channels]
            is_multichannel = True
            if reference_image.shape[1:3] != image.shape[1:3]:
                raise ValueError("Reference image must have same y,x dimensions as main image")
        elif reference_image.shape != image.shape:
            raise ValueError("Reference image must have same dimensions as main image")
    
    # Normalize images once (to save processing time)
    image_normalized = image.astype(float)
    if np.max(image_normalized) > 0:
        image_normalized = image_normalized / np.max(image_normalized)
    
    if reference_image is not None and not is_multichannel:
        ref_normalized = reference_image.astype(float)
        if np.max(ref_normalized) > 0:
            ref_normalized = ref_normalized / np.max(ref_normalized)
        
        # Create a custom colormap function that can be used like a matplotlib colormap
        def get_colormap_rgba(value):
            """Custom colormap function that works regardless of matplotlib version"""
            # Ensure value is in range [0, 1]
            value = np.clip(value, 0, 1)
            
            # Define some simple colormap functions
            if reference_cmap == 'viridis' or reference_cmap == 'plasma' or reference_cmap == 'inferno':
                # Blue to yellow-green to red
                r = np.interp(value, [0, 0.5, 1], [0.267, 0.000, 0.800])
                g = np.interp(value, [0, 0.5, 1], [0.005, 0.800, 0.000])
                b = np.interp(value, [0, 0.5, 1], [0.329, 0.200, 0.000])
            elif reference_cmap == 'jet':
                # Blue to cyan to yellow to red
                r = np.interp(value, [0, 0.35, 0.66, 0.89, 1], [0, 0, 1, 1, 0.5])
                g = np.interp(value, [0, 0.125, 0.375, 0.64, 0.91, 1], [0, 0, 1, 1, 0, 0])
                b = np.interp(value, [0, 0.11, 0.34, 0.65, 1], [0.5, 1, 1, 0, 0])
            else:
                # Default to grayscale
                r = g = b = value
                
            return np.array([r, g, b, 1.0])
        
        # Use our custom colormap function
        cmap = get_colormap_rgba
    
    # Run the brightest path algorithm
    astar = WaypointBidirectionalAStarSearch(
        image=image,
        start_point=np.array(start_point),
        goal_point=np.array(goal_point),
        waypoints=waypoints if waypoints else None
    )
    
    path = astar.search(verbose=True)
    
    if not astar.found_path:
        raise ValueError("Could not find a path through the image")
    
    # Convert path to numpy array for easier manipulation
    path_array = np.array(path)
    
    # Pre-compute tangent vectors (direction of travel)
    tangent_vectors = []
    
    for i in range(len(path)):
        # Simplified tangent calculation
        if i == 0:
            # At start, look ahead
            if len(path) > 1:
                tangent = path[1] - path[0]
            else:
                tangent = np.array([1, 0, 0])  # Default forward if only one point
        elif i == len(path) - 1:
            # At end, look back
            tangent = path[i] - path[i-1]
        else:
            # In middle, average direction
            prev_point = path[i-1]
            next_point = path[i+1]
            tangent = (next_point - prev_point) / 2.0
            
        # Normalize the tangent vector
        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent = tangent / norm
        
        tangent_vectors.append(tangent)
    
    # Create a list to store all data
    all_data = []
    
    # Initialize plane size for viewing
    plane_size = field_of_view // 2
    plane_shape = (plane_size * 2 + 1, plane_size * 2 + 1)
    
    # Calculate half-size of the zoom patch
    half_zoom = zoom_size // 2
    
    # Process each point in the path
    for current_idx in range(len(path)):
        current_point = path[current_idx]
        current_tangent = tangent_vectors[current_idx]
        
        # Convert to integers for indexing
        z, y, x = np.round(current_point).astype(int)
        
        # Ensure we're within image bounds
        z = np.clip(z, 0, image.shape[0] - 1)
        y = np.clip(y, 0, image.shape[1] - 1)
        x = np.clip(x, 0, image.shape[2] - 1)
        
        # Get full slice view at current Z position
        slice_view = image[z].copy()
        
        # Add reference slice if available
        ref_slice = None
        if reference_image is not None:
            if is_multichannel:
                ref_slice = reference_image[z].copy()
            else:
                ref_slice = reference_image[z].copy()
        
        # Calculate the zoomed patch coordinates
        y_min = max(0, y - half_zoom)
        y_max = min(image.shape[1], y + half_zoom)
        x_min = max(0, x - half_zoom)
        x_max = min(image.shape[2], x + half_zoom)
        
        # Extract the zoomed patch from the main image
        zoom_patch = image[z, y_min:y_max, x_min:x_max]
        
        # Extract the zoomed patch from the reference image if available
        zoom_patch_ref = None
        if reference_image is not None:
            if is_multichannel:
                zoom_patch_ref = reference_image[z, y_min:y_max, x_min:x_max]
            else:
                zoom_patch_ref = reference_image[z, y_min:y_max, x_min:x_max]
        
        # Create an orthogonal basis for the viewing plane
        forward = current_tangent  # Already normalized
        
        # Find the least aligned axis to create an orthogonal basis
        axis_alignments = np.abs(forward)
        least_aligned_idx = np.argmin(axis_alignments)
        
        # Create a reference vector along that axis
        reference = np.zeros(3)
        reference[least_aligned_idx] = 1.0
        
        # Compute right vector (orthogonal to forward)
        right = np.cross(forward, reference)
        right = right / np.linalg.norm(right)
        
        # Compute up vector (orthogonal to forward and right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Initialize arrays for both normal and colored planes
        normal_plane = np.zeros(plane_shape)
        
        if reference_image is not None:
            if is_multichannel:
                colored_plane = np.zeros((*plane_shape, 3))
            else:
                colored_plane = np.zeros((*plane_shape, 3))
        else:
            colored_plane = None
        
        # Create mapping arrays for all possible viewing plane points
        points_3d = np.zeros((*plane_shape, 3))
        for i in range(plane_shape[0]):
            for j in range(plane_shape[1]):
                points_3d[i, j] = current_point + (i - plane_size) * up + (j - plane_size) * right
        
        # Round to integers for image lookup
        points_3d_int = np.round(points_3d).astype(int)
        
        # Create mask for valid points (within image bounds)
        valid_mask = (
            (points_3d_int[:, :, 0] >= 0) & (points_3d_int[:, :, 0] < image.shape[0]) &
            (points_3d_int[:, :, 1] >= 0) & (points_3d_int[:, :, 1] < image.shape[1]) &
            (points_3d_int[:, :, 2] >= 0) & (points_3d_int[:, :, 2] < image.shape[2])
        )
        
        # Process only valid points for faster execution
        for i, j in zip(*np.where(valid_mask)):
            pz, py, px = points_3d_int[i, j]
            
            # Set normal plane value
            normal_plane[i, j] = image[pz, py, px]
            
            # Set colored plane if reference image is provided
            if reference_image is not None:
                val = image_normalized[pz, py, px]
                
                if is_multichannel:
                    # Get reference RGB values
                    ref_rgb = reference_image[pz, py, px]
                    # Normalize if needed
                    if np.max(ref_rgb) > 1:
                        ref_rgb = ref_rgb / 255.0
                    
                    # Blend with main image value
                    colored_plane[i, j, 0] = val * (1 - reference_alpha) + ref_rgb[0] * reference_alpha
                    colored_plane[i, j, 1] = val * (1 - reference_alpha) + ref_rgb[1] * reference_alpha
                    colored_plane[i, j, 2] = val * (1 - reference_alpha) + ref_rgb[2] * reference_alpha
                else:
                    # Get reference value
                    ref_val = ref_normalized[pz, py, px]
                    
                    # Get color from colormap
                    rgba = cmap(ref_val)
                    
                    # Blend grayscale and color
                    colored_plane[i, j, 0] = val * (1 - reference_alpha) + rgba[0] * reference_alpha
                    colored_plane[i, j, 1] = val * (1 - reference_alpha) + rgba[1] * reference_alpha
                    colored_plane[i, j, 2] = val * (1 - reference_alpha) + rgba[2] * reference_alpha
        
        # Calculate path ahead points - MODIFIED: Use exact view_distance parameter
        ahead_points = []
        
        # Only calculate ahead points if view_distance > 0
        if view_distance > 0:
            for i in range(1, view_distance + 1):
                next_idx = min(current_idx + i, len(path) - 1)
                if next_idx == current_idx:
                    break
                    
                # Vector from current point to next point
                next_point = path[next_idx]
                vector = next_point - current_point
                
                # Project onto viewing plane
                forward_dist = np.dot(vector, forward)
                
                # Only show points that are ahead
                if forward_dist > 0:
                    up_component = np.dot(vector, up)
                    right_component = np.dot(vector, right)
                    
                    # Convert to viewing plane coordinates
                    view_y = plane_size + int(up_component)
                    view_x = plane_size + int(right_component)
                    
                    # Check if within viewing plane
                    if (0 <= view_y < plane_shape[0] and 0 <= view_x < plane_shape[1]):
                        ahead_points.append((view_y, view_x))
        
        # Find path points within the current z-slice for visualization
        slice_points = path_array[np.round(path_array[:, 0]).astype(int) == z]
        
        # Find path points within the zoomed patch and transform coordinates
        if len(slice_points) > 0:
            patch_slice_points = slice_points[
                (slice_points[:, 1] >= y_min) & (slice_points[:, 1] < y_max) &
                (slice_points[:, 2] >= x_min) & (slice_points[:, 2] < x_max)
            ]
        else:
            patch_slice_points = np.array([])
        
        # Transform patch path points to local coordinates
        if len(patch_slice_points) > 0:
            # Convert to patch-local coordinates
            patch_path_points = patch_slice_points.copy()
            patch_path_points[:, 1] = patch_path_points[:, 1] - y_min
            patch_path_points[:, 2] = patch_path_points[:, 2] - x_min
        else:
            patch_path_points = np.array([])
        
        # Create frame data with all views
        frame_data = {
            # Full slice view
            'slice_view': slice_view,
            'ref_slice': ref_slice,
            'slice_path_points': slice_points if len(slice_points) > 0 else None,
            
            # Zoomed patch view
            'zoom_patch': zoom_patch,
            'zoom_patch_ref': zoom_patch_ref,
            'zoom_coordinates': (y_min, y_max, x_min, x_max),
            'patch_path_points': patch_path_points if len(patch_path_points) > 0 else None,
            
            # Tube view
            'normal_plane': normal_plane,
            'colored_plane': colored_plane,
            'ahead_points': ahead_points,
            
            # Position information
            'position': (z, y, x),
            'frame_index': current_idx,
            'total_frames': len(path),
            'current_point_in_path': current_point,
            
            # Vector basis
            'basis_vectors': {
                'forward': forward,
                'up': up,
                'right': right
            }
        }
        
        # Add to collection
        all_data.append(frame_data)
    
    print(f"Generated complete data for {len(all_data)} frames with full slice, zoomed patch, and tube views")
    return all_data