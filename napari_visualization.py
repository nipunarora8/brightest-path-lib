import napari
import numpy as np
import imageio as io
from napari.layers import Points
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, 
    QHBoxLayout, QFileDialog, QTabWidget,
    QListWidget, QListWidgetItem, QFrame
)
from brightest_path_lib.algorithm import BidirectionalAStarSearch
import sys
import uuid


class BrightestPathWidget(QWidget):
    def __init__(self, viewer, image):
        super().__init__()
        self.viewer = viewer
        self.image = image
        self.start_point = None
        self.end_point = None
        
        # Track multiple paths
        self.paths = {}  # Dictionary to store multiple paths
        self.path_layers = {}  # Dictionary to store individual path layers
        self.next_path_number = 1  # For naming paths (Path 1, Path 2, etc.)
        self.color_idx = 0  # Index to cycle through colors
        
        # Flag to prevent recursive event handling
        self.handling_event = False
        
        # Create layers
        self.image_layer = self.viewer.add_image(
            self.image, name='Image', colormap='gray'
        )
        
        self.start_points_layer = self.viewer.add_points(
            np.empty((0, self.image.ndim)),
            name='Start Point',
            size=15,
            face_color='lime',
            symbol='x'
        )
        
        self.end_points_layer = self.viewer.add_points(
            np.empty((0, self.image.ndim)),
            name='End Point',
            size=15,
            face_color='red',
            symbol='x'
        )
        
        # Add a special layer for 3D traced path visualization
        if self.image.ndim > 2:
            self.traced_path_layer = self.viewer.add_points(
                np.empty((0, self.image.ndim)),
                name='Traced Path (3D)',
                size=4,
                face_color='magenta',
                opacity=0.7,
                visible=False  # Hidden by default until 3D path is calculated
            )
        
        # Set up UI
        self.setup_ui()
        
        # Set up event handling for points layers
        self.start_points_layer.events.data.connect(self.on_start_point_changed)
        self.end_points_layer.events.data.connect(self.on_end_point_changed)
        
        # Default mode for points layers
        self.start_points_layer.mode = 'add'
        self.end_points_layer.mode = 'add'
        
        # By default, activate the start point layer to begin workflow
        self.viewer.layers.selection.active = self.start_points_layer
        show_info("Start point layer activated. Click on the image to set start point.")
    
    def setup_ui(self):
        """Create the UI panel with controls"""
        try:
            layout = QVBoxLayout()
            self.setLayout(layout)
            
            # Title
            title = QLabel("<b>Brightest Path Finder</b>")
            layout.addWidget(title)
            
            # Create tabs for different functionality
            self.tabs = QTabWidget()
            
            # First tab: Point Selection
            point_selection_tab = QWidget()
            point_layout = QVBoxLayout()
            point_selection_tab.setLayout(point_layout)
            
            # Start point section
            start_section = QWidget()
            start_layout = QVBoxLayout()
            start_section.setLayout(start_layout)
            
            start_instr = QLabel("Click on the image to set the start point.")
            start_layout.addWidget(start_instr)
            
            self.select_start_btn = QPushButton("Select Start Point Layer")
            self.select_start_btn.clicked.connect(self.activate_start_layer)
            start_layout.addWidget(self.select_start_btn)
            
            self.start_status = QLabel("Status: No start point set")
            start_layout.addWidget(self.start_status)
            point_layout.addWidget(start_section)
            
            # Separator
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            point_layout.addWidget(separator)
            
            # End point section
            end_section = QWidget()
            end_layout = QVBoxLayout()
            end_section.setLayout(end_layout)
            
            end_instr = QLabel("Click on the image to set the end point.")
            end_layout.addWidget(end_instr)
            
            self.select_end_btn = QPushButton("Select End Point Layer")
            self.select_end_btn.clicked.connect(self.activate_end_layer)
            end_layout.addWidget(self.select_end_btn)
            
            self.end_status = QLabel("Status: No end point set")
            end_layout.addWidget(self.end_status)
            point_layout.addWidget(end_section)
            
            # Find path button
            find_btns_layout = QHBoxLayout()
            
            self.find_path_btn = QPushButton("Find Path")
            self.find_path_btn.clicked.connect(self.find_path)
            self.find_path_btn.setEnabled(False)
            find_btns_layout.addWidget(self.find_path_btn)
            
            self.retrace_path_btn = QPushButton("Retrace Path")
            self.retrace_path_btn.clicked.connect(self.retrace_path)
            self.retrace_path_btn.setEnabled(False)
            find_btns_layout.addWidget(self.retrace_path_btn)
            
            point_layout.addLayout(find_btns_layout)
            
            # Trace Another Path button
            self.trace_another_btn = QPushButton("Trace Another Path")
            self.trace_another_btn.clicked.connect(self.trace_another_path)
            self.trace_another_btn.setEnabled(False)  # Disabled until we have a path
            point_layout.addWidget(self.trace_another_btn)
            
            # Clear points button
            self.clear_points_btn = QPushButton("Clear Points (Start Over)")
            self.clear_points_btn.clicked.connect(self.clear_points)
            point_layout.addWidget(self.clear_points_btn)
            
            # Status messages
            self.error_status = QLabel("")
            self.error_status.setStyleSheet("color: red;")
            point_layout.addWidget(self.error_status)
            
            # Second tab: Path Management
            path_management_tab = QWidget()
            path_layout = QVBoxLayout()
            path_management_tab.setLayout(path_layout)
            
            # Path list with instructions
            path_layout.addWidget(QLabel("Saved Paths (select two paths to connect them):"))
            self.path_list = QListWidget()
            self.path_list.setSelectionMode(QListWidget.ExtendedSelection)  # Allow multiple selection
            self.path_list.itemSelectionChanged.connect(self.on_path_selection_changed)
            path_layout.addWidget(self.path_list)
            
            # Path management buttons
            path_buttons_layout = QHBoxLayout()
            
            self.view_path_btn = QPushButton("View Selected Path")
            self.view_path_btn.clicked.connect(self.view_selected_path)
            self.view_path_btn.setEnabled(False)
            path_buttons_layout.addWidget(self.view_path_btn)
            
            self.delete_path_btn = QPushButton("Delete Selected Path(s)")
            self.delete_path_btn.clicked.connect(self.delete_selected_paths)
            self.delete_path_btn.setEnabled(False)
            path_buttons_layout.addWidget(self.delete_path_btn)
            
            path_layout.addLayout(path_buttons_layout)
            
            # Path connection button
            self.connect_paths_btn = QPushButton("Connect Selected Paths")
            self.connect_paths_btn.setToolTip("Select exactly 2 paths to connect them")
            self.connect_paths_btn.clicked.connect(self.connect_selected_paths)
            self.connect_paths_btn.setEnabled(False)
            path_layout.addWidget(self.connect_paths_btn)
            
            # Path visibility options
            visibility_layout = QHBoxLayout()
            
            self.show_all_btn = QPushButton("Show All Paths")
            self.show_all_btn.clicked.connect(lambda: self.set_paths_visibility(True))
            visibility_layout.addWidget(self.show_all_btn)
            
            self.hide_all_btn = QPushButton("Hide All Paths")
            self.hide_all_btn.clicked.connect(lambda: self.set_paths_visibility(False))
            visibility_layout.addWidget(self.hide_all_btn)
            
            path_layout.addLayout(visibility_layout)
            
            # Export button
            self.export_all_btn = QPushButton("Export All Paths")
            self.export_all_btn.clicked.connect(self.export_all_paths)
            self.export_all_btn.setEnabled(False)
            path_layout.addWidget(self.export_all_btn)
            
            # Add tabs to widget
            self.tabs.addTab(point_selection_tab, "Point Selection")
            self.tabs.addTab(path_management_tab, "Path Management")
            layout.addWidget(self.tabs)
            
            # Current path info at the bottom
            self.path_info = QLabel("Path: Not calculated")
            layout.addWidget(self.path_info)
        except Exception as e:
            print(f"Error setting up UI: {str(e)}")
    
    def activate_start_layer(self):
        """Activate the start point layer for selecting"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            self.viewer.layers.selection.active = self.start_points_layer
            self.error_status.setText("")
            show_info("Start point layer activated. Click on the image to set start point.")
        except Exception as e:
            error_msg = f"Error activating start point layer: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def activate_end_layer(self):
        """Activate the end point layer for selecting"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            self.viewer.layers.selection.active = self.end_points_layer
            self.error_status.setText("")
            show_info("End point layer activated. Click on the image to set end point.")
        except Exception as e:
            error_msg = f"Error activating end point layer: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def on_start_point_changed(self, event=None):
        """Handle when start point is added or changed"""
        # Prevent recursive function calls
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if len(self.start_points_layer.data) > 0:
                # Keep only the last added point
                if len(self.start_points_layer.data) > 1:
                    self.start_points_layer.data = self.start_points_layer.data[-1:]
                
                # Store the point
                self.start_point = self.start_points_layer.data[0].astype(int)
                
                # Validate the point is within image bounds
                valid = True
                for i, coord in enumerate(self.start_point):
                    if coord < 0 or coord >= self.image.shape[i]:
                        valid = False
                        break
                        
                if not valid:
                    show_info("Start point is outside image bounds. Please try again.")
                    self.start_points_layer.data = np.empty((0, self.image.ndim))
                    self.start_point = None
                    return
                
                # Update status
                coords_str = np.array2string(self.start_point, precision=0)
                self.start_status.setText(f"Status: Start point set at {coords_str}")
                
                # Auto-switch to end point layer
                self.viewer.layers.selection.active = self.end_points_layer
                show_info("End point layer activated. Click on the image to set end point.")
                
                # Check if we can enable the find path button
                self.update_find_path_button()
        except Exception as e:
            show_info(f"Error setting start point: {str(e)}")
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
    
    def on_end_point_changed(self, event=None):
        """Handle when end point is added or changed"""
        # Prevent recursive function calls
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if len(self.end_points_layer.data) > 0:
                # Keep only the last added point
                if len(self.end_points_layer.data) > 1:
                    self.end_points_layer.data = self.end_points_layer.data[-1:] 
                
                # Store the point
                self.end_point = self.end_points_layer.data[0].astype(int)
                
                # Validate the point is within image bounds
                valid = True
                for i, coord in enumerate(self.end_point):
                    if coord < 0 or coord >= self.image.shape[i]:
                        valid = False
                        break
                        
                if not valid:
                    show_info("End point is outside image bounds. Please try again.")
                    self.end_points_layer.data = np.empty((0, self.image.ndim))
                    self.end_point = None
                    return
                
                # Update status
                coords_str = np.array2string(self.end_point, precision=0)
                self.end_status.setText(f"Status: End point set at {coords_str}")
                
                # Check if we can enable the find path button
                self.update_find_path_button()
        except Exception as e:
            show_info(f"Error setting end point: {str(e)}")
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
    
    def update_find_path_button(self):
        """Enable or disable the find path button based on point selection"""
        if self.start_point is not None and self.end_point is not None:
            self.find_path_btn.setEnabled(True)
            self.retrace_path_btn.setEnabled(True)
        else:
            self.find_path_btn.setEnabled(False)
            self.retrace_path_btn.setEnabled(False)
    
    def on_path_selection_changed(self):
        """Handle when path selection changes in the list"""
        # Prevent processing during updates
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            num_selected = len(selected_items)
            
            # Enable/disable buttons based on selection
            self.delete_path_btn.setEnabled(num_selected > 0)
            self.view_path_btn.setEnabled(num_selected == 1)
            self.connect_paths_btn.setEnabled(num_selected == 2)
        except Exception as e:
            show_info(f"Error handling selection change: {str(e)}")
        finally:
            self.handling_event = False
    
    def view_selected_path(self):
        """View the selected path from the list"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            if len(selected_items) != 1:
                return
                
            item = selected_items[0]
            path_id = item.data(100)
            if path_id in self.paths:
                path_data = self.paths[path_id]
                
                # Update the start and end points
                if path_data['start'] is not None and path_data['end'] is not None:
                    self.start_point = path_data['start'].copy()
                    self.end_point = path_data['end'].copy()
                    
                    # Update the start and end point layers
                    self.start_points_layer.data = np.array([self.start_point])
                    self.end_points_layer.data = np.array([self.end_point])
                    
                    # Update UI
                    self.start_status.setText(f"Status: Start point set at {np.array2string(self.start_point, precision=0)}")
                    self.end_status.setText(f"Status: End point set at {np.array2string(self.end_point, precision=0)}")
                
                self.path_info.setText(f"Path: {path_data['name']} loaded with {len(path_data['data'])} points")
                
                # Store current path ID
                self.current_path_id = path_id
                
                # Enable buttons
                self.find_path_btn.setEnabled(self.start_point is not None and self.end_point is not None)
                self.retrace_path_btn.setEnabled(self.start_point is not None and self.end_point is not None)
                self.trace_another_btn.setEnabled(True)
                
                # Switch to the Point Selection tab
                self.tabs.setCurrentIndex(0)
                
                # Ensure the selected path's layer is visible
                path_data['layer'].visible = True
                
                # Clear any error messages
                self.error_status.setText("")
                
                show_info(f"Loaded {path_data['name']}")
        except Exception as e:
            error_msg = f"Error viewing path: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def trace_another_path(self):
        """Reset everything to start a new path"""
        self.reset_for_new_path()
        
        # Activate the start point layer for the new path
        self.viewer.layers.selection.active = self.start_points_layer
        show_info("Ready to trace a new path. Click on the image to set start point.")
    
    def reset_for_new_path(self):
        """Reset everything for a new path"""
        # Clear points
        self.start_point = None
        self.end_point = None
        self.start_points_layer.data = np.empty((0, self.image.ndim))
        self.end_points_layer.data = np.empty((0, self.image.ndim))
        
        # Clear traced path layer if it exists
        if hasattr(self, 'traced_path_layer'):
            self.traced_path_layer.data = np.empty((0, self.image.ndim))
            self.traced_path_layer.visible = False
            
        # Reset UI
        self.start_status.setText("Status: No start point set")
        self.end_status.setText("Status: No end point set")
        self.path_info.setText("Path: Not calculated")
        
        # Reset buttons
        self.find_path_btn.setEnabled(False)
        self.retrace_path_btn.setEnabled(False)
        self.trace_another_btn.setEnabled(False)
    
    def clear_points(self):
        """Clear all points and path without saving"""
        self.reset_for_new_path()
        show_info("All points and path cleared. Ready to start over.")
    
    def find_path(self):
        """Find the brightest path between the selected points"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            if self.start_point is None or self.end_point is None:
                show_info("Please set both start and end points")
                self.error_status.setText("Error: Please set both start and end points")
                return
            
            # Clear any previous error messages
            self.error_status.setText("")
            show_info("Finding brightest path...")
            self.path_info.setText("Path: Calculating...")
            
            # Determine if we're doing 2D or 3D search based on whether points are on same frame
            is_same_frame = True
            if self.image.ndim > 2:  # Only relevant for 3D+ images
                is_same_frame = self.start_point[0] == self.end_point[0]
            
            # Prepare points format based on 2D or 3D
            if is_same_frame and self.image.ndim > 2:
                # 2D case: use [y, x] format (ignore z)
                search_start = self.start_point[1:3]  # [y, x]
                search_end = self.end_point[1:3]      # [y, x]
                search_image = self.image[int(self.start_point[0])]  # Use the frame of the start point
                show_info(f"Using 2D path search on frame {int(self.start_point[0])}")
                
                # Hide 3D traced path layer if it exists
                if hasattr(self, 'traced_path_layer'):
                    self.traced_path_layer.visible = False
            else:
                # 3D case or already 2D image: use full coordinates
                search_start = self.start_point
                search_end = self.end_point
                search_image = self.image
                show_info("Using 3D path search across frames")
                
                # Show 3D traced path layer if it exists
                if hasattr(self, 'traced_path_layer'):
                    self.traced_path_layer.visible = True
            
            # Set up the search algorithm
            search_algorithm = BidirectionalAStarSearch(
                search_image, 
                start_point=search_start, 
                goal_point=search_end
            )
            
            # Find the path
            brightest_path = search_algorithm.search()
            
            # Process the found path
            if brightest_path is not None and len(brightest_path) > 0:
                # If 2D search was done, need to add z-coordinate back
                if is_same_frame and self.image.ndim > 2:
                    z_val = self.start_point[0]
                    # Add z coordinate to each point
                    brightest_path_3d = []
                    for point in brightest_path:
                        if len(point) == 2:  # [y, x]
                            brightest_path_3d.append([z_val, point[0], point[1]])
                        else:
                            brightest_path_3d.append(point)  # Already has z
                    
                    brightest_path = brightest_path_3d
                
                # Generate path name
                path_name = f"Path {self.next_path_number}"
                self.next_path_number += 1
                
                # Get color for this path
                path_color = self.get_next_color()
                
                # Create a new layer for this path
                path_data = np.array(brightest_path)
                path_layer = self.viewer.add_points(
                    path_data,
                    name=path_name,
                    size=3,
                    face_color=path_color,
                    opacity=0.8
                )
                
                # For 3D visualization, create a traced path that shows the whole path in every frame
                if not is_same_frame and self.image.ndim > 2 and hasattr(self, 'traced_path_layer'):
                    # Get the z-range (frame range) that we need to span
                    z_values = [point[0] for point in brightest_path]
                    min_z = int(min(z_values))
                    max_z = int(max(z_values))
                    
                    # Create a projection of the path onto every frame in the range
                    traced_points = []
                    
                    # For each frame in our range
                    for z in range(min_z, max_z + 1):
                        # Add all path points to this frame by changing their z-coordinate
                        for point in brightest_path:
                            # Create a new point with the current frame's z-coordinate
                            new_point = point.copy()
                            new_point[0] = z  # Set the z-coordinate to the current frame
                            traced_points.append(new_point)
                    
                    # Update the traced path layer with all these points
                    self.traced_path_layer.data = np.array(traced_points)
                    self.traced_path_layer.visible = True
                    
                    # Navigate to the frame where the path starts to provide better initial view
                    self.viewer.dims.set_point(0, min_z)
                
                # Generate a unique ID for this path
                path_id = str(uuid.uuid4())
                
                # Store the path with all its data
                self.paths[path_id] = {
                    'name': path_name,
                    'data': path_data,
                    'start': self.start_point.copy() if self.start_point is not None else None,
                    'end': self.end_point.copy() if self.end_point is not None else None,
                    'visible': True,
                    'layer': path_layer
                }
                
                # Store reference to the layer
                self.path_layers[path_id] = path_layer
                
                # Add to path list
                item = QListWidgetItem(path_name)
                item.setData(100, path_id)  # Store path ID as custom data
                self.path_list.addItem(item)
                
                # Update UI
                msg = f"Path found with {len(brightest_path)} points, saved as {path_name}"
                show_info(msg)
                self.path_info.setText(f"Path: {msg}")
                
                # Enable trace another path button
                self.trace_another_btn.setEnabled(True)
                
                # Enable export all button
                self.export_all_btn.setEnabled(True)
                
                # Store current path ID
                self.current_path_id = path_id
            else:
                # No path found
                msg = "No path found"
                show_info(msg)
                self.path_info.setText(f"Path: {msg}")
                self.error_status.setText("Error: No path found between selected points")
                self.trace_another_btn.setEnabled(False)
        except Exception as e:
            msg = f"Error finding path: {e}"
            show_info(msg)
            self.path_info.setText(f"Path: Error - {str(e)}")
            self.error_status.setText(f"Error: {str(e)}")
        finally:
            self.handling_event = False
    
    def retrace_path(self):
        """Retrace the path using the same start and end points"""
        # First remove the previous path if it exists
        if hasattr(self, 'current_path_id') and self.current_path_id in self.paths:
            # Get the layer
            path_layer = self.path_layers[self.current_path_id]
            
            # Remove from napari
            self.viewer.layers.remove(path_layer)
            
            # Remove from data structures
            del self.path_layers[self.current_path_id]
            del self.paths[self.current_path_id]
            
            # Remove from list
            for i in range(self.path_list.count()):
                item = self.path_list.item(i)
                if item.data(100) == self.current_path_id:
                    self.path_list.takeItem(i)
                    break
        
        # Now find a new path
        self.find_path()
        show_info("Path retraced with same start and end points")
    
    def get_next_color(self):
        """Get the next color from the predefined list"""
        # List of named colors that work well for paths
        colors = ['cyan', 'magenta', 'green', 'blue', 'orange', 
                  'purple', 'teal', 'coral', 'gold', 'lavender']
        
        # Get the next color and increment the counter
        color = colors[self.color_idx % len(colors)]
        self.color_idx += 1
        
        return color
    
    def save_current_path(self):
        """Save the current path to the path collection"""
        if self.path_layer.data is None or len(self.path_layer.data) == 0:
            show_info("No path to save")
            return False
            
        try:
            # Generate a unique ID for this path
            path_id = str(uuid.uuid4())
            path_name = f"Path {self.next_path_number}"
            self.next_path_number += 1
            
            # Get a color for this path
            path_color = self.get_next_color()
            
            # Create a new layer for this path
            path_layer = self.viewer.add_points(
                self.path_layer.data.copy(),
                name=path_name,
                size=3,
                face_color=path_color,  # Use a named color
                opacity=0.7
            )
            
            # Store the path with all its data
            self.paths[path_id] = {
                'name': path_name,
                'data': self.path_layer.data.copy(),
                'start': self.start_point.copy() if self.start_point is not None else None,
                'end': self.end_point.copy() if self.end_point is not None else None,
                'visible': True,
                'layer': path_layer
            }
            
            # Store reference to the layer
            self.path_layers[path_id] = path_layer
            
            # Add to path list
            item = QListWidgetItem(path_name)
            item.setData(100, path_id)  # Store path ID as custom data
            self.path_list.addItem(item)
            
            # Enable export all button
            self.export_all_btn.setEnabled(True)
            
            show_info(f"Path saved as '{path_name}'")
            return True
        except Exception as e:
            show_info(f"Error saving path: {str(e)}")
            return False
    
    def delete_selected_paths(self):
        """Delete the currently selected paths"""
        selected_items = self.path_list.selectedItems()
        if not selected_items:
            show_info("No paths selected")
            return
        
        for item in selected_items:
            # Get the path ID
            path_id = item.data(100)
            
            if path_id in self.paths:
                path_name = self.paths[path_id]['name']
                
                # Remove the layer from viewer
                if path_id in self.path_layers:
                    self.viewer.layers.remove(self.path_layers[path_id])
                    del self.path_layers[path_id]
                
                # Remove from dictionary
                del self.paths[path_id]
                
                # Remove from list widget
                row = self.path_list.row(item)
                self.path_list.takeItem(row)
                
                show_info(f"Deleted {path_name}")
        
        # Disable buttons if no paths remain
        if self.path_list.count() == 0:
            self.delete_path_btn.setEnabled(False)
            self.view_path_btn.setEnabled(False)
            self.connect_paths_btn.setEnabled(False)
            self.export_all_btn.setEnabled(False)
    
    def set_paths_visibility(self, visible):
        """Set visibility of all saved path layers and update traced path visualization"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            # Show/hide individual path layers
            for path_id, layer in self.path_layers.items():
                layer.visible = visible
                self.paths[path_id]['visible'] = visible
            
            # Update traced path visualization
            if self.image.ndim > 2 and hasattr(self, 'traced_path_layer'):
                if visible:
                    # Create a comprehensive visualization of all paths in the traced layer
                    all_traced_points = []
                    
                    # First, determine the full z-range for all paths
                    min_z = float('inf')
                    max_z = float('-inf')
                    
                    for path_id, path_data in self.paths.items():
                        if len(path_data['data']) > 0:
                            z_values = [point[0] for point in path_data['data']]
                            path_min_z = int(min(z_values))
                            path_max_z = int(max(z_values))
                            
                            min_z = min(min_z, path_min_z)
                            max_z = max(max_z, path_max_z)
                    
                    # If we have valid z-range
                    if min_z != float('inf') and max_z != float('-inf'):
                        # For each frame in the full range
                        for z in range(min_z, max_z + 1):
                            # Add all paths to this frame
                            for path_id, path_data in self.paths.items():
                                if path_data['visible']:
                                    for point in path_data['data']:
                                        # Create a new point with the current frame's z-coordinate
                                        new_point = point.copy()
                                        new_point[0] = z  # Set the z-coordinate to the current frame
                                        all_traced_points.append(new_point)
                        
                        # Update the traced path layer
                        if all_traced_points:
                            self.traced_path_layer.data = np.array(all_traced_points)
                            self.traced_path_layer.visible = True
                            
                            # Navigate to the first frame
                            self.viewer.dims.set_point(0, min_z)
                else:
                    # Hide traced path layer when hiding all paths
                    self.traced_path_layer.data = np.empty((0, self.image.ndim))
                    self.traced_path_layer.visible = False
            
            action = "shown" if visible else "hidden"
            show_info(f"All paths {action}")
        except Exception as e:
            error_msg = f"Error updating path visibility: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
        finally:
            self.handling_event = False
    
    def connect_selected_paths(self):
        """Connect two selected paths"""
        if self.handling_event:
            return
            
        try:
            self.handling_event = True
            
            selected_items = self.path_list.selectedItems()
            
            if len(selected_items) != 2:
                show_info("Please select exactly two paths to connect")
                return
                
            # Get the path IDs
            path_id1 = selected_items[0].data(100)
            path_id2 = selected_items[1].data(100)
            
            if path_id1 not in self.paths or path_id2 not in self.paths:
                show_info("Invalid path selection")
                return
                
            # Get the path data
            path1 = self.paths[path_id1]
            path2 = self.paths[path_id2]
            
            # Check if paths have start/end points
            if path1['start'] is None or path2['end'] is None:
                show_info("Both paths must have start and end points to connect them")
                return
                
            # Get start of path1 and end of path2
            start_point = path1['start']
            end_point = path2['end']
            
            show_info(f"Connecting {path1['name']} to {path2['name']}...")
            
            # Determine if we're doing 2D or 3D search
            is_same_frame = True
            if self.image.ndim > 2:
                is_same_frame = start_point[0] == end_point[0]
                
            # Prepare points format based on 2D or 3D
            if is_same_frame and self.image.ndim > 2:
                # 2D case: use [y, x] format (ignore z)
                search_start = start_point[1:3]  # [y, x]
                search_end = end_point[1:3]      # [y, x]
                search_image = self.image[int(start_point[0])]
                show_info(f"Using 2D path search on frame {int(start_point[0])}")
            else:
                # 3D case or already 2D image: use full coordinates
                search_start = start_point
                search_end = end_point
                search_image = self.image
                show_info("Using 3D path search across frames")
                
            # Search for connecting path
            search_algorithm = BidirectionalAStarSearch(
                search_image, 
                start_point=search_start, 
                goal_point=search_end
            )
            
            connecting_path = search_algorithm.search()
            
            # If path found, create combined path
            if connecting_path is not None and len(connecting_path) > 0:
                # Fix coordinates if needed (2D case)
                if is_same_frame and self.image.ndim > 2:
                    z_val = start_point[0]
                    fixed_connecting_path = []
                    for point in connecting_path:
                        if len(point) == 2:  # [y, x]
                            fixed_connecting_path.append([z_val, point[0], point[1]])
                        else:
                            fixed_connecting_path.append(point)
                    connecting_path = fixed_connecting_path
                
                # Convert to numpy arrays
                path1_data = path1['data']
                path2_data = path2['data']
                connecting_data = np.array(connecting_path)
                
                # Create combined path
                combined_path = np.vstack([path1_data, connecting_data, path2_data])
                
                # Create a name for the combined path
                combined_name = f"{path1['name']} + {path2['name']}"
                
                # Get a color
                combined_color = self.get_next_color()
                
                # Create a new layer
                combined_layer = self.viewer.add_points(
                    combined_path,
                    name=combined_name,
                    size=3,
                    face_color=combined_color,
                    opacity=0.7
                )
                
                # Create traced path visualization for combined path
                # This is especially important when connecting 2D+3D paths
                if self.image.ndim > 2 and hasattr(self, 'traced_path_layer'):
                    # Get the z-range for the entire combined path
                    z_values = [point[0] for point in combined_path]
                    min_z = int(min(z_values))
                    max_z = int(max(z_values))
                    
                    # If z-range spans more than one slice, generate traced path
                    if max_z > min_z:
                        # Create a projection of the path onto every frame in the range
                        traced_points = []
                        
                        # For each frame in our range
                        for z in range(min_z, max_z + 1):
                            # Add all path points to this frame by changing their z-coordinate
                            for point in combined_path:
                                # Create a new point with the current frame's z-coordinate
                                new_point = point.copy()
                                new_point[0] = z  # Set the z-coordinate to the current frame
                                traced_points.append(new_point)
                        
                        # Update the traced path layer with all these points
                        self.traced_path_layer.data = np.array(traced_points)
                        self.traced_path_layer.visible = True
                        
                        # Navigate to the frame where the path starts for better visibility
                        self.viewer.dims.set_point(0, min_z)
                
                # Generate a unique ID for this path
                path_id = str(uuid.uuid4())
                
                # Store the combined path
                self.paths[path_id] = {
                    'name': combined_name,
                    'data': combined_path,
                    'start': path1['start'].copy(),
                    'end': path2['end'].copy(),
                    'visible': True,
                    'layer': combined_layer
                }
                
                # Store reference to the layer
                self.path_layers[path_id] = combined_layer
                
                # Add to path list
                item = QListWidgetItem(combined_name)
                item.setData(100, path_id)
                self.path_list.addItem(item)
                
                # Select the new path
                self.path_list.setCurrentItem(item)
                
                # Update UI
                msg = f"Connected {path1['name']} to {path2['name']} successfully"
                show_info(msg)
                
                # Set up for modifying this path
                self.start_point = path1['start'].copy()
                self.end_point = path2['end'].copy()
                self.start_points_layer.data = np.array([self.start_point])
                self.end_points_layer.data = np.array([self.end_point])
                
                # Update status
                self.start_status.setText(f"Status: Start point set at {np.array2string(self.start_point, precision=0)}")
                self.end_status.setText(f"Status: End point set at {np.array2string(self.end_point, precision=0)}")
                self.path_info.setText(f"Path: {combined_name} with {len(combined_path)} points")
                
                # Store current path ID
                self.current_path_id = path_id
                
                # Enable button
                self.trace_another_btn.setEnabled(True)
                
                # Clear any error messages
                self.error_status.setText("")
            else:
                error_msg = f"Could not find a path connecting {path1['name']} to {path2['name']}"
                show_info(error_msg)
                self.error_status.setText(error_msg)
        except Exception as e:
            error_msg = f"Error connecting paths: {str(e)}"
            show_info(error_msg)
            self.error_status.setText(error_msg)
            print(f"Error details: {str(e)}")
        finally:
            self.handling_event = False
    
    def export_all_paths(self):
        """Export all paths to a file"""
        if not self.paths:
            show_info("No paths to export")
            return
        
        # Get path to save file
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save All Paths", "", "NumPy Files (*.npz)"
        )
        
        if not filepath:
            return
        
        try:
            # Prepare data for export
            path_data = {}
            for path_id, path_info in self.paths.items():
                path_data[path_info['name']] = {
                    'points': path_info['data'],
                    'start': path_info['start'] if 'start' in path_info and path_info['start'] is not None else np.array([]),
                    'end': path_info['end'] if 'end' in path_info and path_info['end'] is not None else np.array([])
                }
            
            # Save as NumPy archive
            np.savez(filepath, paths=path_data)
            
            show_info(f"All paths saved to {filepath}")
            
        except Exception as e:
            show_info(f"Error saving paths: {e}")


def run_interactive_path_finder(image):
    """
    Launch the interactive brightest path finder
    
    Parameters:
    -----------
    image : numpy.ndarray
        3D or higher-dimensional image data
    """
    # Create a viewer
    viewer = napari.Viewer()
    
    # Create and add our widget
    path_widget = BrightestPathWidget(viewer, image)
    viewer.window.add_dock_widget(
        path_widget, name="Brightest Path Finder", area="right"
    )
    
    # Set initial view
    if image.ndim >= 3:
        # Start with a mid-slice view for 3D+ images
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
    
    # Set up proper cleanup when closed
    # Try different approaches to ensure it works across napari versions
    try:
        # For newer Qt versions
        viewer.window._qt_window.closing.connect(lambda: sys.exit(0))
    except (AttributeError, TypeError):
        try:
            # For older Qt versions
            viewer.window._qt_window.destroyed.connect(lambda: sys.exit(0))
        except (AttributeError, TypeError):
            # Fallback approach
            from qtpy.QtCore import QTimer
            def check_if_closed():
                if not viewer.window._qt_window.isVisible():
                    sys.exit(0)
            # Check periodically if window is still visible
            timer = QTimer()
            timer.timeout.connect(check_if_closed)
            timer.start(1000)  # Check every second
    
    # Make sure the viewer doesn't block when closed and properly exits
    from qtpy.QtCore import Qt
    try:
        viewer.window._qt_window.setAttribute(Qt.WA_DeleteOnClose)
    except (AttributeError, TypeError):
        pass
    
    # Add a direct cleanup option via a button
    class CleanupWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            self.setLayout(layout)
            
            quit_btn = QPushButton("Exit Application")
            quit_btn.clicked.connect(lambda: sys.exit(0))
            layout.addWidget(quit_btn)
    
    # Add the cleanup widget to allow manual exit
    cleanup_widget = CleanupWidget()
    viewer.window.add_dock_widget(cleanup_widget, name="Exit", area="bottom")
    
    print("\n===== BRIGHTEST PATH FINDER =====")
    print("1. Click on the image to set start and end points")
    print("2. Click 'Find Path' to calculate and save the path")
    print("3. Use 'Retrace Path' to try a different path between the same points")
    print("4. Use 'Trace Another Path' to start a new path")
    print("5. In Path Management, select two paths and click 'Connect Selected Paths'")
    print("   to create a path from the start of first path to end of second path")
    print("===============================\n")
    
    return viewer

# Example usage:
if __name__ == "__main__":
    # Assuming 'image' is already loaded
    benchmark = np.asarray(io.imread(r'../../DeepD3_Benchmark.tif'))
    
    viewer = run_interactive_path_finder(benchmark)
    napari.run()  # Start the Napari event loop