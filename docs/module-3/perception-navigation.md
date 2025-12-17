---
sidebar_position: 4
---

# Perception and Navigation for Humanoid Robots: Cognitive Systems and Spatial Reasoning

## Overview

Perception and navigation form the cognitive foundation of humanoid robotics, enabling robots to understand their environment and move safely through complex spaces. Unlike wheeled or tracked robots, humanoid robots face unique challenges in perception and navigation due to their complex morphology, dynamic balance requirements, and human-like interaction patterns. This chapter explores the specialized perception and navigation systems designed for humanoid robots, including computer vision algorithms, sensor fusion techniques, and path planning strategies optimized for bipedal locomotion.

The integration of perception and navigation systems in humanoid robots requires sophisticated algorithms that can handle the inherent instability of bipedal walking, the need for real-time processing, and the complexity of human-scale environments. These systems must operate reliably in dynamic environments while maintaining the robot's balance and ensuring safe interaction with humans and obstacles.

## Learning Objectives

By the end of this section, you should be able to:
- Implement computer vision algorithms for humanoid robot perception
- Design navigation systems suitable for humanoid robot morphology
- Integrate sensor fusion for robust environment understanding
- Apply path planning algorithms optimized for humanoid locomotion

## Introduction to Humanoid Robot Perception

### Unique Challenges in Humanoid Perception

Humanoid robots face distinct perception challenges compared to other robot types:

- **Dynamic Perspective**: The robot's head moves during walking, affecting visual processing
- **Balance Constraints**: Perception must account for robot's stability during locomotion
- **Human-Scale Interactions**: Perception optimized for human-sized objects and environments
- **Multi-Modal Integration**: Combining various sensor modalities for robust understanding

```python
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import sensor_msgs.point_cloud2 as pc2

class HumanoidPerceptionSystem:
    def __init__(self):
        # Perception components
        self.visual_processor = VisualProcessor()
        self.depth_processor = DepthProcessor()
        self.sensor_fusion = SensorFusionModule()
        self.object_detector = ObjectDetectionModule()
        self.human_detector = HumanDetectionModule()

        # State variables
        self.robot_pose = None
        self.balance_state = None
        self.environment_map = None

    def process_perception_data(self, sensor_data):
        """Process multi-modal sensor data for humanoid perception"""
        # Process visual data
        visual_features = self.visual_processor.process(sensor_data['image'])

        # Process depth data
        depth_features = self.depth_processor.process(sensor_data['depth'])

        # Detect objects
        objects = self.object_detector.detect(
            sensor_data['image'],
            sensor_data['depth']
        )

        # Detect humans
        humans = self.human_detector.detect(sensor_data['image'])

        # Fuse sensor data
        fused_data = self.sensor_fusion.fuse(
            visual_features,
            depth_features,
            objects,
            humans
        )

        return fused_data

    def update_environment_map(self, fused_data):
        """Update environment representation for navigation"""
        # Create or update occupancy grid
        self.environment_map = self.create_occupancy_grid(fused_data)

        # Update object positions
        self.update_object_positions(fused_data)

        return self.environment_map
```

### Visual Processing for Humanoid Robots

```python
class VisualProcessor:
    def __init__(self):
        # Feature extraction
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()

        # Color space processing
        self.color_spaces = {
            'hsv': cv2.COLOR_BGR2HSV,
            'lab': cv2.COLOR_BGR2LAB,
            'gray': cv2.COLOR_BGR2GRAY
        }

        # Visual tracking
        self.tracker = cv2.TrackerKCF_create()

    def process(self, image_msg):
        """Process visual data from camera"""
        # Convert ROS image to OpenCV
        cv_image = self.ros_to_cv2(image_msg)

        # Extract features
        features = self.extract_features(cv_image)

        # Detect edges and contours
        edges = self.detect_edges(cv_image)
        contours = self.find_contours(edges)

        # Color-based segmentation
        color_segments = self.segment_by_color(cv_image)

        return {
            'features': features,
            'edges': edges,
            'contours': contours,
            'color_segments': color_segments
        }

    def extract_features(self, image):
        """Extract visual features using multiple algorithms"""
        # SIFT features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_sift, desc_sift = self.sift.detectAndCompute(gray, None)

        # ORB features
        kp_orb, desc_orb = self.orb.detectAndCompute(gray, None)

        return {
            'sift': {'keypoints': kp_sift, 'descriptors': desc_sift},
            'orb': {'keypoints': kp_orb, 'descriptors': desc_orb}
        }

    def detect_edges(self, image):
        """Detect edges using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def find_contours(self, edges):
        """Find contours in edge image"""
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def segment_by_color(self, image):
        """Segment image based on color properties"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges (example: red objects)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        return red_mask
```

## Depth Processing and 3D Understanding

### Depth-Based Environment Modeling

```python
class DepthProcessor:
    def __init__(self):
        # Depth processing parameters
        self.min_depth = 0.1  # meters
        self.max_depth = 10.0  # meters
        self.depth_threshold = 0.02  # 2cm threshold

    def process(self, depth_msg):
        """Process depth data for 3D understanding"""
        # Convert depth message to numpy array
        depth_image = self.ros_depth_to_numpy(depth_msg)

        # Create point cloud from depth
        point_cloud = self.depth_to_pointcloud(depth_image)

        # Segment ground plane
        ground_plane = self.segment_ground_plane(point_cloud)

        # Detect obstacles
        obstacles = self.detect_obstacles(point_cloud, ground_plane)

        # Calculate surface normals
        normals = self.calculate_surface_normals(point_cloud)

        return {
            'point_cloud': point_cloud,
            'ground_plane': ground_plane,
            'obstacles': obstacles,
            'normals': normals
        }

    def depth_to_pointcloud(self, depth_image):
        """Convert depth image to 3D point cloud"""
        height, width = depth_image.shape

        # Camera intrinsic parameters (example values)
        fx, fy = 525.0, 525.0  # focal length
        cx, cy = 319.5, 239.5  # principal point

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )

        # Convert to 3D coordinates
        x_3d = (x_coords - cx) * depth_image / fx
        y_3d = (y_coords - cy) * depth_image / fy
        z_3d = depth_image

        # Stack coordinates
        points = np.stack([x_3d, y_3d, z_3d], axis=-1)

        # Filter valid points (remove invalid depth values)
        valid_mask = (depth_image > self.min_depth) & \
                    (depth_image < self.max_depth) & \
                    ~np.isnan(depth_image)

        valid_points = points[valid_mask]

        return valid_points

    def segment_ground_plane(self, point_cloud):
        """Segment ground plane using RANSAC algorithm"""
        from sklearn.linear_model import RANSACRegressor

        if len(point_cloud) < 100:
            return np.array([]), np.array([])

        # Prepare data for RANSAC
        X = point_cloud[:, [0, 1]]  # x, y coordinates
        y = point_cloud[:, 2]       # z coordinate (height)

        # Apply RANSAC
        ransac = RANSACRegressor(
            min_samples=50,
            residual_threshold=0.05,  # 5cm threshold
            max_trials=1000
        )

        ransac.fit(X, y)

        # Identify inliers (ground points)
        inlier_mask = ransac.inlier_mask_
        ground_points = point_cloud[inlier_mask]
        obstacle_points = point_cloud[~inlier_mask]

        return ground_points, obstacle_points

    def detect_obstacles(self, point_cloud, ground_plane):
        """Detect obstacles above ground plane"""
        ground_points, obstacle_points = ground_plane

        # Group obstacle points into clusters
        obstacles = self.cluster_obstacles(obstacle_points)

        return obstacles

    def cluster_obstacles(self, points):
        """Cluster obstacle points into individual obstacles"""
        from sklearn.cluster import DBSCAN

        if len(points) == 0:
            return []

        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.2, min_samples=10)  # 20cm eps, 10 min points
        labels = clustering.fit_predict(points[:, :2])  # Use x,y coordinates

        obstacles = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue

            cluster_points = points[labels == label]
            obstacle = self.create_obstacle_from_cluster(cluster_points)
            obstacles.append(obstacle)

        return obstacles

    def create_obstacle_from_cluster(self, points):
        """Create obstacle representation from point cluster"""
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Calculate center and dimensions
        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords

        return {
            'center': center,
            'dimensions': dimensions,
            'points': points,
            'type': self.classify_obstacle_type(dimensions)
        }
```

## Object Detection and Recognition

### Multi-Modal Object Detection

```python
class ObjectDetectionModule:
    def __init__(self):
        # Initialize deep learning models
        self.detection_model = self.load_detection_model()
        self.classification_model = self.load_classification_model()

        # Confidence thresholds
        self.detection_threshold = 0.5
        self.classification_threshold = 0.7

    def detect(self, image, depth=None):
        """Detect objects in image with optional depth information"""
        # Run object detection
        detections = self.run_detection(image)

        # Filter by confidence
        confident_detections = [
            det for det in detections
            if det['confidence'] > self.detection_threshold
        ]

        # Add 3D information if depth is available
        if depth is not None:
            confident_detections = self.add_3d_information(
                confident_detections,
                image,
                depth
            )

        # Classify objects
        classified_detections = self.classify_objects(confident_detections)

        return classified_detections

    def run_detection(self, image):
        """Run object detection on image"""
        # This would use a pre-trained model like YOLO, SSD, or similar
        # For demonstration, we'll simulate detection results

        # Convert image for model input
        input_tensor = self.preprocess_image(image)

        # Run inference (simulated)
        # detections = self.detection_model(input_tensor)

        # Simulate detection results
        detections = [
            {
                'bbox': [100, 100, 200, 200],  # [x, y, width, height]
                'class': 'chair',
                'confidence': 0.85
            },
            {
                'bbox': [300, 150, 150, 150],
                'class': 'table',
                'confidence': 0.92
            }
        ]

        return detections

    def add_3d_information(self, detections, image, depth):
        """Add 3D information to detections using depth data"""
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox

            # Extract depth region corresponding to detection
            depth_region = depth[y:y+h, x:x+w]

            # Calculate distance to object
            valid_depths = depth_region[depth_region > 0]
            if len(valid_depths) > 0:
                avg_distance = np.mean(valid_depths)
                min_distance = np.min(valid_depths)
            else:
                avg_distance = float('inf')
                min_distance = float('inf')

            # Calculate 3D position (simplified)
            center_x = x + w // 2
            center_y = y + h // 2
            depth_center = depth[center_y, center_x]

            # Convert to 3D coordinates (simplified)
            # This would use actual camera intrinsics in practice
            detection['distance'] = avg_distance
            detection['min_distance'] = min_distance
            detection['position_3d'] = self.pixel_to_3d(
                center_x, center_y, depth_center
            )

        return detections

    def classify_objects(self, detections):
        """Classify detected objects"""
        for detection in detections:
            # This would run a classification model on the detection
            # For now, we'll just verify the class from detection
            detection['class_confidence'] = detection['confidence']

        return detections

class HumanDetectionModule:
    def __init__(self):
        # Human detection model
        self.human_detector = self.load_human_detector()
        self.pose_estimator = self.load_pose_estimator()

    def detect(self, image):
        """Detect humans in image"""
        # Run human detection
        humans = self.run_human_detection(image)

        # Estimate poses
        for human in humans:
            human['pose'] = self.estimate_pose(
                image,
                human['bbox']
            )

        return humans

    def run_human_detection(self, image):
        """Run human detection on image"""
        # This would use a human detection model
        # For demonstration, simulate detection
        humans = [
            {
                'bbox': [50, 100, 80, 200],  # [x, y, width, height]
                'confidence': 0.95,
                'distance': self.estimate_distance([50, 100, 80, 200])
            }
        ]

        return humans

    def estimate_pose(self, image, bbox):
        """Estimate human pose in bounding box"""
        # This would use a pose estimation model
        # For now, return a simple representation
        return {
            'head_position': [bbox[0] + bbox[2]//2, bbox[1]],  # Top center
            'shoulder_width': bbox[2] * 0.7,
            'estimated_height': bbox[3]
        }
```

## Sensor Fusion for Robust Perception

### Multi-Sensor Integration

```python
class SensorFusionModule:
    def __init__(self):
        # Fusion weights for different sensors
        self.fusion_weights = {
            'camera': 0.4,
            'lidar': 0.3,
            'imu': 0.2,
            'other': 0.1
        }

        # Data association
        self.association_threshold = 0.3  # meters

    def fuse(self, visual_data, depth_data, objects, humans):
        """Fuse data from multiple sensors"""
        # Create unified representation
        fused_environment = {
            'objects': self.fuse_objects(objects),
            'humans': self.fuse_humans(humans),
            'obstacles': self.fuse_obstacles_from_depth(depth_data['obstacles']),
            'ground_plane': depth_data['ground_plane'],
            'features': self.fuse_features(visual_data['features'])
        }

        # Update uncertainty estimates
        fused_environment = self.calculate_uncertainty(fused_environment)

        return fused_environment

    def fuse_objects(self, objects):
        """Fuse object detections from multiple sources"""
        if not objects:
            return []

        # Group objects by spatial proximity
        fused_objects = []
        processed_indices = set()

        for i, obj1 in enumerate(objects):
            if i in processed_indices:
                continue

            # Find nearby objects to merge
            nearby_objects = [obj1]
            nearby_indices = [i]

            for j, obj2 in enumerate(objects[i+1:], i+1):
                if j in processed_indices:
                    continue

                # Calculate distance between objects
                if self.objects_close(obj1, obj2):
                    nearby_objects.append(obj2)
                    nearby_indices.append(j)

            # Merge nearby objects
            merged_object = self.merge_objects(nearby_objects)
            fused_objects.append(merged_object)

            # Mark as processed
            processed_indices.update(nearby_indices)

        return fused_objects

    def objects_close(self, obj1, obj2):
        """Check if two objects are spatially close"""
        if 'position_3d' in obj1 and 'position_3d' in obj2:
            pos1 = np.array(obj1['position_3d'])
            pos2 = np.array(obj2['position_3d'])
            distance = np.linalg.norm(pos1 - pos2)
            return distance < self.association_threshold

        # If no 3D position, use 2D overlap for now
        return self.bboxes_overlap(obj1['bbox'], obj2['bbox'])

    def merge_objects(self, objects):
        """Merge multiple object detections"""
        if len(objects) == 1:
            return objects[0]

        # Average positions and adjust confidence
        avg_position = np.mean([
            obj.get('position_3d', [0, 0, 0])
            for obj in objects
        ], axis=0)

        avg_confidence = np.mean([obj['confidence'] for obj in objects])

        # Determine most common class
        classes = [obj['class'] for obj in objects]
        most_common_class = max(set(classes), key=classes.count)

        merged_object = objects[0].copy()
        merged_object['position_3d'] = avg_position
        merged_object['confidence'] = avg_confidence
        merged_object['class'] = most_common_class
        merged_object['fused_count'] = len(objects)

        return merged_object

    def calculate_uncertainty(self, environment):
        """Calculate uncertainty estimates for fused data"""
        for obj in environment['objects']:
            # Uncertainty based on fusion confidence
            obj['uncertainty'] = 1.0 - obj['confidence']

            # Add distance-based uncertainty
            if 'distance' in obj:
                obj['uncertainty'] += min(obj['distance'] * 0.01, 0.3)

        return environment
```

## Humanoid Robot Navigation Systems

### Navigation Architecture for Bipedal Robots

```python
import heapq
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header

class HumanoidNavigationSystem:
    def __init__(self):
        # Navigation components
        self.global_planner = GlobalPathPlanner()
        self.local_planner = LocalPathPlanner()
        self.controller = HumanoidMotionController()

        # State variables
        self.current_pose = Pose()
        self.goal_pose = Pose()
        self.path = Path()
        self.obstacles = []

        # Navigation parameters
        self.navigation_state = 'IDLE'  # IDLE, PLANNING, EXECUTING, RECOVERY
        self.recovery_behavior = 'CLEAR_COSTMAPS'

    def navigate_to_goal(self, goal_pose):
        """Navigate humanoid robot to specified goal"""
        self.goal_pose = goal_pose

        # Plan global path
        global_path = self.global_planner.plan(
            self.current_pose,
            goal_pose,
            self.obstacles
        )

        if not global_path:
            return False, "Global path planning failed"

        # Execute navigation
        self.navigation_state = 'EXECUTING'

        # Follow path with local planning and obstacle avoidance
        success = self.follow_path(global_path)

        if success:
            return True, "Navigation completed successfully"
        else:
            return False, "Navigation failed"

    def follow_path(self, global_path):
        """Follow global path with local obstacle avoidance"""
        path_index = 0

        while path_index < len(global_path.poses) and self.navigation_state == 'EXECUTING':
            # Get current goal from global path
            current_goal = global_path.poses[path_index]

            # Plan local path to current goal
            local_path = self.local_planner.plan(
                self.current_pose,
                current_goal,
                self.obstacles
            )

            if not local_path:
                # Try to recover from local planning failure
                recovery_success = self.recovery_behavior()
                if not recovery_success:
                    return False

            # Execute local path
            execution_success = self.controller.follow_path(local_path)

            if not execution_success:
                # Local obstacle avoidance
                avoidance_path = self.local_planner.avoid_obstacles(
                    self.current_pose,
                    current_goal,
                    self.obstacles
                )

                if avoidance_path:
                    execution_success = self.controller.follow_path(avoidance_path)

            if execution_success:
                path_index += 1
            else:
                return False

        return True

    def recovery_behavior(self):
        """Execute recovery behavior when navigation fails"""
        if self.recovery_behavior == 'CLEAR_COSTMAPS':
            # Clear costmaps to remove temporary obstacles
            self.clear_costmaps()
            return True
        elif self.recovery_behavior == 'ROTATE_IN_PLACE':
            # Rotate to find alternative path
            return self.controller.rotate_in_place()
        elif self.recovery_behavior == 'BACKUP':
            # Back up and try alternative path
            return self.controller.backup_and_replan()

        return False

class GlobalPathPlanner:
    def __init__(self):
        # Global planner parameters
        self.planner_type = 'A_STAR'  # A_STAR, D_STAR, RRT, etc.
        self.costmap_resolution = 0.05  # 5cm resolution
        self.planning_frequency = 1.0  # Hz

    def plan(self, start_pose, goal_pose, obstacles):
        """Plan global path from start to goal"""
        if self.planner_type == 'A_STAR':
            return self.a_star_plan(start_pose, goal_pose, obstacles)
        elif self.planner_type == 'D_STAR':
            return self.d_star_plan(start_pose, goal_pose, obstacles)
        else:
            raise ValueError(f"Unknown planner type: {self.planner_type}")

    def a_star_plan(self, start_pose, goal_pose, obstacles):
        """A* path planning algorithm"""
        # Convert poses to grid coordinates
        start_grid = self.pose_to_grid(start_pose)
        goal_grid = self.pose_to_grid(goal_pose)

        # Create costmap
        costmap = self.create_costmap(obstacles)

        # Implement A* algorithm
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = self.reconstruct_path(came_from, current)
                return self.grid_path_to_ros_path(path)

            for neighbor in self.get_neighbors(current, costmap):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def create_costmap(self, obstacles):
        """Create costmap from obstacles"""
        # This would create a 2D costmap representing the environment
        # Implementation details omitted for brevity
        pass

    def heuristic(self, pos1, pos2):
        """Heuristic function for A* (Euclidean distance)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_neighbors(self, pos, costmap):
        """Get valid neighbors for path planning"""
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (pos[0] + dx, pos[1] + dy)

            # Check if neighbor is valid (not in obstacle and within bounds)
            if self.is_valid_position(neighbor, costmap):
                neighbors.append(neighbor)

        return neighbors

class LocalPathPlanner:
    def __init__(self):
        # Local planner parameters
        self.local_radius = 3.0  # meters
        self.control_frequency = 10.0  # Hz
        self.max_vel_x = 0.5  # m/s
        self.max_vel_theta = 1.0  # rad/s

    def plan(self, current_pose, goal_pose, obstacles):
        """Plan local path with obstacle avoidance"""
        # Use Dynamic Window Approach (DWA) or similar
        return self.dwa_plan(current_pose, goal_pose, obstacles)

    def dwa_plan(self, current_pose, goal_pose, obstacles):
        """Dynamic Window Approach for local path planning"""
        # Calculate dynamic window
        vs = self.calculate_velocity_space()
        vd = self.calculate_dynamic_window(current_pose)

        # Evaluate trajectories
        best_trajectory = None
        best_score = float('-inf')

        for vel_x in np.linspace(vd[0], vd[1], 10):  # Linear velocity
            for vel_theta in np.linspace(vd[2], vd[3], 10):  # Angular velocity
                trajectory = self.predict_trajectory(
                    current_pose,
                    vel_x,
                    vel_theta
                )

                score = self.evaluate_trajectory(
                    trajectory,
                    goal_pose,
                    obstacles
                )

                if score > best_score:
                    best_score = score
                    best_trajectory = trajectory

        return best_trajectory

    def calculate_dynamic_window(self, current_pose):
        """Calculate dynamic window based on robot constraints"""
        # This would calculate the feasible velocity space
        # based on robot dynamics and constraints
        pass

    def predict_trajectory(self, start_pose, vel_x, vel_theta, dt=0.1, steps=10):
        """Predict trajectory given velocity commands"""
        trajectory = []
        current_pose = start_pose

        for _ in range(steps):
            # Simple kinematic model for prediction
            new_x = current_pose.position.x + vel_x * dt * np.cos(current_pose.orientation.z)
            new_y = current_pose.position.y + vel_x * dt * np.sin(current_pose.orientation.z)
            new_theta = current_pose.orientation.z + vel_theta * dt

            pose = Pose()
            pose.position.x = new_x
            pose.position.y = new_y
            pose.orientation.z = new_theta

            trajectory.append(pose)

            # Update for next step
            current_pose = pose

        return trajectory

    def evaluate_trajectory(self, trajectory, goal_pose, obstacles):
        """Evaluate trajectory based on multiple criteria"""
        # Calculate scores for different criteria
        goal_score = self.calculate_goal_score(trajectory[-1], goal_pose)
        obs_score = self.calculate_obstacle_score(trajectory, obstacles)
        speed_score = self.calculate_speed_score(trajectory)

        # Weighted combination of scores
        total_score = (0.4 * goal_score +
                      0.4 * obs_score +
                      0.2 * speed_score)

        return total_score
```

## Humanoid Motion Control

### Bipedal Locomotion Control

```python
class HumanoidMotionController:
    def __init__(self):
        # Control parameters
        self.step_height = 0.1  # meters
        self.step_length = 0.3  # meters
        self.walk_frequency = 1.0  # Hz
        self.balance_threshold = 0.05  # meters (CoM deviation)

        # Control systems
        self.balance_controller = BalanceController()
        self.step_controller = StepController()
        self.foot_placement_controller = FootPlacementController()

    def follow_path(self, path):
        """Follow a path with bipedal locomotion"""
        for i in range(len(path.poses) - 1):
            current_pose = path.poses[i]
            next_pose = path.poses[i + 1]

            # Calculate direction and distance to next waypoint
            dx = next_pose.position.x - current_pose.position.x
            dy = next_pose.position.y - current_pose.position.y
            distance = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)

            # Walk to next waypoint
            success = self.walk_to_waypoint(
                current_pose,
                next_pose,
                direction,
                distance
            )

            if not success:
                return False

        return True

    def walk_to_waypoint(self, start_pose, target_pose, direction, distance):
        """Walk to a specific waypoint"""
        # Calculate number of steps needed
        num_steps = int(distance / self.step_length) + 1

        for step in range(num_steps):
            # Calculate target position for this step
            step_distance = min(self.step_length, distance - step * self.step_length)
            target_x = start_pose.position.x + step_distance * np.cos(direction)
            target_y = start_pose.position.y + step_distance * np.sin(direction)

            target_pose = Pose()
            target_pose.position.x = target_x
            target_pose.position.y = target_y
            target_pose.orientation = start_pose.orientation

            # Execute single step
            success = self.execute_step(target_pose)

            if not success:
                return False

        return True

    def execute_step(self, target_pose):
        """Execute a single walking step"""
        # Plan step trajectory
        step_trajectory = self.plan_step_trajectory(target_pose)

        # Execute step while maintaining balance
        success = self.balance_controller.maintain_balance_during_step(
            step_trajectory
        )

        if success:
            # Update robot's actual pose after successful step
            self.update_robot_pose(target_pose)

        return success

class BalanceController:
    def __init__(self):
        # Balance control parameters
        self.zmp_controller = ZMPController()
        self.com_controller = COMController()
        self.ankle_controller = AnkleController()

    def maintain_balance_during_step(self, step_trajectory):
        """Maintain balance while executing a step"""
        # Calculate Zero Moment Point (ZMP) trajectory
        zmp_trajectory = self.zmp_controller.calculate_zmp_trajectory(
            step_trajectory
        )

        # Control Center of Mass (CoM)
        com_trajectory = self.com_controller.calculate_com_trajectory(
            zmp_trajectory
        )

        # Control ankle torques to maintain balance
        ankle_torques = self.ankle_controller.calculate_ankle_torques(
            com_trajectory
        )

        # Execute balance control
        success = self.execute_balance_control(
            zmp_trajectory,
            com_trajectory,
            ankle_torques
        )

        return success

class ZMPController:
    def __init__(self):
        # ZMP control parameters
        self.zmp_margin = 0.05  # 5cm safety margin
        self.zmp_frequency = 100.0  # Hz control frequency

    def calculate_zmp_trajectory(self, step_trajectory):
        """Calculate ZMP trajectory for stable walking"""
        # Implement ZMP-based balance control
        # This would calculate the desired ZMP trajectory
        # to maintain balance during the step
        pass

class COMController:
    def __init__(self):
        # CoM control parameters
        self.com_height = 0.8  # meters (default CoM height)
        self.com_frequency = 100.0  # Hz

    def calculate_com_trajectory(self, zmp_trajectory):
        """Calculate CoM trajectory from ZMP trajectory"""
        # Use inverted pendulum model to calculate CoM trajectory
        # from desired ZMP trajectory
        pass

class AnkleController:
    def __init__(self):
        # Ankle control parameters
        self.ankle_stiffness = 100.0  # N*m/rad
        self.ankle_damping = 10.0    # N*m*s/rad

    def calculate_ankle_torques(self, com_trajectory):
        """Calculate ankle torques for balance"""
        # Calculate required ankle torques to maintain balance
        # based on CoM trajectory and current state
        pass
```

## Integration with ROS 2 Navigation Stack

### Nav2 Integration for Humanoid Robots

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import TransformListener, Buffer
import numpy as np

class HumanoidNav2Interface(Node):
    def __init__(self):
        super().__init__('humanoid_nav2_interface')

        # Initialize ROS 2 components
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers and subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # State variables
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.laser_data = None
        self.imu_data = None

    def navigate_to_pose(self, goal_pose):
        """Navigate to specified pose using Nav2"""
        # Create navigation goal
        goal = NavigateToPose.Goal()
        goal.pose = goal_pose

        # Wait for action server
        self.nav_client.wait_for_server()

        # Send navigation goal
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.navigation_done_callback)

        return future

    def navigation_done_callback(self, future):
        """Handle navigation completion"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation completed: {result}')

    def odom_callback(self, msg):
        """Handle odometry data"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg
        # Process laser data for obstacle detection
        self.process_laser_obstacles(msg)

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg
        # Process IMU data for balance estimation
        self.process_imu_balance(msg)

    def process_laser_obstacles(self, laser_msg):
        """Process laser scan for obstacle detection"""
        # Convert laser scan to obstacle information
        ranges = np.array(laser_msg.ranges)
        angles = np.linspace(
            laser_msg.angle_min,
            laser_msg.angle_max,
            len(ranges)
        )

        # Detect obstacles
        obstacle_ranges = ranges[ranges < 2.0]  # Within 2m
        obstacle_angles = angles[ranges < 2.0]

        # Create obstacle representation
        obstacles = []
        for r, theta in zip(obstacle_ranges, obstacle_angles):
            if not np.isnan(r) and r > laser_msg.range_min:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                obstacles.append((x, y))

        return obstacles

    def process_imu_balance(self, imu_msg):
        """Process IMU data for balance estimation"""
        # Extract orientation from IMU
        orientation = imu_msg.orientation

        # Convert quaternion to roll/pitch
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )

        # Check balance thresholds
        max_lean = 0.1  # 5.7 degrees
        if abs(roll) > max_lean or abs(pitch) > max_lean:
            self.get_logger().warn('Balance threshold exceeded')
            # Trigger balance recovery
            self.trigger_balance_recovery()

    def trigger_balance_recovery(self):
        """Trigger balance recovery behavior"""
        # Stop navigation and execute balance recovery
        self.nav_client.cancel_goal_async()
        # Implement balance recovery logic
        pass

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
```

## Perception-Action Integration

### Closed-Loop Perception-Action Systems

```python
class PerceptionActionSystem:
    def __init__(self):
        # Perception system
        self.perception = HumanoidPerceptionSystem()

        # Navigation system
        self.navigation = HumanoidNavigationSystem()

        # Control system
        self.controller = HumanoidMotionController()

        # Integration parameters
        self.perception_frequency = 10.0  # Hz
        self.control_frequency = 100.0   # Hz
        self.integration_delay = 0.1     # seconds

        # Action selection
        self.action_selector = ActionSelector()

    def run_perception_action_loop(self):
        """Run closed-loop perception-action system"""
        while True:
            # Sense environment
            sensor_data = self.acquire_sensor_data()

            # Process perception
            environment_state = self.perception.process_perception_data(sensor_data)

            # Update navigation map
            self.navigation.update_environment_map(environment_state)

            # Select appropriate action
            action = self.action_selector.select_action(
                environment_state,
                self.navigation.get_current_state()
            )

            # Execute action
            self.execute_action(action)

            # Wait for next iteration
            time.sleep(1.0 / self.perception_frequency)

    def acquire_sensor_data(self):
        """Acquire data from all sensors"""
        # This would acquire data from ROS topics
        sensor_data = {
            'image': self.get_camera_data(),
            'depth': self.get_depth_data(),
            'lidar': self.get_lidar_data(),
            'imu': self.get_imu_data(),
            'odom': self.get_odom_data()
        }

        return sensor_data

    def execute_action(self, action):
        """Execute selected action"""
        if action['type'] == 'navigate':
            goal = action['goal']
            self.navigation.navigate_to_goal(goal)
        elif action['type'] == 'avoid':
            direction = action['direction']
            self.controller.avoid_obstacle(direction)
        elif action['type'] == 'interact':
            object_id = action['object_id']
            self.controller.interact_with_object(object_id)
        elif action['type'] == 'wait':
            duration = action['duration']
            time.sleep(duration)

class ActionSelector:
    def __init__(self):
        # Action selection parameters
        self.threat_threshold = 0.5  # distance threshold for threats
        self.goal_bias = 0.8         # bias toward goal-directed actions

    def select_action(self, environment_state, robot_state):
        """Select appropriate action based on environment and robot state"""
        # Analyze environment for threats
        threats = self.identify_threats(environment_state)

        if threats:
            # Handle threats first
            return self.select_threat_response(threats, robot_state)

        # Check if goal is reached
        if self.goal_reached(robot_state):
            return {'type': 'wait', 'duration': 1.0}

        # Navigate toward goal
        goal = robot_state.get('goal', None)
        if goal:
            return {'type': 'navigate', 'goal': goal}

        # Default: wait
        return {'type': 'wait', 'duration': 0.1}

    def identify_threats(self, environment_state):
        """Identify potential threats in environment"""
        threats = []

        # Check for close obstacles
        for obj in environment_state.get('objects', []):
            if obj.get('distance', float('inf')) < self.threat_threshold:
                threats.append(obj)

        # Check for humans in path
        for human in environment_state.get('humans', []):
            if human.get('distance', float('inf')) < self.threat_threshold * 1.5:
                threats.append(human)

        return threats

    def select_threat_response(self, threats, robot_state):
        """Select response to identified threats"""
        # Find closest threat
        closest_threat = min(threats, key=lambda x: x.get('distance', float('inf')))

        # Determine avoidance direction
        threat_pos = closest_threat.get('position_3d', [0, 0, 0])
        robot_pos = robot_state.get('position', [0, 0, 0])

        # Calculate direction to avoid threat
        avoidance_vector = np.array(robot_pos) - np.array(threat_pos)
        avoidance_direction = avoidance_vector / np.linalg.norm(avoidance_vector)

        return {
            'type': 'avoid',
            'direction': avoidance_direction.tolist()
        }
```

## Performance Optimization

### Real-Time Perception Optimization

```python
import threading
import queue
from collections import deque

class OptimizedPerceptionSystem:
    def __init__(self):
        # Multi-threaded processing
        self.processing_threads = []
        self.data_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)

        # Processing pipelines
        self.visual_pipeline = VisualProcessingPipeline()
        self.depth_pipeline = DepthProcessingPipeline()
        self.fusion_pipeline = FusionPipeline()

        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)

        # Adaptive processing
        self.processing_quality = 'high'  # high, medium, low
        self.target_fps = 30.0

    def start_processing_pipeline(self):
        """Start multi-threaded processing pipeline"""
        # Start visual processing thread
        visual_thread = threading.Thread(
            target=self.visual_pipeline.run,
            args=(self.data_queue, self.result_queue)
        )
        visual_thread.start()
        self.processing_threads.append(visual_thread)

        # Start depth processing thread
        depth_thread = threading.Thread(
            target=self.depth_pipeline.run,
            args=(self.data_queue, self.result_queue)
        )
        depth_thread.start()
        self.processing_threads.append(depth_thread)

        # Start fusion thread
        fusion_thread = threading.Thread(
            target=self.fusion_pipeline.run,
            args=(self.result_queue, self.final_output)
        )
        fusion_thread.start()
        self.processing_threads.append(fusion_thread)

    def adaptive_processing(self, current_performance):
        """Adjust processing quality based on performance"""
        avg_frame_time = np.mean(list(self.frame_times))
        target_frame_time = 1.0 / self.target_fps

        if avg_frame_time > target_frame_time * 1.2:  # 20% over target
            # Reduce processing quality
            if self.processing_quality == 'high':
                self.processing_quality = 'medium'
                self.reduce_processing_quality()
            elif self.processing_quality == 'medium':
                self.processing_quality = 'low'
                self.reduce_processing_quality()
        elif avg_frame_time < target_frame_time * 0.8:  # 20% under target
            # Increase processing quality
            if self.processing_quality == 'low':
                self.processing_quality = 'medium'
                self.increase_processing_quality()
            elif self.processing_quality == 'medium':
                self.processing_quality = 'high'
                self.increase_processing_quality()

    def reduce_processing_quality(self):
        """Reduce processing quality to improve performance"""
        # Reduce image resolution
        self.visual_pipeline.set_resolution_scale(0.75)

        # Reduce detection confidence thresholds
        self.visual_pipeline.set_confidence_threshold(0.6)

        # Simplify point cloud processing
        self.depth_pipeline.set_point_cloud_decimation(0.1)  # 10cm resolution

    def increase_processing_quality(self):
        """Increase processing quality"""
        # Increase image resolution
        self.visual_pipeline.set_resolution_scale(1.0)

        # Increase detection confidence thresholds
        self.visual_pipeline.set_confidence_threshold(0.7)

        # Improve point cloud processing
        self.depth_pipeline.set_point_cloud_decimation(0.05)  # 5cm resolution

class VisualProcessingPipeline:
    def __init__(self):
        self.resolution_scale = 1.0
        self.confidence_threshold = 0.7
        self.max_objects = 20

    def run(self, input_queue, output_queue):
        """Run visual processing pipeline"""
        while True:
            try:
                # Get input data
                data = input_queue.get(timeout=1.0)

                # Process visual data
                start_time = time.time()
                result = self.process_frame(data)
                processing_time = time.time() - start_time

                # Put result in output queue
                output_queue.put({
                    'result': result,
                    'processing_time': processing_time,
                    'timestamp': time.time()
                })

            except queue.Empty:
                continue

    def process_frame(self, frame_data):
        """Process a single frame"""
        # Apply resolution scaling
        if self.resolution_scale < 1.0:
            new_size = (
                int(frame_data.shape[1] * self.resolution_scale),
                int(frame_data.shape[0] * self.resolution_scale)
            )
            frame_data = cv2.resize(frame_data, new_size)

        # Run object detection
        detections = self.run_object_detection(frame_data)

        # Filter by confidence
        high_conf_detections = [
            det for det in detections
            if det['confidence'] > self.confidence_threshold
        ]

        # Limit number of objects
        if len(high_conf_detections) > self.max_objects:
            high_conf_detections = high_conf_detections[:self.max_objects]

        return high_conf_detections
```

## Practical Applications in Humanoid Robotics

### Human-Robot Interaction Navigation

```python
class SocialNavigationSystem:
    def __init__(self):
        # Social navigation parameters
        self.personal_space_radius = 1.0  # meters
        self.social_force_coefficient = 0.5
        self.group_detection_threshold = 3  # people to form group

        # Human behavior prediction
        self.behavior_predictor = HumanBehaviorPredictor()

    def navigate_with_social_awareness(self, goal_pose, humans):
        """Navigate while considering social norms and human behavior"""
        # Calculate social forces from humans
        social_forces = self.calculate_social_forces(humans)

        # Predict human movements
        predicted_human_paths = self.behavior_predictor.predict_paths(humans)

        # Modify navigation to respect social norms
        modified_goal = self.adjust_goal_for_social_norms(
            goal_pose,
            humans,
            predicted_human_paths
        )

        # Plan path considering social forces
        path = self.plan_socially_aware_path(
            self.current_pose,
            modified_goal,
            humans,
            social_forces
        )

        return path

    def calculate_social_forces(self, humans):
        """Calculate repulsive forces from humans"""
        forces = []

        for human in humans:
            # Calculate distance to human
            human_pos = np.array([
                human['position_3d'][0],
                human['position_3d'][1]
            ])
            robot_pos = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y
            ])

            distance = np.linalg.norm(human_pos - robot_pos)

            if distance < self.personal_space_radius:
                # Calculate repulsive force
                direction = robot_pos - human_pos
                direction = direction / np.linalg.norm(direction)

                # Force decreases with distance
                force_magnitude = (self.personal_space_radius - distance) * self.social_force_coefficient
                force = direction * force_magnitude

                forces.append({
                    'position': human_pos,
                    'force': force,
                    'magnitude': force_magnitude
                })

        return forces

    def adjust_goal_for_social_norms(self, goal, humans, predicted_paths):
        """Adjust goal to respect social norms"""
        adjusted_goal = goal

        # Check if goal is in personal space of humans
        for human in humans:
            human_pos = np.array([human['position_3d'][0], human['position_3d'][1]])
            goal_pos = np.array([goal.position.x, goal.position.y])

            distance = np.linalg.norm(human_pos - goal_pos)

            if distance < self.personal_space_radius:
                # Adjust goal to maintain personal space
                direction = goal_pos - human_pos
                direction = direction / np.linalg.norm(direction)

                new_goal_pos = human_pos + direction * self.personal_space_radius
                adjusted_goal.position.x = new_goal_pos[0]
                adjusted_goal.position.y = new_goal_pos[1]

        return adjusted_goal
```

## Best Practices for Humanoid Perception and Navigation

### Safety and Robustness Considerations

```python
class SafeNavigationSystem:
    def __init__(self):
        # Safety parameters
        self.safety_margin = 0.5  # meters
        self.max_navigation_speed = 0.3  # m/s
        self.emergency_stop_distance = 0.3  # meters

        # Validation systems
        self.path_validator = PathValidator()
        self.obstacle_validator = ObstacleValidator()

    def validate_navigation_plan(self, path, environment):
        """Validate navigation plan for safety"""
        # Check path clearance
        if not self.path_validator.check_clearance(path, environment, self.safety_margin):
            return False, "Path has insufficient clearance"

        # Check for dynamic obstacles
        if not self.obstacle_validator.check_dynamic_obstacles(path, environment):
            return False, "Path intersects with dynamic obstacles"

        # Check robot kinematic constraints
        if not self.check_kinematic_feasibility(path):
            return False, "Path violates kinematic constraints"

        return True, "Path is safe"

    def check_kinematic_feasibility(self, path):
        """Check if path is kinematically feasible for humanoid"""
        # Check step length constraints
        for i in range(len(path.poses) - 1):
            pose1 = path.poses[i]
            pose2 = path.poses[i + 1]

            step_distance = np.sqrt(
                (pose2.position.x - pose1.position.x)**2 +
                (pose2.position.y - pose1.position.y)**2
            )

            if step_distance > self.max_step_length:
                return False

        return True

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        # Stop all motion immediately
        self.controller.stop_motion()

        # Clear navigation goals
        self.navigation.cancel_all_goals()

        # Enter safe posture
        self.controller.enter_safe_posture()

        # Log emergency event
        self.log_emergency_event()
```

## Summary

Perception and navigation systems form the cognitive foundation of humanoid robotics, enabling robots to understand their environment and move safely through complex spaces. The integration of computer vision, depth processing, sensor fusion, and navigation algorithms creates sophisticated systems that can handle the unique challenges of bipedal locomotion and human-scale interaction.

Successful implementation requires careful consideration of real-time performance, safety constraints, and the dynamic nature of human environments. The closed-loop integration of perception and action systems enables humanoid robots to adapt to changing conditions while maintaining stability and achieving navigation goals.

## Further Reading

- "Principles of Robot Motion" by Howie Choset et al.
- "Probabilistic Robotics" by Sebastian Thrun, Wolfram Burgard, and Dieter Fox
- "Humanoid Robotics: A Reference" by Ambarish Goswami and Prahlad Vadakkepat
- "Robotics, Vision and Control" by Peter Corke
- ROS Navigation Stack Documentation: http://wiki.ros.org/navigation