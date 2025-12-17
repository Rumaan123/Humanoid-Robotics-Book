---
sidebar_position: 7
---

# Architectural View of Robot Control Pipelines: From Perception to Action

## Overview

Robot control pipelines represent the systematic flow of information and commands from sensor perception through decision-making to actuator execution in humanoid robotic systems. In Physical AI applications, these pipelines form the critical infrastructure that enables robots to perceive their environment, reason about their state and goals, and execute appropriate physical actions. Understanding the architectural patterns of robot control pipelines is essential for developing robust, efficient, and safe humanoid robots that can operate in complex, dynamic environments.

The architectural view encompasses multiple levels of control, from low-level motor control to high-level cognitive planning, each with distinct timing requirements, computational needs, and safety considerations. This hierarchical structure enables the coordination of diverse subsystems while maintaining the real-time performance necessary for stable robot operation.

## Learning Objectives

By the end of this section, you should be able to:
- Understand the hierarchical structure of robot control pipelines and their timing requirements
- Design multi-layered control architectures for humanoid robots with appropriate feedback loops
- Integrate perception, planning, and control components into cohesive pipeline architectures
- Implement safety mechanisms and fault tolerance within control pipeline architectures
- Analyze and optimize control pipeline performance for real-time humanoid robot operation

## Introduction to Robot Control Pipelines

### What is a Control Pipeline?

A **robot control pipeline** is an architectural pattern that organizes the flow of information and control commands through multiple processing layers, each operating at different temporal and spatial scales. The pipeline typically flows from sensor inputs at the bottom to actuator commands at the top, with intermediate layers for state estimation, planning, and coordination.

### Key Characteristics of Control Pipelines

- **Hierarchical Structure**: Multiple layers of control with different update rates and responsibilities
- **Real-time Requirements**: Strict timing constraints for stable robot operation
- **Feedback Integration**: Continuous monitoring and adjustment based on sensor data
- **Safety Criticality**: Multiple layers of safety checks and emergency procedures
- **Distributed Processing**: Components may run on different hardware platforms

### Control Pipeline in Physical AI Context

In Physical AI systems, control pipelines bridge the gap between digital intelligence and physical action, requiring careful consideration of:
- **Latency**: Time delays between perception and action
- **Accuracy**: Precision of state estimation and control commands
- **Robustness**: Ability to handle sensor noise and environmental disturbances
- **Adaptability**: Capacity to adjust to changing conditions and goals

## Hierarchical Control Architecture

### The Three-Tier Control Architecture

Robot control systems typically follow a three-tier architecture:

1. **Low-Level Control (1-10 kHz)**: Direct motor control and hardware interfaces
2. **Mid-Level Control (10-100 Hz)**: Trajectory following and motion control
3. **High-Level Control (1-10 Hz)**: Planning, decision making, and task execution

### Low-Level Control Layer

The low-level control layer handles direct hardware interfaces and motor control:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class LowLevelController(Node):
    def __init__(self):
        super().__init__('low_level_controller')

        # Publishers for motor commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/hardware_interface/joint_commands',
            10
        )

        # Subscribers for sensor feedback
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # High-frequency control loop (1kHz)
        self.control_timer = self.create_timer(0.001, self.low_level_control)

        self.current_positions = {}
        self.current_velocities = {}
        self.current_efforts = {}
        self.desired_positions = {}
        self.desired_velocities = {}

    def joint_state_callback(self, msg):
        """Update current joint states from hardware feedback"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_efforts[name] = msg.effort[i]

    def low_level_control(self):
        """High-frequency control loop for motor commands"""
        # Calculate control commands using PID or other control algorithms
        commands = Float64MultiArray()

        for joint_name in self.desired_positions.keys():
            # Simple PD controller example
            current_pos = self.current_positions.get(joint_name, 0.0)
            current_vel = self.current_velocities.get(joint_name, 0.0)
            desired_pos = self.desired_positions.get(joint_name, 0.0)
            desired_vel = self.desired_velocities.get(joint_name, 0.0)

            # Calculate position and velocity errors
            pos_error = desired_pos - current_pos
            vel_error = desired_vel - current_vel

            # PD control law
            kp = 100.0  # Position gain
            kd = 10.0   # Velocity gain
            control_effort = kp * pos_error + kd * vel_error

            commands.data.append(control_effort)

        self.joint_cmd_pub.publish(commands)
```

### Mid-Level Control Layer

The mid-level control layer handles trajectory following and motion control:

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.interpolate import interp1d

class MidLevelController(Node):
    def __init__(self):
        super().__init__('mid_level_controller')

        # Trajectory subscribers and publishers
        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            self.trajectory_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for low-level commands
        self.low_level_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/low_level_commands',
            10
        )

        # Control loop (100Hz)
        self.control_timer = self.create_timer(0.01, self.mid_level_control)

        self.active_trajectory = None
        self.trajectory_start_time = None
        self.current_trajectory_index = 0
        self.current_positions = {}
        self.current_velocities = {}

    def trajectory_callback(self, msg):
        """Receive and process joint trajectory commands"""
        self.active_trajectory = msg
        self.trajectory_start_time = self.get_clock().now()
        self.current_trajectory_index = 0

    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_velocities[name] = msg.velocity[i]

    def mid_level_control(self):
        """Execute trajectory following control"""
        if self.active_trajectory is None:
            return

        # Calculate current time relative to trajectory start
        current_time = (self.get_clock().now() - self.trajectory_start_time).nanoseconds * 1e-9

        # Get desired positions and velocities from trajectory
        desired_positions = {}
        desired_velocities = {}

        # Interpolate trajectory points based on current time
        for i, point in enumerate(self.active_trajectory.points):
            point_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9

            if i == 0:
                # First point
                for j, joint_name in enumerate(self.active_trajectory.joint_names):
                    desired_positions[joint_name] = point.positions[j]
                    if point.velocities:
                        desired_velocities[joint_name] = point.velocities[j]
            elif point_time >= current_time:
                # Interpolate between previous and current point
                prev_point = self.active_trajectory.points[i-1]
                prev_time = prev_point.time_from_start.sec + prev_point.time_from_start.nanosec * 1e-9

                if point_time > prev_time:
                    alpha = (current_time - prev_time) / (point_time - prev_time)
                    for j, joint_name in enumerate(self.active_trajectory.joint_names):
                        prev_pos = prev_point.positions[j]
                        curr_pos = point.positions[j]
                        desired_positions[joint_name] = prev_pos + alpha * (curr_pos - prev_pos)

                        if point.velocities and prev_point.velocities:
                            prev_vel = prev_point.velocities[j]
                            curr_vel = point.velocities[j]
                            desired_velocities[joint_name] = prev_vel + alpha * (curr_vel - prev_vel)
                break

        # Publish to low-level controller
        commands = Float64MultiArray()
        commands.data = list(desired_positions.values())

        self.low_level_cmd_pub.publish(commands)
```

### High-Level Control Layer

The high-level control layer handles planning, decision making, and task execution:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionFK
import numpy as np

class HighLevelController(Node):
    def __init__(self):
        super().__init__('high_level_controller')

        # Publishers for high-level commands
        self.task_pub = self.create_publisher(String, '/task_commands', 10)

        # Service clients for planning
        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik'
        )
        self.fk_client = self.create_client(
            GetPositionFK,
            '/compute_fk'
        )

        # Task execution timer (1Hz)
        self.task_timer = self.create_timer(1.0, self.high_level_planning)

        self.current_task = None
        self.task_queue = []
        self.robot_state = None

    def high_level_planning(self):
        """High-level planning and task execution"""
        if not self.task_queue and self.current_task is None:
            # Generate new tasks based on goals or environment
            self.generate_tasks()

        if self.task_queue and self.current_task is None:
            # Start next task
            self.current_task = self.task_queue.pop(0)
            self.execute_task(self.current_task)

    def generate_tasks(self):
        """Generate tasks based on current state and goals"""
        # Example: Generate walking tasks based on navigation goals
        # This would integrate with perception and planning systems
        pass

    def execute_task(self, task):
        """Execute a high-level task"""
        # Break down task into mid-level trajectory commands
        # This might involve path planning, inverse kinematics, etc.
        pass
```

## Control Pipeline Patterns

### Feedback Control Loop Pattern

The fundamental pattern in robot control is the feedback control loop:

```python
class FeedbackController:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.integral_error = 0.0
        self.previous_error = 0.0

    def update(self, setpoint, measurement):
        """Standard PID feedback control"""
        error = setpoint - measurement

        # Proportional term
        p_term = error

        # Integral term
        self.integral_error += error * self.dt

        # Derivative term
        derivative_error = (error - self.previous_error) / self.dt
        self.previous_error = error

        # PID output
        output = (1.0 * p_term +  # kp
                 0.1 * self.integral_error +  # ki
                 0.05 * derivative_error)  # kd

        return output
```

### State Machine Pattern

State machines are commonly used for task-level control:

```python
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    WALKING = 2
    STANDING = 3
    FALLING = 4
    RECOVERING = 5

class StateMachineController:
    def __init__(self):
        self.current_state = RobotState.IDLE
        self.state_start_time = 0.0

    def update(self, sensor_data, time):
        """State transition logic"""
        new_state = self.current_state

        # State transition logic
        if self.current_state == RobotState.IDLE:
            if self.should_start_walking(sensor_data):
                new_state = RobotState.WALKING
        elif self.current_state == RobotState.WALKING:
            if self.should_stop_walking(sensor_data):
                new_state = RobotState.STANDING
            elif self.is_falling(sensor_data):
                new_state = RobotState.FALLING
        elif self.current_state == RobotState.FALLING:
            if self.should_recover(sensor_data):
                new_state = RobotState.RECOVERING
        elif self.current_state == RobotState.RECOVERING:
            if self.is_stable(sensor_data):
                new_state = RobotState.STANDING

        if new_state != self.current_state:
            self.exit_state(self.current_state)
            self.current_state = new_state
            self.state_start_time = time
            self.enter_state(self.current_state)

        # Execute state-specific control
        return self.execute_state(sensor_data)

    def enter_state(self, state):
        """Actions to take when entering a state"""
        pass

    def exit_state(self, state):
        """Actions to take when exiting a state"""
        pass

    def execute_state(self, sensor_data):
        """Execute control logic for current state"""
        pass
```

### Component-Based Architecture

Modular components enable flexible pipeline composition:

```python
class ControlComponent:
    """Base class for control pipeline components"""
    def __init__(self, name):
        self.name = name
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}

    def update(self, inputs):
        """Process inputs and generate outputs"""
        self.inputs = inputs
        self.outputs = self.process(inputs)
        return self.outputs

    def process(self, inputs):
        """Implement specific processing logic"""
        raise NotImplementedError

class PerceptionComponent(ControlComponent):
    """Component for processing sensor data"""
    def process(self, inputs):
        # Process raw sensor data into meaningful information
        processed_data = {}

        if 'raw_sensors' in inputs:
            # Example: process IMU data for orientation
            imu_data = inputs['raw_sensors'].get('imu', {})
            processed_data['orientation'] = self.process_orientation(imu_data)

        if 'camera' in inputs:
            # Example: process camera data for obstacle detection
            processed_data['obstacles'] = self.detect_obstacles(inputs['camera'])

        return processed_data

class PlanningComponent(ControlComponent):
    """Component for path planning and trajectory generation"""
    def process(self, inputs):
        # Generate trajectories based on goals and current state
        trajectory = {}

        if 'goal' in inputs and 'current_state' in inputs:
            trajectory = self.plan_trajectory(
                inputs['goal'],
                inputs['current_state'],
                inputs.get('obstacles', [])
            )

        return {'trajectory': trajectory}

class ControlComponent(ControlComponent):
    """Component for low-level control"""
    def process(self, inputs):
        # Generate motor commands from trajectories
        commands = {}

        if 'trajectory' in inputs and 'current_state' in inputs:
            commands = self.generate_commands(
                inputs['trajectory'],
                inputs['current_state']
            )

        return {'motor_commands': commands}
```

## Safety and Fault Tolerance

### Safety Architecture

Safety is paramount in humanoid robot control pipelines:

```python
class SafetyMonitor:
    def __init__(self):
        self.emergency_stop = False
        self.safety_limits = {
            'joint_position': {'min': -3.14, 'max': 3.14},
            'joint_velocity': {'max': 5.0},
            'torque': {'max': 100.0},
            'imu_angle': {'max': 0.5}  # 30 degrees
        }

    def check_safety(self, robot_state):
        """Check if robot state is within safe limits"""
        violations = []

        # Check joint positions
        for joint_name, position in robot_state.get('joint_positions', {}).items():
            if (position < self.safety_limits['joint_position']['min'] or
                position > self.safety_limits['joint_position']['max']):
                violations.append(f'Joint {joint_name} position limit violation')

        # Check joint velocities
        for joint_name, velocity in robot_state.get('joint_velocities', {}).items():
            if abs(velocity) > self.safety_limits['joint_velocity']['max']:
                violations.append(f'Joint {joint_name} velocity limit violation')

        # Check IMU for dangerous angles
        imu_orientation = robot_state.get('imu_orientation', {})
        if 'roll' in imu_orientation and abs(imu_orientation['roll']) > self.safety_limits['imu_angle']['max']:
            violations.append('Dangerous roll angle detected')

        if violations:
            self.emergency_stop = True
            return False, violations

        return True, []

    def reset_safety(self):
        """Reset emergency stop state"""
        self.emergency_stop = False
```

### Fault Detection and Recovery

```python
class FaultDetector:
    def __init__(self):
        self.fault_history = {}
        self.recovery_strategies = {}

    def detect_faults(self, sensor_data, control_outputs):
        """Detect potential faults in the system"""
        faults = []

        # Check for sensor failures
        if self.is_sensor_faulty(sensor_data):
            faults.append('Sensor fault detected')

        # Check for actuator failures
        if self.is_actuator_faulty(control_outputs):
            faults.append('Actuator fault detected')

        # Check for unexpected behavior
        if self.is_behavior_unexpected(sensor_data):
            faults.append('Unexpected behavior detected')

        return faults

    def is_sensor_faulty(self, sensor_data):
        """Check for sensor anomalies"""
        # Example: Check for sensor value jumps
        return False

    def is_actuator_faulty(self, control_outputs):
        """Check for actuator anomalies"""
        # Example: Check for excessive effort commands
        return False

    def is_behavior_unexpected(self, sensor_data):
        """Check for unexpected robot behavior"""
        # Example: Check for instability indicators
        return False
```

## Real-time Performance Considerations

### Timing Analysis

```python
import time
from collections import deque

class TimingAnalyzer:
    def __init__(self, window_size=100):
        self.execution_times = deque(maxlen=window_size)
        self.period_times = deque(maxlen=window_size)
        self.deadline_misses = 0
        self.target_period = 0.01  # 10ms for 100Hz control

    def start_timing(self):
        """Start timing measurement"""
        self.start_time = time.time()

    def stop_timing(self):
        """Stop timing measurement and record results"""
        end_time = time.time()
        execution_time = end_time - self.start_time
        period_time = end_time - getattr(self, 'last_end_time', end_time)

        self.execution_times.append(execution_time)
        self.period_times.append(period_time)

        if execution_time > self.target_period:
            self.deadline_misses += 1

        self.last_end_time = end_time

    def get_statistics(self):
        """Get timing performance statistics"""
        if not self.execution_times:
            return {}

        avg_exec_time = sum(self.execution_times) / len(self.execution_times)
        max_exec_time = max(self.execution_times)
        min_exec_time = min(self.execution_times)

        avg_period = sum(self.period_times) / len(self.period_times)

        return {
            'avg_execution_time': avg_exec_time,
            'max_execution_time': max_exec_time,
            'min_execution_time': min_exec_time,
            'avg_period': avg_period,
            'deadline_misses': self.deadline_misses,
            'utilization': avg_exec_time / self.target_period
        }
```

### Priority-Based Scheduling

```python
import heapq
import threading
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ControlTask:
    priority: int
    execution_time: float
    deadline: float
    function: callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    task_id: str = ""

class PriorityScheduler:
    def __init__(self):
        self.task_queue = []
        self.running = True
        self.scheduler_thread = threading.Thread(target=self.run_scheduler)
        self.scheduler_thread.start()

    def add_task(self, task: ControlTask):
        """Add a task to the priority queue"""
        heapq.heappush(self.task_queue, (task.priority, task))

    def run_scheduler(self):
        """Run the priority-based scheduler"""
        while self.running:
            if self.task_queue:
                priority, task = heapq.heappop(self.task_queue)

                # Check if task has missed its deadline
                if time.time() > task.deadline:
                    print(f"Task {task.task_id} missed deadline")
                    continue

                # Execute task
                try:
                    task.function(*task.args, **task.kwargs)
                except Exception as e:
                    print(f"Task {task.task_id} execution failed: {e}")
```

## Integration with ROS 2

### ROS 2 Control Architecture

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import String
import threading

class ROS2ControlPipeline(Node):
    def __init__(self):
        super().__init__('control_pipeline')

        # QoS profiles for different types of data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        control_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            sensor_qos
        )

        self.controller_state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/controller_state',
            self.controller_state_callback,
            control_qos
        )

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            control_qos
        )

        self.status_pub = self.create_publisher(
            String,
            '/control_status',
            10
        )

        # Control pipeline threads
        self.control_thread = threading.Thread(target=self.control_loop)
        self.perception_thread = threading.Thread(target=self.perception_loop)

        self.running = True
        self.current_state = None
        self.desired_trajectory = None

        # Start threads
        self.control_thread.start()
        self.perception_thread.start()

    def joint_state_callback(self, msg):
        """Handle incoming joint state messages"""
        self.current_state = msg

    def controller_state_callback(self, msg):
        """Handle controller state messages"""
        # Update internal state based on controller feedback
        pass

    def control_loop(self):
        """Main control loop running in separate thread"""
        rate = self.create_rate(100)  # 100Hz
        while self.running and rclpy.ok():
            if self.current_state and self.desired_trajectory:
                # Execute control algorithm
                commands = self.compute_control_commands()
                self.trajectory_pub.publish(commands)
            rate.sleep()

    def perception_loop(self):
        """Perception processing loop"""
        rate = self.create_rate(30)  # 30Hz for perception
        while self.running and rclpy.ok():
            # Process sensor data and update world model
            self.process_perception_data()
            rate.sleep()

    def compute_control_commands(self):
        """Compute control commands based on current state and desired trajectory"""
        # Implement control algorithm
        trajectory = JointTrajectory()
        # ... populate trajectory message
        return trajectory

    def process_perception_data(self):
        """Process sensor data for state estimation and environment understanding"""
        # Implement perception processing
        pass
```

## Practical Applications in Humanoid Robotics

### Walking Control Pipeline

```python
class WalkingController:
    def __init__(self):
        self.footstep_planner = FootstepPlanner()
        self.trajectory_generator = WalkingTrajectoryGenerator()
        self.balance_controller = BalanceController()
        self.state_estimator = StateEstimator()

        # Walking states
        self.current_step = 0
        self.support_foot = 'left'
        self.walking_state = 'stance'

    def update_walking(self, robot_state, walk_command):
        """Main walking control pipeline"""
        # 1. State estimation
        estimated_state = self.state_estimator.estimate(robot_state)

        # 2. Footstep planning (if needed)
        if self.should_plan_footsteps(walk_command):
            footsteps = self.footstep_planner.plan(
                walk_command,
                estimated_state['position'],
                estimated_state['orientation']
            )

        # 3. Trajectory generation
        walking_trajectory = self.trajectory_generator.generate(
            footsteps,
            estimated_state,
            self.current_step
        )

        # 4. Balance control
        balance_commands = self.balance_controller.compute(
            walking_trajectory,
            estimated_state
        )

        # 5. Output to joint controllers
        return balance_commands

    def should_plan_footsteps(self, command):
        """Determine if new footsteps need to be planned"""
        # Logic to determine if replanning is needed
        return False
```

### Manipulation Control Pipeline

```python
class ManipulationController:
    def __init__(self):
        self.ik_solver = InverseKinematicsSolver()
        self.trajectory_planner = TrajectoryPlanner()
        self.grasp_planner = GraspPlanner()
        self.force_controller = ForceController()

    def execute_manipulation(self, task_description, robot_state):
        """Manipulation control pipeline"""
        # 1. Task planning
        grasp_pose = self.grasp_planner.plan_grasp(
            task_description['object'],
            task_description['goal']
        )

        # 2. Inverse kinematics
        joint_trajectory = self.ik_solver.solve(
            grasp_pose,
            robot_state['end_effector_pose']
        )

        # 3. Trajectory optimization
        optimized_trajectory = self.trajectory_planner.optimize(
            joint_trajectory,
            robot_state['obstacles']
        )

        # 4. Force control integration
        force_commands = self.force_controller.compute(
            optimized_trajectory,
            expected_contact_points
        )

        return optimized_trajectory, force_commands
```

## Performance Optimization

### Pipeline Parallelization

```python
import concurrent.futures
import queue
import threading

class ParallelControlPipeline:
    def __init__(self, num_threads=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.components = []
        self.running = True

    def add_component(self, component):
        """Add a processing component to the pipeline"""
        self.components.append(component)

    def process_pipeline(self, input_data):
        """Process data through all pipeline components in parallel where possible"""
        # Submit independent components in parallel
        futures = []
        for component in self.components:
            future = self.executor.submit(component.update, input_data)
            futures.append(future)

        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

        return results
```

### Memory Management

```python
import numpy as np
from collections import deque

class MemoryEfficientController:
    def __init__(self, buffer_size=1000):
        # Pre-allocate arrays to avoid memory allocation during control
        self.position_buffer = np.zeros((buffer_size, 20))  # 20 joints
        self.velocity_buffer = np.zeros((buffer_size, 20))
        self.command_buffer = np.zeros((buffer_size, 20))

        self.buffer_index = 0
        self.buffer_size = buffer_size

    def update_control(self, current_positions, current_velocities):
        """Memory-efficient control update"""
        # Use pre-allocated arrays
        self.position_buffer[self.buffer_index] = current_positions
        self.velocity_buffer[self.buffer_index] = current_velocities

        # Compute control commands
        commands = self.compute_efficient_control(
            self.position_buffer[self.buffer_index],
            self.velocity_buffer[self.buffer_index]
        )

        self.command_buffer[self.buffer_index] = commands

        # Update buffer index
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size

        return commands
```

## Summary

Robot control pipelines provide the architectural foundation for humanoid robots, organizing the complex flow of information from perception to action through hierarchical control layers. Each layer operates at different temporal and spatial scales, with appropriate feedback mechanisms and safety considerations. Understanding these architectural patterns is crucial for developing robust Physical AI systems that can operate safely and effectively in real-world environments.

The integration of safety mechanisms, real-time performance considerations, and ROS 2's distributed architecture enables the development of sophisticated humanoid control systems that can handle the complexity and demands of physical interaction with the environment. Proper pipeline design ensures that humanoid robots can perceive, reason, and act in a coordinated and reliable manner.

## Further Reading

- "Robotics: Control, Sensing, Vision, and Intelligence" by Fu, Gonzalez, and Lee
- "Handbook of Robotics" edited by Siciliano and Khatib
- "Planning Algorithms" by LaValle
- "Probabilistic Robotics" by Thrun, Burgard, and Fox
- ROS 2 Control Documentation: https://control.ros.org/