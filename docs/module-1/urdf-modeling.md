---
sidebar_position: 6
---

# Humanoid Robot Modeling with URDF: Unified Robot Description Format

## Overview

Unified Robot Description Format (URDF) is the standard XML-based format used in ROS 2 to describe robot models, including their physical properties, kinematic structure, and visual appearance. In Physical AI applications, URDF serves as the foundational representation that bridges the gap between abstract robot concepts and concrete implementations in simulation and real-world systems. For humanoid robots, which possess complex kinematic chains and multiple degrees of freedom, URDF provides the essential framework for modeling the intricate relationships between joints, links, and sensors.

URDF enables the precise specification of robot geometry, mass properties, joint constraints, and sensor placements, making it possible to accurately simulate humanoid robots in digital environments before deploying them in physical systems. This modeling approach is crucial for sim-to-real transfer, where behaviors developed in simulation must translate effectively to real-world robotic platforms.

## Learning Objectives

By the end of this section, you should be able to:
- Understand the structure and components of URDF files for humanoid robot modeling
- Create complete URDF descriptions for complex humanoid robots with multiple limbs
- Define kinematic chains, joint constraints, and physical properties using URDF
- Integrate sensor models and collision properties into URDF descriptions
- Use URDF with ROS 2 tools for visualization, simulation, and robot control

## Introduction to URDF

### What is URDF?

**URDF (Unified Robot Description Format)** is an XML-based format that describes robot models in ROS 2. It defines the robot's physical structure through a tree of connected links and joints, enabling accurate simulation, visualization, and control of robotic systems. For humanoid robots, URDF provides the essential kinematic and dynamic information needed for motion planning, control, and simulation.

### Core Components of URDF

A URDF model consists of several key elements:

- **Links**: Rigid bodies that represent physical parts of the robot
- **Joints**: Connections between links that define allowable motion
- **Visual**: Elements that define how the robot appears in visualization
- **Collision**: Elements that define collision properties for physics simulation
- **Inertial**: Mass, center of mass, and inertia properties for dynamics
- **Materials**: Visual appearance properties (colors, textures)

### URDF in the Physical AI Pipeline

URDF serves as the foundational representation that connects:
- **Simulation**: Gazebo and other physics engines use URDF for robot models
- **Control**: Robot controllers use URDF for kinematic calculations
- **Perception**: Vision systems use URDF for robot state estimation
- **Planning**: Motion planners use URDF for collision checking and path planning

## URDF Structure and Syntax

### Basic URDF File Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links definition -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints definition -->
  <joint name="base_to_link1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>
</robot>
```

### Link Definition

A link represents a rigid body in the robot model and contains three main elements:

1. **Inertial**: Physical properties for dynamics simulation
2. **Visual**: Appearance for visualization
3. **Collision**: Geometry for collision detection

```xml
<link name="link_name">
  <!-- Physical properties -->
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>

  <!-- Visual appearance -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>

  <!-- Collision geometry -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
</link>
```

### Joint Definition

Joints connect links and define the allowable motion between them:

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Joint Types in URDF

### Revolute Joints

Revolute joints allow rotation around a single axis, commonly used for rotational joints in humanoid robots:

```xml
<joint name="hip_yaw_joint" type="revolute">
  <parent link="torso_link"/>
  <child link="left_hip_link"/>
  <origin xyz="0 0.1 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="200" velocity="2"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>
```

### Continuous Joints

Continuous joints allow unlimited rotation, useful for wheels or continuously rotating parts:

```xml
<joint name="continuous_joint" type="continuous">
  <parent link="base_link"/>
  <child link="rotating_part"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.1"/>
</joint>
```

### Prismatic Joints

Prismatic joints allow linear motion along an axis:

```xml
<joint name="linear_joint" type="prismatic">
  <parent link="base_link"/>
  <child link="slider_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="0.5"/>
</joint>
```

### Fixed Joints

Fixed joints create rigid connections between links:

```xml
<joint name="fixed_joint" type="fixed">
  <parent link="base_link"/>
  <child link="sensor_mount"/>
  <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
</joint>
```

## Humanoid Robot Kinematic Structure

### Typical Humanoid Kinematic Chain

Humanoid robots typically follow a kinematic structure with:

- **Torso**: Central body with head, arms, and legs attached
- **Arms**: Shoulder → Elbow → Wrist → Hand chain
- **Legs**: Hip → Knee → Ankle → Foot chain
- **Head**: Neck joint for orientation

### Example Humanoid URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="50" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder_link"/>
    <origin xyz="0.2 0.15 0.5" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Advanced URDF Features

### Transmission Elements

Transmission elements define how actuators connect to joints:

```xml
<transmission name="left_shoulder_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_shoulder_joint">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

Gazebo plugins can be integrated directly into URDF:

```xml
<gazebo reference="base_link">
  <material>Gazebo/Gray</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <update_rate>30</update_rate>
    <joint_name>left_shoulder_joint</joint_name>
  </plugin>
</gazebo>
```

### Sensor Integration

Sensors can be attached to links using fixed joints:

```xml
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
</joint>

<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

## Xacro for Complex Models

Xacro (XML Macros) extends URDF with macros, properties, and mathematical expressions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="link_length" value="0.3" />
  <xacro:property name="link_radius" value="0.05" />
  <xacro:property name="link_mass" value="1.0" />

  <!-- Macro for creating a simple link -->
  <xacro:macro name="simple_link" params="name xyz_length">
    <link name="${name}">
      <inertial>
        <mass value="${link_mass}"/>
        <origin xyz="0 0 ${xyz_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="0 0 ${xyz_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${link_radius}" length="${xyz_length}"/>
        </geometry>
        <material name="light_gray">
          <color rgba="0.7 0.7 0.7 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 ${xyz_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${link_radius}" length="${xyz_length}"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Macro for creating a revolute joint -->
  <xacro:macro name="simple_joint" params="name parent child xyz rpy axis lower upper">
    <joint name="${name}" type="revolute">
      <parent link="${parent}"/>
      <child link="${child}"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="${axis}"/>
      <limit lower="${lower}" upper="${upper}" effort="100" velocity="1"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>
  </xacro:macro>

  <!-- Use macros to build the robot -->
  <xacro:simple_link name="base_link" xyz_length="0.5" />

  <xacro:simple_link name="upper_arm_link" xyz_length="${link_length}" />

  <xacro:simple_joint name="shoulder_joint"
                      parent="base_link"
                      child="upper_arm_link"
                      xyz="0.2 0 0.25"
                      rpy="0 0 0"
                      axis="0 1 0"
                      lower="${-M_PI/2}"
                      upper="${M_PI/2}" />
</robot>
```

## URDF Tools and Visualization

### Robot State Publisher

The robot_state_publisher node publishes the robot's joint states as transformations:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.publish_joint_states)

    def publish_joint_states(self):
        msg = JointState()
        msg.name = ['left_shoulder_joint', 'right_shoulder_joint']
        msg.position = [math.sin(self.get_clock().now().nanoseconds * 1e-9),
                       math.cos(self.get_clock().now().nanoseconds * 1e-9)]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(msg)
```

### URDF Validation

URDF files should be validated for proper syntax and kinematic structure:

```bash
# Validate URDF file
check_urdf /path/to/robot.urdf

# Parse and display robot information
urdf_to_graphiz /path/to/robot.urdf
```

## Practical Applications in Humanoid Robotics

### Walking Pattern Generation

URDF models are essential for generating walking patterns and balance control:

```xml
<!-- Example of a simplified walking robot model -->
<link name="left_foot">
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
  </inertial>
  <visual>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.15 0.1 0.05"/>
    </geometry>
    <material name="black">
      <color rgba="0 0 0 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.15 0.1 0.05"/>
    </geometry>
  </collision>
</link>
```

### Sensor Placement and Integration

Proper sensor placement in URDF is crucial for perception and control:

```xml
<!-- Camera on head -->
<link name="camera_link">
  <inertial>
    <mass value="0.1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head_link"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

## Best Practices for Humanoid URDF Modeling

### Kinematic Chain Design

- **Tree Structure**: Ensure the kinematic structure forms a tree (no loops)
- **Joint Limits**: Set realistic limits based on physical constraints
- **Mass Properties**: Use accurate mass and inertia values
- **Collision Models**: Create simplified collision geometry for performance

### Performance Considerations

- **Simplified Geometry**: Use simple shapes for collision detection
- **Appropriate Detail**: Balance visual detail with computational efficiency
- **Modular Design**: Organize complex robots into logical subsystems
- **Parameterization**: Use Xacro for parameterized designs

### Validation and Testing

- **Kinematic Validation**: Verify joint ranges and workspace
- **Dynamics Validation**: Test mass properties and inertial parameters
- **Simulation Testing**: Validate in simulation before physical deployment
- **URDF Checking**: Use ROS tools to validate syntax and structure

## Integration with Physical AI Systems

### Sim-to-Real Transfer

URDF models serve as the bridge between simulation and real-world deployment:

```xml
<!-- Include both simulation and real-world properties -->
<gazebo reference="joint1">
  <implicitSpringDamper>1</implicitSpringDamper>
  <provideFeedback>true</provideFeedback>
</gazebo>

<!-- Real robot controller configuration -->
<xacro:property name="real_robot" value="true"/>
<xacro:if value="${real_robot}">
  <transmission name="joint1_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="joint1">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="joint1_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </actuator>
  </transmission>
</xacro:if>
```

### Multi-Robot Scenarios

URDF supports multiple robot instances with proper naming:

```xml
<!-- Using namespace prefixes for multiple robots -->
<xacro:macro name="robot_model" params="prefix">
  <link name="${prefix}_base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <!-- ... rest of the model ... -->
  </link>
</xacro:macro>

<!-- Instantiate multiple robots -->
<xacro:robot_model prefix="robot1"/>
<xacro:robot_model prefix="robot2"/>
```

## Summary

URDF is the essential format for describing humanoid robot models in ROS 2, providing the foundation for simulation, visualization, and control. Proper URDF modeling is crucial for Physical AI applications, as it enables accurate simulation of robot behavior and facilitates the transfer of learned behaviors from simulation to real-world deployment. Understanding URDF structure, kinematic chains, and advanced features like Xacro macros is essential for developing complex humanoid robotic systems that can operate effectively in physical environments.

The integration of sensors, actuators, and proper physical properties in URDF models ensures that simulation accurately reflects real-world behavior, making it possible to develop and test Physical AI algorithms in virtual environments before deploying them on actual robotic platforms.

## Further Reading

- URDF Documentation: http://wiki.ros.org/urdf
- Xacro Documentation: http://wiki.ros.org/xacro
- Robot Modeling and Control by Spong, Hutchinson, and Vidyasagar
- Modern Robotics: Mechanics, Planning, and Control by Lynch and Park
- ROS 2 URDF Tutorials: https://docs.ros.org/en/humble/Tutorials/URDF.html