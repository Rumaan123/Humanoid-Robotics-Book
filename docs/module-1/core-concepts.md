---
sidebar_position: 4
---

# Core Concepts: Nodes, Topics, Services, and Actions

## Overview

The core communication patterns in ROS 2 form the foundation of all robotic applications. Understanding nodes, topics, services, and actions is essential for developing Physical AI systems that integrate perception, cognition, and action. These patterns provide the architectural building blocks for distributed robotic systems.

## Nodes: The Building Blocks of ROS 2

### What is a Node?

A **node** is a process that performs computation in a ROS 2 system. Nodes are the fundamental execution units that implement specific functionality within a robotic application. In Physical AI systems, nodes might represent:

- **Sensor drivers**: Converting raw sensor data to ROS 2 messages
- **Perception algorithms**: Processing sensor data to extract meaningful information
- **Planning algorithms**: Generating motion plans or high-level strategies
- **Control algorithms**: Converting plans to actuator commands
- **Monitoring systems**: Tracking system health and performance

### Node Characteristics

- **Process-based**: Each node runs as a separate process
- **Namespaced**: Nodes have unique names within the ROS 2 graph
- **Composable**: Nodes can be combined to create complex behaviors
- **Distributed**: Nodes can run on different machines or processes

### Node Implementation

Nodes are implemented in various programming languages (C++, Python, etc.) and must:

- Initialize the ROS 2 client library
- Create and manage ROS 2 entities (publishers, subscribers, etc.)
- Execute application logic in a loop or callback-based manner
- Handle lifecycle transitions (startup, shutdown, error conditions)

## Topics: The Publish-Subscribe Pattern

### Understanding Topics

A **topic** is a named bus over which nodes exchange messages. Topics implement a publish-subscribe communication pattern where:

- **Publishers** send messages to a topic
- **Subscribers** receive messages from a topic
- Communication is **asynchronous** and **one-to-many**

### Use Cases for Topics

Topics are ideal for:

- **Sensor data streams**: Camera images, LiDAR scans, IMU readings
- **Robot state information**: Joint positions, velocities, robot pose
- **Control commands**: Velocity commands, trajectory points
- **Environmental information**: Map updates, obstacle locations

### Quality of Service (QoS)

ROS 2 topics support QoS policies that define communication characteristics:

- **Reliability**: Reliable (all messages delivered) or best-effort
- **Durability**: Volatile (new subscribers don't receive old messages) or transient-local
- **History**: Keep-all or keep-last N messages
- **Deadline**: Maximum time between messages
- **Liveliness**: How to detect if a publisher is still active

### Example: Sensor Data Streaming

```text
Camera Driver Node
    ↓ (Publishes)
Image Messages → /camera/image_raw topic
    ↑ (Subscribes)
Image Processing Node
```

Multiple image processing nodes can subscribe to the same topic simultaneously.

## Services: Request-Response Communication

### Understanding Services

A **service** implements a request-response communication pattern where:

- **Service clients** send requests and wait for responses
- **Service servers** receive requests and send responses
- Communication is **synchronous** and **one-to-one**

### Use Cases for Services

Services are appropriate for:

- **Configuration changes**: Setting parameters, changing modes
- **One-time queries**: Requesting current robot state, calibration
- **Blocking operations**: Actions that must complete before proceeding
- **State management**: Getting/setting persistent state

### Example: Robot Configuration

```text
Navigation Node (Client)
    ↓ Request: "Set goal position"
/set_goal service
    ↑ Response: "Goal accepted" or "Goal invalid"
Move Base Server (Server)
```

## Actions: Long-Running Tasks with Feedback

### Understanding Actions

An **action** extends the service pattern to support long-running operations with:

- **Goals**: Requests for long-running operations
- **Feedback**: Periodic updates during execution
- **Results**: Final outcome when the operation completes
- **Cancelation**: Ability to interrupt long-running operations

### Use Cases for Actions

Actions are used for:

- **Navigation**: Moving to a goal with periodic progress updates
- **Arm manipulation**: Complex multi-step manipulation tasks
- **Calibration**: Long-running sensor or robot calibration procedures
- **Mapping**: Building maps that take significant time

### Action Structure

Actions have three message types:

- **Goal**: Defines what the action should do
- **Feedback**: Provides progress updates during execution
- **Result**: Contains the final outcome when complete

### Example: Robot Navigation

```text
Navigation Client
    ↓ Goal: "Move to position (x, y, theta)"
/navigate_to_pose action
    ↑ Feedback: "30% complete, current position: (x1, y1)"
    ↑ Feedback: "60% complete, current position: (x2, y2)"
    ↑ Result: "Goal reached" or "Failed to reach goal"
Navigation Server
```

## Comparing Communication Patterns

| Pattern | Type | Use Case | Characteristics |
|---------|------|----------|-----------------|
| **Topics** | Publish-Subscribe | Data streams | Asynchronous, one-to-many, continuous |
| **Services** | Request-Response | One-time operations | Synchronous, one-to-one, blocking |
| **Actions** | Request-Feedback-Result | Long-running tasks | Asynchronous, one-to-one, cancellable |

### When to Use Each Pattern

#### Use Topics for:
- **Continuous data streams** (sensors, state)
- **Real-time communication** (control commands)
- **Broadcasting information** to multiple consumers
- **Low-latency requirements**

#### Use Services for:
- **One-time queries** or configuration changes
- **Operations with clear start/stop** boundaries
- **Blocking calls** where you need an immediate result
- **State changes** that must be acknowledged

#### Use Actions for:
- **Long-running operations** (more than a few seconds)
- **Operations requiring feedback** during execution
- **Tasks that might be cancelled**
- **Complex behaviors** with intermediate results

## Practical Examples in Physical AI

### Perception Pipeline Example

```text
Camera Driver Node
    ↓ /camera/image_raw (Topic: Image stream)
Image Rectification Node
    ↓ /camera/image_rect (Topic: Processed images)
Object Detection Node
    ↓ /detected_objects (Topic: Detected objects)
Perception Fusion Node
    ↓ Service: /get_environment_state
Planning Node
    ↓ Action: /navigate_to_pose (Goal: destination)
Navigation Server
```

### Humanoid Control Example

```text
State Estimation Node
    ↓ /robot_state (Topic: Current robot pose, joint angles)
Walking Controller Node
    ↓ /leg_trajectory (Topic: Desired leg positions)
Balance Controller Node
    ↓ /torso_control (Topic: Torso orientation commands)
High-level Planner Node
    ↓ Action: /execute_behavior (Goal: complex behavior)
Behavior Server
```

## Best Practices

### Node Design

- **Single Responsibility**: Each node should have a clear, focused purpose
- **Loose Coupling**: Minimize dependencies between nodes
- **Error Handling**: Implement robust error handling and recovery
- **Lifecycle Management**: Properly handle startup, shutdown, and error states

### Topic Design

- **Naming Conventions**: Use consistent, descriptive names
- **Message Types**: Choose appropriate message types for data
- **QoS Configuration**: Set appropriate QoS policies for requirements
- **Data Rate**: Consider the impact of high-frequency publications

### Service and Action Design

- **Clear Interfaces**: Define clear, well-documented interfaces
- **Error Handling**: Include error conditions in response messages
- **Timeout Handling**: Implement appropriate timeout strategies
- **Resource Management**: Properly manage resources in long-running actions

## Integration with Physical AI

### Sensor Integration

Topics enable seamless integration of diverse sensors:

```text
Multiple sensors → Multiple topics → Single processing node
```

### Multi-Robot Systems

Communication patterns scale to multi-robot systems:

- **Topics** for broadcasting sensor data
- **Services** for robot coordination
- **Actions** for complex multi-robot behaviors

### Human-Robot Interaction

Different patterns support various interaction modalities:

- **Topics** for continuous feedback (robot state, system status)
- **Services** for discrete commands (stop, pause, change mode)
- **Actions** for complex tasks (navigate, manipulate, respond)

## Summary

Understanding the core ROS 2 communication patterns is essential for building effective Physical AI systems. Nodes, topics, services, and actions provide the architectural foundation for distributed robotic applications. The choice of communication pattern significantly impacts system performance, reliability, and maintainability.

These concepts will be crucial as we explore more complex aspects of Physical AI, including simulation, perception, and cognition in the subsequent modules. Each pattern serves specific purposes and understanding when to use each is key to successful Physical AI system design.

The next section will explore Python-based robot control using rclpy, demonstrating how these concepts are implemented in practice.