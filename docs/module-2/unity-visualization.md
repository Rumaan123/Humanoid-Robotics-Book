---
sidebar_position: 7
---

# Visualization and Interaction using Unity: Advanced 3D Rendering for Physical AI

## Overview

Unity has emerged as a powerful platform for advanced visualization and interaction in Physical AI systems, offering high-fidelity 3D rendering capabilities, realistic physics simulation, and sophisticated user interaction mechanisms. In the context of humanoid robotics and digital twins, Unity provides an alternative or complementary visualization environment to traditional robotics simulators, enabling photorealistic rendering, immersive human-robot interaction, and complex multi-agent simulations. The platform's extensive asset ecosystem, real-time rendering capabilities, and cross-platform deployment options make it particularly valuable for creating engaging and informative visualization interfaces for robotic systems.

Unity's integration with ROS 2 through specialized packages enables seamless data exchange between robotic systems and visualization environments, allowing real-time display of robot states, sensor data, and environmental information. This integration is crucial for developing intuitive interfaces that support robot monitoring, teleoperation, and immersive training environments for Physical AI systems.

## Learning Objectives

By the end of this section, you should be able to:
- Integrate Unity as a visualization and interaction platform
- Implement realistic rendering for robot and environment visualization
- Create interactive interfaces for robot teleoperation and monitoring
- Connect Unity visualization with ROS 2 simulation systems

## Introduction to Unity for Robotics

### Unity's Role in Robotics Visualization

Unity serves multiple roles in robotics and Physical AI:

- **Photorealistic Rendering**: High-quality visualization for perception training and realistic simulation
- **Human-Robot Interaction**: Intuitive interfaces for robot control and monitoring
- **Training Environments**: Safe, controllable environments for AI development
- **Multi-Agent Simulation**: Complex scenarios with multiple interacting agents
- **Virtual Reality Integration**: Immersive environments for teleoperation and training

### Unity vs. Traditional Robotics Simulators

| Aspect | Unity | Traditional Robotics Simulators (Gazebo, etc.) |
|--------|-------|-----------------------------------------------|
| Visual Quality | Photorealistic | Functional/Technical |
| Rendering Performance | GPU-intensive | CPU-based physics focus |
| Interaction Model | Game engine paradigm | Robotics-specific |
| Asset Library | Extensive marketplace | Limited robotics models |
| Physics Engine | Built-in (Unity Physics) | Specialized (ODE, Bullet) |
| Target Use | Visualization, interaction | Simulation, testing |

## Setting Up Unity for Robotics

### Unity Installation and ROS Integration

To set up Unity for robotics applications, you'll need:

1. **Unity Hub and Editor**: Download from Unity's official website
2. **ROS# Package**: For ROS communication
3. **Unity Robotics Package**: Official Unity robotics tools

```csharp
// Example: Basic Unity ROS connection setup
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class UnityROSBridge : MonoBehaviour
{
    private ROSConnection ros;

    void Start()
    {
        // Get the ROS connection system
        ros = ROSConnection.GetOrCreateInstance();

        // Set the IP address of the ROS system
        ros.Initialize("127.0.0.1", 10000);

        // Subscribe to ROS topics
        ros.Subscribe<UInt8Msg>("robot_mode", OnRobotModeReceived);
    }

    void OnRobotModeReceived(UInt8Msg robotMode)
    {
        Debug.Log("Robot mode received: " + robotMode.data);
        // Handle robot mode change
    }

    void Update()
    {
        // Publish messages at regular intervals
        if (Time.time % 1.0f < Time.deltaTime) // Every second
        {
            var message = new UInt8Msg();
            message.data = 1; // Example data
            ros.Publish("unity_status", message);
        }
    }
}
```

### Unity Robotics Hub Setup

The Unity Robotics Hub provides essential tools for robotics development:

```csharp
// Example: Robot control interface in Unity
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    [SerializeField] private string jointStateTopic = "/joint_states";
    [SerializeField] private string cmdVelTopic = "/cmd_vel";

    private ROSConnection ros;
    private JointStateMsg lastJointState;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<JointStateMsg>(jointStateTopic, OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        lastJointState = jointState;
        UpdateRobotVisualization();
    }

    void UpdateRobotVisualization()
    {
        if (lastJointState == null) return;

        // Update joint positions in Unity
        for (int i = 0; i < lastJointState.name.Count; i++)
        {
            string jointName = lastJointState.name[i];
            float jointPosition = (float)lastJointState.position[i];

            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Update joint rotation based on received position
                jointTransform.localRotation = Quaternion.Euler(0, jointPosition * Mathf.Rad2Deg, 0);
            }
        }
    }

    Transform FindJointByName(string name)
    {
        // Find the corresponding joint in the Unity hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == name)
                return child;
        }
        return null;
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        var cmd = new TwistMsg();
        cmd.linear = new Vector3Msg(linearX, 0, 0);
        cmd.angular = new Vector3Msg(0, 0, angularZ);

        ros.Publish(cmdVelTopic, cmd);
    }
}
```

## Robot Model Integration

### Importing Robot Models

Unity can import robot models in various formats, with FBX being the most common:

```csharp
// Example: Robot model importer and animator
using UnityEngine;

public class RobotModelImporter : MonoBehaviour
{
    [System.Serializable]
    public class JointMapping
    {
        public string rosJointName;
        public Transform unityJoint;
        public JointType jointType;
        public float minAngle;
        public float maxAngle;
    }

    public enum JointType
    {
        Revolute,
        Prismatic,
        Fixed
    }

    public JointMapping[] jointMappings;

    void Start()
    {
        SetupJointConstraints();
    }

    void SetupJointConstraints()
    {
        foreach (var mapping in jointMappings)
        {
            if (mapping.jointType == JointType.Revolute)
            {
                // Add hinge joint constraints if needed
                ConfigurableJoint joint = mapping.unityJoint.GetComponent<ConfigurableJoint>();
                if (joint != null)
                {
                    joint.lowAngularXLimit = mapping.minAngle;
                    joint.highAngularXLimit = mapping.maxAngle;
                }
            }
        }
    }

    public void UpdateJointPositions(float[] positions)
    {
        for (int i = 0; i < jointMappings.Length && i < positions.Length; i++)
        {
            JointMapping mapping = jointMappings[i];
            float position = positions[i];

            switch (mapping.jointType)
            {
                case JointType.Revolute:
                    mapping.unityJoint.localRotation =
                        Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
                    break;
                case JointType.Prismatic:
                    mapping.unityJoint.localPosition =
                        new Vector3(position, 0, 0);
                    break;
            }
        }
    }
}
```

### Animation and Control Systems

```csharp
// Example: Advanced robot animation controller
using UnityEngine;
using UnityEngine.Animations;

public class RobotAnimationController : MonoBehaviour
{
    [Header("Animation Parameters")]
    public Animator animator;
    public AnimationCurve walkCycle;
    public AnimationCurve balanceCorrection;

    [Header("IK Settings")]
    public bool useFootIK = true;
    public Transform leftFootTarget;
    public Transform rightFootTarget;

    private float animationSpeed = 1.0f;
    private bool isWalking = false;
    private bool isBalancing = false;

    void Start()
    {
        if (animator == null)
            animator = GetComponent<Animator>();
    }

    void OnAnimatorIK(int layerIndex)
    {
        if (!useFootIK) return;

        // Left foot IK
        if (leftFootTarget != null)
        {
            animator.SetIKPositionWeight(AvatarIKGoal.LeftFoot, 1.0f);
            animator.SetIKPosition(AvatarIKGoal.LeftFoot, leftFootTarget.position);
        }

        // Right foot IK
        if (rightFootTarget != null)
        {
            animator.SetIKPositionWeight(AvatarIKGoal.RightFoot, 1.0f);
            animator.SetIKPosition(AvatarIKGoal.RightFoot, rightFootTarget.position);
        }
    }

    public void SetWalking(bool walking)
    {
        isWalking = walking;
        animator.SetBool("IsWalking", walking);
    }

    public void SetBalanceCorrection(float correction)
    {
        isBalancing = Mathf.Abs(correction) > 0.1f;
        animator.SetFloat("BalanceCorrection", correction);
    }

    void Update()
    {
        // Update animation parameters based on robot state
        animator.SetFloat("Speed", animationSpeed);
        animator.SetBool("IsBalancing", isBalancing);
    }
}
```

## Environment Creation and Management

### Procedural Environment Generation

```csharp
// Example: Procedural environment generation for robotics training
using UnityEngine;
using System.Collections.Generic;

public class ProceduralEnvironmentGenerator : MonoBehaviour
{
    [System.Serializable]
    public class EnvironmentPrefab
    {
        public GameObject prefab;
        public int minCount;
        public int maxCount;
        public Bounds spawnBounds;
        public float minScale = 0.8f;
        public float maxScale = 1.2f;
    }

    public EnvironmentPrefab[] environmentPrefabs;
    public Transform environmentRoot;

    [Header("Terrain Settings")]
    public int terrainWidth = 100;
    public int terrainHeight = 100;
    public float terrainScale = 1.0f;

    private List<GameObject> spawnedObjects = new List<GameObject>();

    public void GenerateEnvironment()
    {
        ClearEnvironment();

        // Generate terrain
        GenerateTerrain();

        // Spawn environment objects
        foreach (var prefabData in environmentPrefabs)
        {
            int count = Random.Range(prefabData.minCount, prefabData.maxCount + 1);

            for (int i = 0; i < count; i++)
            {
                SpawnEnvironmentObject(prefabData);
            }
        }
    }

    void GenerateTerrain()
    {
        // Create terrain programmatically
        Terrain terrain = Terrain.activeTerrain;
        if (terrain == null)
        {
            GameObject terrainObj = new GameObject("ProceduralTerrain");
            terrain = terrainObj.AddComponent<Terrain>();
            terrain.terrainData = new TerrainData();
        }

        // Set terrain size
        terrain.terrainData.size = new Vector3(terrainWidth, 10, terrainHeight);

        // Generate heightmap
        int resolution = terrain.terrainData.heightmapResolution;
        float[,] heights = new float[resolution, resolution];

        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                float xCoord = (float)x / resolution * terrainScale;
                float yCoord = (float)y / resolution * terrainScale;

                // Generate terrain height using noise
                heights[x, y] = Mathf.PerlinNoise(xCoord, yCoord) * 0.1f;
            }
        }

        terrain.terrainData.SetHeights(0, 0, heights);
    }

    void SpawnEnvironmentObject(EnvironmentPrefab prefabData)
    {
        Vector3 spawnPosition = new Vector3(
            Random.Range(prefabData.spawnBounds.min.x, prefabData.spawnBounds.max.x),
            prefabData.spawnBounds.min.y,
            Random.Range(prefabData.spawnBounds.min.z, prefabData.spawnBounds.max.z)
        );

        GameObject spawnedObject = Instantiate(
            prefabData.prefab,
            spawnPosition,
            Quaternion.Euler(0, Random.Range(0, 360), 0),
            environmentRoot
        );

        // Random scaling
        float scale = Random.Range(prefabData.minScale, prefabData.maxScale);
        spawnedObject.transform.localScale = Vector3.one * scale;

        spawnedObjects.Add(spawnedObject);
    }

    void ClearEnvironment()
    {
        foreach (GameObject obj in spawnedObjects)
        {
            if (obj != null)
                DestroyImmediate(obj);
        }
        spawnedObjects.Clear();
    }
}
```

### Dynamic Environment Interaction

```csharp
// Example: Physics-based environment interaction
using UnityEngine;

public class EnvironmentInteraction : MonoBehaviour
{
    [Header("Interaction Settings")]
    public float interactionDistance = 3.0f;
    public LayerMask interactionLayers;

    [Header("Physics Settings")]
    public float pushForce = 10.0f;
    public float grabDistance = 2.0f;

    private Camera mainCamera;
    private GameObject grabbedObject;
    private FixedJoint grabJoint;

    void Start()
    {
        mainCamera = Camera.main;
    }

    void Update()
    {
        HandleInteractionInput();
    }

    void HandleInteractionInput()
    {
        // Raycast for interaction
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, interactionDistance, interactionLayers))
        {
            GameObject target = hit.collider.gameObject;

            if (Input.GetKeyDown(KeyCode.E))
            {
                // Grab object
                GrabObject(target, hit.point);
            }
            else if (Input.GetKeyDown(KeyCode.F))
            {
                // Push object
                PushObject(target, hit.normal);
            }
        }

        // Update grabbed object position
        if (grabbedObject != null && grabJoint == null)
        {
            // Move object with mouse
            Ray rayToObj = mainCamera.ScreenPointToRay(Input.mousePosition);
            Vector3 targetPosition = rayToObj.GetPoint(grabDistance);
            grabbedObject.transform.position = Vector3.Lerp(
                grabbedObject.transform.position,
                targetPosition,
                Time.deltaTime * 10f
            );
        }
    }

    void GrabObject(GameObject obj, Vector3 hitPoint)
    {
        Rigidbody rb = obj.GetComponent<Rigidbody>();
        if (rb != null)
        {
            grabbedObject = obj;

            // Create fixed joint for grabbing
            grabJoint = obj.AddComponent<FixedJoint>();
            grabJoint.connectedBody = this.GetComponent<Rigidbody>();
        }
    }

    void PushObject(GameObject obj, Vector3 normal)
    {
        Rigidbody rb = obj.GetComponent<Rigidbody>();
        if (rb != null)
        {
            Vector3 pushDirection = normal * pushForce;
            rb.AddForce(pushDirection, ForceMode.Impulse);
        }
    }

    public void ReleaseObject()
    {
        if (grabbedObject != null)
        {
            if (grabJoint != null)
            {
                DestroyImmediate(grabJoint);
            }
            grabbedObject = null;
        }
    }
}
```

## Sensor Visualization

### LiDAR Point Cloud Rendering

```csharp
// Example: Real-time LiDAR point cloud visualization
using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(PointCloudRenderer))]
public class LidarPointCloudVisualizer : MonoBehaviour
{
    [Header("Point Cloud Settings")]
    public Material pointMaterial;
    public float pointSize = 0.02f;
    public Color pointColor = Color.green;

    [Header("Performance Settings")]
    public int maxPoints = 100000;
    public float updateRate = 30.0f; // Hz

    private PointCloudRenderer pointCloudRenderer;
    private List<Vector3> points = new List<Vector3>();
    private List<Color> colors = new List<Color>();
    private float lastUpdateTime;

    void Start()
    {
        pointCloudRenderer = GetComponent<PointCloudRenderer>();
        if (pointCloudRenderer == null)
        {
            pointCloudRenderer = gameObject.AddComponent<PointCloudRenderer>();
        }

        // Create material if not assigned
        if (pointMaterial == null)
        {
            pointMaterial = new Material(Shader.Find("Sprites/Default"));
            pointMaterial.color = pointColor;
        }
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= 1.0f / updateRate)
        {
            UpdatePointCloud();
            lastUpdateTime = Time.time;
        }
    }

    public void AddLidarScan(float[] ranges, float[] intensities, float angleMin, float angleIncrement)
    {
        points.Clear();
        colors.Clear();

        for (int i = 0; i < ranges.Length && points.Count < maxPoints; i++)
        {
            float range = ranges[i];
            if (range > 0 && range < 30.0f) // Valid range
            {
                float angle = angleMin + i * angleIncrement;

                Vector3 point = new Vector3(
                    range * Mathf.Cos(angle),
                    0, // Assuming 2D LiDAR
                    range * Mathf.Sin(angle)
                );

                points.Add(point);

                // Color based on intensity or distance
                float intensity = intensities != null && i < intensities.Length ? intensities[i] : 1.0f;
                Color pointColor = Color.Lerp(Color.red, Color.green, intensity / 255.0f);
                colors.Add(pointColor);
            }
        }

        // Update renderer with new points
        pointCloudRenderer.UpdatePointCloud(points, colors, pointSize);
    }

    public void Add3DLidarScan(Vector3[] pointCloud)
    {
        points.Clear();
        colors.Clear();

        for (int i = 0; i < Mathf.Min(pointCloud.Length, maxPoints); i++)
        {
            points.Add(pointCloud[i]);
            colors.Add(pointColor);
        }

        pointCloudRenderer.UpdatePointCloud(points, colors, pointSize);
    }
}

// Custom point cloud renderer
public class PointCloudRenderer : MonoBehaviour
{
    private GameObject[] pointObjects;
    private Material pointMaterial;

    public void UpdatePointCloud(List<Vector3> points, List<Color> colors, float pointSize)
    {
        // Ensure we have enough point objects
        if (pointObjects == null || pointObjects.Length != points.Count)
        {
            ClearPoints();
            CreatePoints(points.Count);
        }

        // Update positions and colors
        for (int i = 0; i < points.Count; i++)
        {
            if (pointObjects[i] != null)
            {
                pointObjects[i].transform.position = transform.position + points[i];
                pointObjects[i].transform.localScale = Vector3.one * pointSize;

                // Update color (this is a simplified approach)
                Renderer renderer = pointObjects[i].GetComponent<Renderer>();
                if (renderer != null && i < colors.Count)
                {
                    renderer.material.color = colors[i];
                }
            }
        }

        // Hide extra points
        for (int i = points.Count; i < pointObjects.Length; i++)
        {
            if (pointObjects[i] != null)
            {
                pointObjects[i].SetActive(false);
            }
        }
    }

    void CreatePoints(int count)
    {
        pointObjects = new GameObject[count];
        for (int i = 0; i < count; i++)
        {
            pointObjects[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            pointObjects[i].transform.SetParent(transform);
            pointObjects[i].SetActive(false);

            // Remove collider for performance
            DestroyImmediate(pointObjects[i].GetComponent<Collider>());
        }
    }

    void ClearPoints()
    {
        if (pointObjects != null)
        {
            foreach (GameObject point in pointObjects)
            {
                if (point != null)
                    DestroyImmediate(point);
            }
            pointObjects = null;
        }
    }
}
```

### Camera Feed Integration

```csharp
// Example: Real-time camera feed visualization
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;

public class CameraFeedVisualizer : MonoBehaviour
{
    [Header("Camera Settings")]
    public RawImage cameraDisplay;
    public int cameraWidth = 640;
    public int cameraHeight = 480;
    public int cameraFps = 30;

    [Header("ROS Integration")]
    public string imageTopic = "/camera/color/image_raw";

    private Texture2D cameraTexture;
    private byte[] imageBuffer;

    void Start()
    {
        InitializeCameraTexture();
        SubscribeToCameraTopic();
    }

    void InitializeCameraTexture()
    {
        cameraTexture = new Texture2D(cameraWidth, cameraHeight, TextureFormat.RGB24, false);
        if (cameraDisplay != null)
        {
            cameraDisplay.texture = cameraTexture;
        }
    }

    void SubscribeToCameraTopic()
    {
        // Subscribe to ROS image topic
        ROSConnection.GetOrCreateInstance()
            .Subscribe<RosMessageTypes.Sensor.ImageMsg>(
                imageTopic,
                OnImageReceived
            );
    }

    void OnImageReceived(RosMessageTypes.Sensor.ImageMsg imageMsg)
    {
        if (imageMsg.encoding == "rgb8" || imageMsg.encoding == "bgr8")
        {
            // Decode image data
            UpdateCameraTexture(imageMsg.data);
        }
    }

    void UpdateCameraTexture(byte[] imageData)
    {
        // Handle different encodings
        if (imageData.Length == cameraWidth * cameraHeight * 3)
        {
            // Direct RGB/BGR data
            Color32[] colors = new Color32[imageData.Length / 3];

            for (int i = 0; i < colors.Length; i++)
            {
                if (imageMsg.encoding == "bgr8")
                {
                    // Convert BGR to RGB
                    byte r = imageData[i * 3 + 2];
                    byte g = imageData[i * 3 + 1];
                    byte b = imageData[i * 3 + 0];
                    colors[i] = new Color32(r, g, b, 255);
                }
                else
                {
                    // RGB format
                    colors[i] = new Color32(
                        imageData[i * 3],
                        imageData[i * 3 + 1],
                        imageData[i * 3 + 2],
                        255
                    );
                }
            }

            cameraTexture.SetPixels32(colors);
            cameraTexture.Apply();
        }
    }

    // Alternative method for depth camera visualization
    public void UpdateDepthVisualization(float[,] depthData)
    {
        Color32[] depthColors = new Color32[depthData.GetLength(0) * depthData.GetLength(1)];

        float maxDepth = 10.0f; // meters

        for (int y = 0; y < depthData.GetLength(0); y++)
        {
            for (int x = 0; x < depthData.GetLength(1); x++)
            {
                float depth = depthData[y, x];
                float normalizedDepth = Mathf.Clamp01(depth / maxDepth);

                // Map depth to color gradient
                Color depthColor = Color.Lerp(Color.black, Color.white, normalizedDepth);
                depthColors[y * depthData.GetLength(1) + x] =
                    new Color32(
                        (byte)(depthColor.r * 255),
                        (byte)(depthColor.g * 255),
                        (byte)(depthColor.b * 255),
                        255
                    );
            }
        }

        cameraTexture.SetPixels32(depthColors);
        cameraTexture.Apply();
    }
}
```

## Interaction Systems

### VR/AR Integration

```csharp
// Example: VR interaction for robot teleoperation
using UnityEngine;
using UnityEngine.XR;

public class VRRobotController : MonoBehaviour
{
    [Header("VR Controller Setup")]
    public XRNode controllerNode = XRNode.RightHand;
    public Transform robotBase;

    [Header("Teleoperation Settings")]
    public float moveSpeed = 1.0f;
    public float rotationSpeed = 50.0f;

    [Header("Haptic Feedback")]
    public bool enableHaptics = true;
    public float hapticIntensity = 0.5f;

    private InputDevice controller;
    private Vector2 primaryAxis;
    private bool triggerPressed = false;

    void Start()
    {
        UpdateController();
    }

    void Update()
    {
        UpdateController();
        HandleVRInput();
        ApplyHapticFeedback();
    }

    void UpdateController()
    {
        var devices = new List<InputDevice>();
        InputDevices.GetDevicesAtXRNode(controllerNode, devices);

        if (devices.Count > 0)
        {
            controller = devices[0];
        }
    }

    void HandleVRInput()
    {
        if (controller == null) return;

        // Get thumbstick input for movement
        controller.TryGetFeatureValue(CommonUsages.primary2DAxis, out primaryAxis);

        // Get trigger input for actions
        controller.TryGetFeatureValue(CommonUsages.triggerButton, out triggerPressed);

        // Move robot based on thumbstick input
        if (primaryAxis.magnitude > 0.1f)
        {
            Vector3 movement = new Vector3(primaryAxis.x, 0, primaryAxis.y);
            robotBase.Translate(movement * moveSpeed * Time.deltaTime, Space.World);
        }

        // Rotate robot based on grip button
        bool gripPressed = false;
        controller.TryGetFeatureValue(CommonUsages.gripButton, out gripPressed);

        if (gripPressed)
        {
            robotBase.Rotate(0, primaryAxis.x * rotationSpeed * Time.deltaTime, 0);
        }
    }

    void ApplyHapticFeedback()
    {
        if (!enableHaptics || controller == null) return;

        // Apply haptic feedback based on robot state
        if (triggerPressed)
        {
            controller.SendHapticImpulse(0, hapticIntensity, 0.1f);
        }
    }

    public void SetRobotControlled(bool controlled)
    {
        // Visual feedback for robot control state
        Renderer[] renderers = robotBase.GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            Material mat = renderer.material;
            if (controlled)
            {
                mat.SetColor("_EmissionColor", Color.green);
            }
            else
            {
                mat.SetColor("_EmissionColor", Color.black);
            }
        }
    }
}
```

### UI and Control Panels

```csharp
// Example: Robot monitoring and control UI
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class RobotControlPanel : MonoBehaviour
{
    [Header("UI References")]
    public Text robotStatusText;
    public Text batteryLevelText;
    public Text jointPositionText;
    public Slider speedSlider;
    public Button emergencyStopButton;
    public Toggle autoModeToggle;

    [Header("Robot Data")]
    public float maxBattery = 100.0f;
    public float currentBattery = 85.0f;
    public float[] jointPositions;
    public float robotSpeed = 1.0f;

    private bool emergencyStopActive = false;
    private bool autoModeEnabled = false;

    void Start()
    {
        InitializeUI();
        SetupEventHandlers();
    }

    void InitializeUI()
    {
        speedSlider.minValue = 0.0f;
        speedSlider.maxValue = 2.0f;
        speedSlider.value = robotSpeed;

        UpdateUI();
    }

    void SetupEventHandlers()
    {
        speedSlider.onValueChanged.AddListener(OnSpeedChanged);
        emergencyStopButton.onClick.AddListener(OnEmergencyStop);
        autoModeToggle.onValueChanged.AddListener(OnAutoModeChanged);
    }

    void Update()
    {
        UpdateRobotData();
        UpdateUI();
    }

    void UpdateRobotData()
    {
        // Update from ROS or robot state
        // This would typically come from ROS messages
        currentBattery -= Time.deltaTime * 0.01f; // Simulate battery drain
        if (currentBattery < 0) currentBattery = 0;
    }

    void UpdateUI()
    {
        robotStatusText.text = emergencyStopActive ? "EMERGENCY STOP" :
                              autoModeEnabled ? "AUTO MODE" : "MANUAL";

        batteryLevelText.text = $"Battery: {currentBattery:F1}%";

        if (jointPositions != null && jointPositions.Length > 0)
        {
            string jointInfo = "Joints: ";
            for (int i = 0; i < Mathf.Min(5, jointPositions.Length); i++)
            {
                jointInfo += $"J{i}:{jointPositions[i]:F2} ";
            }
            jointPositionText.text = jointInfo;
        }

        robotStatusText.color = emergencyStopActive ? Color.red :
                               autoModeEnabled ? Color.blue : Color.white;
    }

    void OnSpeedChanged(float value)
    {
        robotSpeed = value;
        // Send speed command to robot
        SendSpeedCommand(robotSpeed);
    }

    void OnEmergencyStop()
    {
        emergencyStopActive = !emergencyStopActive;
        if (emergencyStopActive)
        {
            SendEmergencyStopCommand();
        }
        else
        {
            SendResumeCommand();
        }
    }

    void OnAutoModeChanged(bool isOn)
    {
        autoModeEnabled = isOn;
        if (autoModeEnabled)
        {
            SendAutoModeCommand();
        }
        else
        {
            SendManualModeCommand();
        }
    }

    void SendSpeedCommand(float speed)
    {
        // Send speed command via ROS
        var cmd = new RosMessageTypes.Std.Float32Msg();
        cmd.data = speed;
        ROSConnection.GetOrCreateInstance().Publish("/robot_speed", cmd);
    }

    void SendEmergencyStopCommand()
    {
        var cmd = new RosMessageTypes.Std.BoolMsg();
        cmd.data = true;
        ROSConnection.GetOrCreateInstance().Publish("/emergency_stop", cmd);
    }

    void SendResumeCommand()
    {
        var cmd = new RosMessageTypes.Std.BoolMsg();
        cmd.data = false;
        ROSConnection.GetOrCreateInstance().Publish("/emergency_stop", cmd);
    }

    void SendAutoModeCommand()
    {
        var cmd = new RosMessageTypes.Std.StringMsg();
        cmd.data = "auto";
        ROSConnection.GetOrCreateInstance().Publish("/control_mode", cmd);
    }

    void SendManualModeCommand()
    {
        var cmd = new RosMessageTypes.Std.StringMsg();
        cmd.data = "manual";
        ROSConnection.GetOrCreateInstance().Publish("/control_mode", cmd);
    }
}
```

## Advanced Rendering Techniques

### Realistic Lighting and Shading

```csharp
// Example: Advanced rendering for realistic robot visualization
using UnityEngine;
using UnityEngine.Rendering;

public class AdvancedRobotRenderer : MonoBehaviour
{
    [Header("Material Properties")]
    public Material robotMaterial;
    public Texture2D normalMap;
    public Texture2D roughnessMap;
    public Texture2D metallicMap;

    [Header("Lighting Settings")]
    public Light mainLight;
    public float ambientIntensity = 0.3f;
    public Color robotColor = Color.gray;

    [Header("Post-Processing")]
    public bool enableSSAO = true;
    public bool enableBloom = true;

    private Renderer robotRenderer;
    private MaterialPropertyBlock materialProperties;

    void Start()
    {
        robotRenderer = GetComponent<Renderer>();
        materialProperties = new MaterialPropertyBlock();

        SetupAdvancedRendering();
    }

    void SetupAdvancedRendering()
    {
        if (robotMaterial != null)
        {
            robotRenderer.material = robotMaterial;
        }

        // Set up material properties
        materialProperties.SetColor("_BaseColor", robotColor);
        materialProperties.SetTexture("_NormalMap", normalMap);
        materialProperties.SetTexture("_MetallicGlossMap", metallicMap);
        materialProperties.SetTexture("_BumpMap", roughnessMap);
        materialProperties.SetFloat("_Metallic", 0.5f);
        materialProperties.SetFloat("_Smoothness", 0.5f);

        robotRenderer.SetPropertyBlock(materialProperties);

        // Configure lighting
        RenderSettings.ambientIntensity = ambientIntensity;
    }

    public void UpdateRobotAppearance(float wearLevel, Color damageColor)
    {
        // Update material based on robot condition
        Color finalColor = Color.Lerp(robotColor, damageColor, wearLevel);
        materialProperties.SetColor("_BaseColor", finalColor);

        // Add wear and tear effects
        materialProperties.SetFloat("_ScratchIntensity", wearLevel);
        materialProperties.SetFloat("_DirtAmount", wearLevel * 0.5f);

        robotRenderer.SetPropertyBlock(materialProperties);
    }

    public void SetRobotHighlight(bool highlighted)
    {
        if (highlighted)
        {
            materialProperties.SetColor("_EmissionColor", Color.yellow * 2f);
            materialProperties.SetFloat("_EmissionIntensity", 1.0f);
        }
        else
        {
            materialProperties.SetColor("_EmissionColor", Color.black);
            materialProperties.SetFloat("_EmissionIntensity", 0.0f);
        }

        robotRenderer.SetPropertyBlock(materialProperties);
    }

    // Custom shader for robot-specific effects
    public void ApplyCustomShader()
    {
        Shader customShader = Shader.Find("Custom/RobotShader");
        if (customShader != null)
        {
            robotRenderer.material.shader = customShader;
        }
    }
}
```

### Multi-Agent Visualization

```csharp
// Example: Multi-agent robot visualization system
using UnityEngine;
using System.Collections.Generic;

public class MultiAgentVisualization : MonoBehaviour
{
    [Header("Agent Settings")]
    public GameObject agentPrefab;
    public int maxAgents = 100;

    [Header("Visualization Settings")]
    public Color[] agentColors;
    public float agentSize = 1.0f;
    public float connectionDistance = 5.0f;

    private List<GameObject> agents = new List<GameObject>();
    private Dictionary<string, GameObject> agentMap = new Dictionary<string, GameObject>();

    void Start()
    {
        InitializeAgents();
    }

    void InitializeAgents()
    {
        for (int i = 0; i < maxAgents; i++)
        {
            GameObject agent = Instantiate(
                agentPrefab,
                Vector3.zero,
                Quaternion.identity
            );

            agent.name = $"Agent_{i:D3}";
            agents.Add(agent);

            // Set unique color
            Renderer renderer = agent.GetComponent<Renderer>();
            if (renderer != null && agentColors.Length > 0)
            {
                int colorIndex = i % agentColors.Length;
                renderer.material.color = agentColors[colorIndex];
            }

            // Add to map for quick lookup
            agentMap[agent.name] = agent;
        }
    }

    public void UpdateAgentPositions(Dictionary<string, Vector3> agentPositions)
    {
        foreach (var kvp in agentPositions)
        {
            if (agentMap.ContainsKey(kvp.Key))
            {
                agentMap[kvp.Key].transform.position = kvp.Value;
            }
        }
    }

    public void UpdateAgentStates(Dictionary<string, AgentState> agentStates)
    {
        foreach (var kvp in agentStates)
        {
            if (agentMap.ContainsKey(kvp.Key))
            {
                UpdateAgentVisualState(agentMap[kvp.Key], kvp.Value);
            }
        }
    }

    void UpdateAgentVisualState(GameObject agent, AgentState state)
    {
        Renderer renderer = agent.GetComponent<Renderer>();
        if (renderer != null)
        {
            // Change color based on state
            Color stateColor = GetColorForState(state);
            renderer.material.color = stateColor;

            // Change size based on importance or priority
            float scale = agentSize * state.importance;
            agent.transform.localScale = Vector3.one * scale;
        }

        // Add state-specific visual effects
        switch (state.status)
        {
            case AgentStatus.Busy:
                AddBusyEffect(agent);
                break;
            case AgentStatus.Idle:
                AddIdleEffect(agent);
                break;
            case AgentStatus.Error:
                AddErrorEffect(agent);
                break;
        }
    }

    Color GetColorForState(AgentState state)
    {
        switch (state.status)
        {
            case AgentStatus.Busy: return Color.blue;
            case AgentStatus.Idle: return Color.green;
            case AgentStatus.Error: return Color.red;
            case AgentStatus.Charging: return Color.yellow;
            default: return Color.white;
        }
    }

    void AddBusyEffect(GameObject agent)
    {
        // Add rotating indicator or other visual effect
        ParticleSystem busyEffect = agent.GetComponent<ParticleSystem>();
        if (busyEffect == null)
        {
            busyEffect = agent.AddComponent<ParticleSystem>();
            var main = busyEffect.main;
            main.startColor = Color.blue;
            main.startSize = 0.2f;
        }
        busyEffect.Play();
    }

    void AddIdleEffect(GameObject agent)
    {
        // Add pulsing effect for idle agents
        ParticleSystem idleEffect = agent.GetComponent<ParticleSystem>();
        if (idleEffect == null)
        {
            idleEffect = agent.AddComponent<ParticleSystem>();
            var main = idleEffect.main;
            main.startColor = Color.green;
            main.startSize = 0.1f;
        }
        idleEffect.Stop();
    }

    void AddErrorEffect(GameObject agent)
    {
        // Add red flashing effect for errors
        ParticleSystem errorEffect = agent.GetComponent<ParticleSystem>();
        if (errorEffect == null)
        {
            errorEffect = agent.AddComponent<ParticleSystem>();
            var main = errorEffect.main;
            main.startColor = Color.red;
            main.startSize = 0.3f;
        }
        errorEffect.Play();
    }

    void Update()
    {
        // Draw connections between nearby agents
        DrawAgentConnections();
    }

    void DrawAgentConnections()
    {
        // Draw lines between agents within connection distance
        for (int i = 0; i < agents.Count; i++)
        {
            for (int j = i + 1; j < agents.Count; j++)
            {
                float distance = Vector3.Distance(
                    agents[i].transform.position,
                    agents[j].transform.position
                );

                if (distance <= connectionDistance)
                {
                    Debug.DrawLine(
                        agents[i].transform.position,
                        agents[j].transform.position,
                        Color.cyan * 0.5f
                    );
                }
            }
        }
    }
}

[System.Serializable]
public class AgentState
{
    public AgentStatus status;
    public float importance = 1.0f;
    public float batteryLevel = 100.0f;
    public string currentTask = "idle";
}

public enum AgentStatus
{
    Idle,
    Busy,
    Error,
    Charging,
    Moving
}
```

## Integration with ROS 2

### ROS 2 Communication in Unity

```csharp
// Example: ROS 2 communication system for Unity
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

public class ROS2UnityBridge : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Topics")]
    public string jointStateTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";
    public string odomTopic = "/odom";
    public string imageTopic = "/camera/color/image_raw";

    private ROSConnection ros;
    private JointStateMsg currentJointState;
    private OdometryMsg currentOdometry;
    private bool isConnected = false;

    void Start()
    {
        ConnectToROS();
        SubscribeToTopics();
    }

    void ConnectToROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);
        isConnected = true;
    }

    void SubscribeToTopics()
    {
        ros.Subscribe<JointStateMsg>(jointStateTopic, OnJointStateReceived);
        ros.Subscribe<OdometryMsg>(odomTopic, OnOdometryReceived);
        ros.Subscribe<ImageMsg>(imageTopic, OnImageReceived);
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        currentJointState = jointState;
        // Trigger joint update in visualization
        UpdateJointVisualization();
    }

    void OnOdometryReceived(OdometryMsg odometry)
    {
        currentOdometry = odometry;
        // Update robot position in Unity
        UpdateRobotPosition();
    }

    void OnImageReceived(ImageMsg image)
    {
        // Process camera image
        ProcessCameraImage(image);
    }

    void UpdateJointVisualization()
    {
        if (currentJointState == null) return;

        // Update robot joints based on received state
        for (int i = 0; i < currentJointState.name.Count; i++)
        {
            string jointName = currentJointState.name[i];
            double jointPosition = currentJointState.position[i];

            // Find and update corresponding joint in Unity
            Transform jointTransform = FindJointTransform(jointName);
            if (jointTransform != null)
            {
                jointTransform.localRotation =
                    Quaternion.Euler(0, (float)jointPosition * Mathf.Rad2Deg, 0);
            }
        }
    }

    void UpdateRobotPosition()
    {
        if (currentOdometry == null) return;

        // Update robot position and orientation
        Vector3 position = new Vector3(
            (float)currentOdometry.pose.pose.position.x,
            (float)currentOdometry.pose.pose.position.z, // Unity Y is up
            (float)currentOdometry.pose.pose.position.y
        );

        Quaternion rotation = new Quaternion(
            (float)currentOdometry.pose.pose.orientation.x,
            (float)currentOdometry.pose.pose.orientation.z,
            (float)currentOdometry.pose.pose.orientation.y,
            (float)currentOdometry.pose.pose.orientation.w
        );

        transform.position = position;
        transform.rotation = rotation;
    }

    Transform FindJointTransform(string jointName)
    {
        // Find joint in robot hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == jointName)
                return child;
        }
        return null;
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (!isConnected) return;

        var cmd = new TwistMsg();
        cmd.linear = new Vector3Msg(linearX, 0, 0);
        cmd.angular = new Vector3Msg(0, 0, angularZ);

        ros.Publish(cmdVelTopic, cmd);
    }

    public void SendJointCommands(float[] positions)
    {
        if (!isConnected) return;

        var cmd = new JointStateMsg();
        cmd.position = new System.Collections.Generic.List<double>();

        foreach (float pos in positions)
        {
            cmd.position.Add(pos);
        }

        ros.Publish("/joint_commands", cmd);
    }

    void ProcessCameraImage(ImageMsg image)
    {
        // Forward image to camera visualizer
        CameraFeedVisualizer cameraVis = FindObjectOfType<CameraFeedVisualizer>();
        if (cameraVis != null)
        {
            cameraVis.OnImageReceived(image);
        }
    }

    void OnApplicationQuit()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

## Performance Optimization

### Rendering Optimization

```csharp
// Example: Performance optimization for large-scale visualization
using UnityEngine;
using System.Collections.Generic;

public class VisualizationOptimizer : MonoBehaviour
{
    [Header("LOD Settings")]
    public float lodDistance = 10.0f;
    public int maxVisibleAgents = 50;

    [Header("Culling Settings")]
    public float frustumCullDistance = 50.0f;
    public bool enableOcclusionCulling = true;

    [Header("Batching Settings")]
    public bool enableDynamicBatching = true;
    public bool enableStaticBatching = true;

    private List<Renderer> agentRenderers = new List<Renderer>();
    private Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;
        SetupOptimization();
    }

    void SetupOptimization()
    {
        // Configure Unity rendering settings
        QualitySettings.vSyncCount = 0; // Disable vsync for performance
        Application.targetFrameRate = 60;

        // Set up level of detail
        SetupLODGroups();

        // Configure occlusion culling
        if (enableOcclusionCulling)
        {
            SetupOcclusionCulling();
        }
    }

    void SetupLODGroups()
    {
        // Create LOD groups for agents
        foreach (GameObject agent in FindObjectsOfType<GameObject>())
        {
            if (agent.name.Contains("Agent"))
            {
                SetupAgentLOD(agent);
            }
        }
    }

    void SetupAgentLOD(GameObject agent)
    {
        LODGroup lodGroup = agent.GetComponent<LODGroup>();
        if (lodGroup == null)
        {
            lodGroup = agent.AddComponent<LODGroup>();
        }

        // Create LOD levels
        LOD[] lods = new LOD[3];

        // High detail (close)
        Renderer[] highDetailRenderers = agent.GetComponentsInChildren<Renderer>();
        lods[0] = new LOD(0.5f, highDetailRenderers);

        // Medium detail (medium distance)
        Renderer[] mediumDetailRenderers = GetMediumDetailRenderers(agent);
        lods[1] = new LOD(0.2f, mediumDetailRenderers);

        // Low detail (far)
        Renderer[] lowDetailRenderers = GetLowDetailRenderers(agent);
        lods[2] = new LOD(0.01f, lowDetailRenderers);

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }

    Renderer[] GetMediumDetailRenderers(GameObject agent)
    {
        // Return simplified renderers for medium distance
        List<Renderer> renderers = new List<Renderer>();
        foreach (Renderer r in agent.GetComponentsInChildren<Renderer>())
        {
            if (ShouldIncludeInMediumLOD(r))
                renderers.Add(r);
        }
        return renderers.ToArray();
    }

    Renderer[] GetLowDetailRenderers(GameObject agent)
    {
        // Return minimal renderers for far distance
        Renderer mainRenderer = agent.GetComponent<Renderer>();
        return mainRenderer != null ? new Renderer[] { mainRenderer } : new Renderer[0];
    }

    bool ShouldIncludeInMediumLOD(Renderer renderer)
    {
        // Determine if renderer should be included in medium LOD
        return renderer.name != "DetailObject"; // Example: exclude detail objects
    }

    void SetupOcclusionCulling()
    {
        // Mark static objects for occlusion culling
        foreach (GameObject obj in GameObject.FindGameObjectsWithTag("Static"))
        {
            obj.GetComponent<Renderer>().receiveGI = ReceiveGI.Lightmaps;
        }
    }

    void Update()
    {
        OptimizeAgentRendering();
    }

    void OptimizeAgentRendering()
    {
        // Frustum culling
        foreach (Renderer renderer in agentRenderers)
        {
            if (renderer != null)
            {
                renderer.enabled = IsInFrustum(renderer.bounds);
            }
        }

        // Distance-based culling
        if (mainCamera != null)
        {
            foreach (Renderer renderer in agentRenderers)
            {
                if (renderer != null)
                {
                    float distance = Vector3.Distance(
                        mainCamera.transform.position,
                        renderer.bounds.center
                    );

                    renderer.enabled = distance <= frustumCullDistance;
                }
            }
        }
    }

    bool IsInFrustum(Bounds bounds)
    {
        if (mainCamera == null) return true;

        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(mainCamera);
        return GeometryUtility.TestPlanesAABB(planes, bounds);
    }
}
```

## Practical Applications in Humanoid Robotics

### Teleoperation Interface

```csharp
// Example: Advanced teleoperation interface
using UnityEngine;
using UnityEngine.UI;

public class HumanoidTeleoperationInterface : MonoBehaviour
{
    [Header("Teleoperation Controls")]
    public Transform robotBase;
    public Transform cameraRig;
    public Canvas controlCanvas;

    [Header("Input Mapping")]
    public KeyCode walkForwardKey = KeyCode.W;
    public KeyCode walkBackwardKey = KeyCode.S;
    public KeyCode turnLeftKey = KeyCode.A;
    public KeyCode turnRightKey = KeyCode.D;
    public KeyCode jumpKey = KeyCode.Space;

    [Header("Teleoperation Settings")]
    public float walkSpeed = 2.0f;
    public float turnSpeed = 50.0f;
    public float jumpForce = 5.0f;

    private CharacterController characterController;
    private Vector3 movementDirection;
    private bool isGrounded;

    void Start()
    {
        characterController = robotBase.GetComponent<CharacterController>();
        if (characterController == null)
        {
            characterController = robotBase.gameObject.AddComponent<CharacterController>();
        }
    }

    void Update()
    {
        HandleTeleoperationInput();
        ApplyMovement();
    }

    void HandleTeleoperationInput()
    {
        movementDirection = Vector3.zero;

        // Movement input
        if (Input.GetKey(walkForwardKey))
            movementDirection += robotBase.forward;
        if (Input.GetKey(walkBackwardKey))
            movementDirection -= robotBase.forward;
        if (Input.GetKey(turnLeftKey))
            robotBase.Rotate(0, -turnSpeed * Time.deltaTime, 0);
        if (Input.GetKey(turnRightKey))
            robotBase.Rotate(0, turnSpeed * Time.deltaTime, 0);

        // Normalize movement direction
        movementDirection = Vector3.ProjectOnPlane(movementDirection, Vector3.up).normalized;

        // Jump input
        if (Input.GetKeyDown(jumpKey) && isGrounded)
        {
            movementDirection.y = jumpForce;
            isGrounded = false;
        }
    }

    void ApplyMovement()
    {
        // Apply gravity
        if (!isGrounded)
        {
            movementDirection.y -= 9.81f * Time.deltaTime;
        }

        // Move the robot
        Vector3 movement = movementDirection * walkSpeed * Time.deltaTime;
        characterController.Move(movement);

        // Update grounded state
        isGrounded = characterController.isGrounded;
        if (isGrounded && movementDirection.y < 0)
        {
            movementDirection.y = 0;
        }
    }

    public void SetTeleoperationMode(bool enabled)
    {
        controlCanvas.enabled = enabled;
        Cursor.visible = enabled;
        Cursor.lockState = enabled ? CursorLockMode.None : CursorLockMode.Locked;
    }

    public void SendTeleoperationCommand(Vector3 command)
    {
        // Send command to actual robot via ROS
        var cmd = new RosMessageTypes.Geometry.TwistMsg();
        cmd.linear = new RosMessageTypes.Geometry.Vector3Msg(
            command.x, command.y, command.z
        );
        cmd.angular = new RosMessageTypes.Geometry.Vector3Msg(0, 0, 0);

        ROSConnection.GetOrCreateInstance().Publish("/cmd_vel", cmd);
    }
}
```

### Training Environment

```csharp
// Example: Training environment for humanoid robot learning
using UnityEngine;
using System.Collections;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class HumanoidTrainingEnvironment : Agent
{
    [Header("Training Settings")]
    public Transform target;
    public float targetRadius = 1.0f;
    public float maxDistance = 10.0f;

    [Header("Reward Settings")]
    public float reachTargetReward = 10.0f;
    public float distanceRewardMultiplier = 0.1f;
    public float timePenalty = -0.01f;

    private HumanoidRobotController robotController;
    private float episodeStartTime;

    public override void Initialize()
    {
        robotController = GetComponent<HumanoidRobotController>();
        episodeStartTime = Time.time;
    }

    public override void OnEpisodeBegin()
    {
        // Reset robot position
        transform.position = new Vector3(
            Random.Range(-maxDistance/2, maxDistance/2),
            0,
            Random.Range(-maxDistance/2, maxDistance/2)
        );

        // Reset target position
        target.position = new Vector3(
            Random.Range(-maxDistance/2, maxDistance/2),
            0,
            Random.Range(-maxDistance/2, maxDistance/2)
        );

        // Reset robot state
        robotController.Reset();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Collect observations for the agent
        sensor.AddObservation(transform.position); // Current position
        sensor.AddObservation(target.position);   // Target position
        sensor.AddObservation(GetDistanceToTarget()); // Distance to target
        sensor.AddObservation(robotController.GetJointPositions()); // Joint positions
        sensor.AddObservation(robotController.GetJointVelocities()); // Joint velocities
        sensor.AddObservation(robotController.GetBodyVelocity()); // Body velocity
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Apply actions to the robot
        float[] continuousActions = actions.ContinuousActions.ToArray();
        robotController.ApplyActions(continuousActions);

        // Calculate reward
        float distanceToTarget = GetDistanceToTarget();
        float distanceReward = (maxDistance - distanceToTarget) * distanceRewardMultiplier;

        SetReward(timePenalty + distanceReward);

        // Check if target reached
        if (distanceToTarget < targetRadius)
        {
            SetReward(GetReward() + reachTargetReward);
            EndEpisode();
        }

        // Check if episode should end due to time limit
        if (Time.time - episodeStartTime > 30.0f) // 30 seconds max
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // For testing with keyboard input
        var continuousActionsOut = actionsOut.ContinuousActions;

        continuousActionsOut[0] = Input.GetAxis("Horizontal"); // Move X
        continuousActionsOut[1] = Input.GetAxis("Vertical");   // Move Z
    }

    float GetDistanceToTarget()
    {
        return Vector3.Distance(transform.position, target.position);
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Target"))
        {
            SetReward(GetReward() + reachTargetReward);
            EndEpisode();
        }
    }
}
```

## Best Practices for Unity Robotics

### Architecture and Design Patterns

```csharp
// Example: Unity robotics architecture using SOA (Service-Oriented Architecture)
using UnityEngine;
using System.Collections.Generic;

public class UnityRoboticsSystem : MonoBehaviour
{
    private Dictionary<string, IRoboticsService> services;

    void Start()
    {
        InitializeServices();
    }

    void InitializeServices()
    {
        services = new Dictionary<string, IRoboticsService>();

        // Add services
        services.Add("visualization", new VisualizationService());
        services.Add("communication", new CommunicationService());
        services.Add("input", new InputService());
        services.Add("physics", new PhysicsService());
    }

    void Update()
    {
        // Update all services
        foreach (var service in services.Values)
        {
            service.Update();
        }
    }

    public T GetService<T>() where T : class
    {
        foreach (var service in services.Values)
        {
            if (service is T)
                return service as T;
        }
        return null;
    }
}

public interface IRoboticsService
{
    void Initialize();
    void Update();
    void Shutdown();
}

public class VisualizationService : IRoboticsService
{
    public void Initialize() { /* Initialize visualization system */ }
    public void Update() { /* Update visualization */ }
    public void Shutdown() { /* Clean up visualization */ }
}

public class CommunicationService : IRoboticsService
{
    public void Initialize() { /* Initialize ROS communication */ }
    public void Update() { /* Handle ROS messages */ }
    public void Shutdown() { /* Close ROS connections */ }
}

public class InputService : IRoboticsService
{
    public void Initialize() { /* Initialize input handling */ }
    public void Update() { /* Process input */ }
    public void Shutdown() { /* Clean up input */ }
}

public class PhysicsService : IRoboticsService
{
    public void Initialize() { /* Initialize physics */ }
    public void Update() { /* Update physics */ }
    public void Shutdown() { /* Clean up physics */ }
}
```

## Summary

Unity provides a powerful platform for advanced visualization and interaction in Physical AI systems, offering photorealistic rendering, sophisticated interaction mechanisms, and integration capabilities with ROS 2. The platform's flexibility allows for the creation of immersive training environments, intuitive teleoperation interfaces, and complex multi-agent simulations that enhance the development and deployment of humanoid robotics systems.

The integration of Unity with robotics workflows enables the creation of sophisticated visualization tools that support both development and operational phases of robotic systems. Proper implementation of rendering optimization, interaction systems, and communication protocols ensures that Unity-based visualization systems can handle the complexity and real-time requirements of Physical AI applications.

## Further Reading

- Unity Robotics Hub Documentation: https://github.com/Unity-Technologies/Unity-Robotics-Hub
- Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
- ROS# (ROS Sharp): https://github.com/syuntoku14/ROS-TCP-Endpoint
- "Game Engine Architecture" by Jason Gregory
- "Real-Time Rendering" by Tomas Akenine-Möller et al.