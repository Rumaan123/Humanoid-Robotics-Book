
/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1/index',
        'module-1/role-of-ros2',
        'module-1/core-concepts',
        'module-1/python-control',
        'module-1/urdf-modeling',
        'module-1/control-pipelines'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2/index',
        'module-2/purpose-of-digital-twins',
        'module-2/physics-simulation',
        'module-2/gazebo-simulation',
        'module-2/sensor-simulation',
        'module-2/unity-visualization'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-3/index',
        'module-3/perception-navigation',
        'module-3/isaac-sim',
        'module-3/isaac-ros',
        'module-3/nav2-planning',
        'module-3/sim-to-real'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4/index',
        'module-4/llm-integration',
        'module-4/voice-action-pipelines',
        'module-4/cognitive-planning',
        'module-4/multi-modal',
        'module-4/capstone-overview'
      ],
    },
    'conclusion',
    'references'
  ],
};

module.exports = sidebars;