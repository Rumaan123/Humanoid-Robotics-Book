import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/Humanoid-Robotics-Book/docs',
    component: ComponentCreator('/Humanoid-Robotics-Book/docs', 'f66'),
    routes: [
      {
        path: '/Humanoid-Robotics-Book/docs',
        component: ComponentCreator('/Humanoid-Robotics-Book/docs', '29c'),
        routes: [
          {
            path: '/Humanoid-Robotics-Book/docs',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs', 'efd'),
            routes: [
              {
                path: '/Humanoid-Robotics-Book/docs/conclusion',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/conclusion', '32e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/content-validation',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/content-validation', '816'),
                exact: true
              },
              {
                path: '/Humanoid-Robotics-Book/docs/intro',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/intro', '42f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-1',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-1', 'b6f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-1/control-pipelines',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-1/control-pipelines', 'eba'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-1/core-concepts',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-1/core-concepts', 'ab1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-1/python-control',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-1/python-control', 'd38'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-1/role-of-ros2',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-1/role-of-ros2', 'ac3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-1/urdf-modeling',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-1/urdf-modeling', '8e3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-2',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-2', '097'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-2/gazebo-simulation',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-2/gazebo-simulation', 'b68'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-2/physics-simulation',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-2/physics-simulation', 'c9a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-2/purpose-of-digital-twins',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-2/purpose-of-digital-twins', '7e4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-2/sensor-simulation',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-2/sensor-simulation', '816'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-2/unity-visualization',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-2/unity-visualization', '2e6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-3',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-3', '361'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-3/isaac-ros',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-3/isaac-ros', 'f0d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-3/isaac-sim',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-3/isaac-sim', 'c30'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-3/nav2-planning',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-3/nav2-planning', 'a9d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-3/perception-navigation',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-3/perception-navigation', 'c1d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-3/sim-to-real',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-3/sim-to-real', 'e4b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-4',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-4', 'ebb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-4/capstone-overview',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-4/capstone-overview', 'a07'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-4/cognitive-planning',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-4/cognitive-planning', '23b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-4/llm-integration',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-4/llm-integration', '3b1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-4/multi-modal',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-4/multi-modal', 'c4f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module-4/voice-action-pipelines',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module-4/voice-action-pipelines', 'f16'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/Humanoid-Robotics-Book/docs/module1-intro',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/module1-intro', '6eb'),
                exact: true
              },
              {
                path: '/Humanoid-Robotics-Book/docs/references',
                component: ComponentCreator('/Humanoid-Robotics-Book/docs/references', '25e'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/Humanoid-Robotics-Book/',
    component: ComponentCreator('/Humanoid-Robotics-Book/', '238'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
