import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/Humanoid-Robotics-Book/docs',
    component: ComponentCreator('/Humanoid-Robotics-Book/docs', 'cec'),
    routes: [
      {
        path: '/Humanoid-Robotics-Book/docs',
        component: ComponentCreator('/Humanoid-Robotics-Book/docs', 'beb'),
        routes: [
          {
            path: '/Humanoid-Robotics-Book/docs/tags',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags', '232'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/actions',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/actions', 'cf1'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/ai',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/ai', '7af'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/bibliography',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/bibliography', '009'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/citations',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/citations', '065'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/cognitive',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/cognitive', 'da8'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/communication',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/communication', 'c52'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/conclusion',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/conclusion', '225'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/digital-twin',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/digital-twin', '4e8'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/education',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/education', 'a33'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/gazebo',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/gazebo', 'cd0'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/interaction',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/interaction', 'f16'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/isaac',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/isaac', 'c68'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/learning-objectives',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/learning-objectives', '22a'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/middleware',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/middleware', '65f'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/navigation',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/navigation', 'f52'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/nodes',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/nodes', '853'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/perception',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/perception', '635'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/physical-ai',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/physical-ai', '746'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/references',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/references', 'cda'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/robotics',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/robotics', '465'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/ros-2',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/ros-2', '4ee'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/services',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/services', '618'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/simulation',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/simulation', '01c'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/sources',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/sources', 'c7a'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/synthesis',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/synthesis', 'b32'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/topics',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/topics', '18a'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/unity',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/unity', '060'),
            exact: true
          },
          {
            path: '/Humanoid-Robotics-Book/docs/tags/vla',
            component: ComponentCreator('/Humanoid-Robotics-Book/docs/tags/vla', '157'),
            exact: true
          },
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
