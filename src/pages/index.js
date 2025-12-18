import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title="Physical AI & Humanoid Robotics"
      description="Your Comprehensive Guide to Intelligent Machines"
    >
      {/* 🎯 MODERN HERO SECTION WITH BACKGROUND IMAGE */}
      <div style={{
        padding: '8rem 2rem',
        textAlign: 'center',
        background: `
          linear-gradient(135deg, 
            rgba(15, 23, 42, 0.85) 0%, 
            rgba(30, 41, 59, 0.9) 100%
          ),
          url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80')
        `,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
        color: 'white',
        position: 'relative',
        overflow: 'hidden',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        
        {/* ANIMATED GRADIENT OVERLAY */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(circle at 20% 30%, rgba(96, 165, 250, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
            linear-gradient(to right, 
              rgba(59, 130, 246, 0.1) 1px, 
              transparent 1px
            ),
            linear-gradient(to bottom, 
              rgba(59, 130, 246, 0.1) 1px, 
              transparent 1px
            )
          `,
          backgroundSize: '50px 50px',
          animation: 'gridMove 20s linear infinite',
          pointerEvents: 'none'
        }}></div>

        {/* GLOWING ORB EFFECT */}
        <div style={{
          position: 'absolute',
          width: '500px',
          height: '500px',
          background: 'radial-gradient(circle, rgba(96, 165, 250, 0.2) 0%, transparent 70%)',
          filter: 'blur(60px)',
          top: '20%',
          left: '10%',
          animation: 'orbFloat 15s ease-in-out infinite'
        }}></div>

        <div style={{
          position: 'relative',
          zIndex: 2,
          maxWidth: '900px'
        }}>
          <h1 style={{
            fontSize: '4.5rem',
            marginBottom: '1.5rem',
            fontWeight: '800',
            background: 'linear-gradient(to right, #fff, #60a5fa)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            textShadow: '0 2px 10px rgba(96, 165, 250, 0.3)'
          }}>
            {siteConfig.title}
          </h1>
          
          <p style={{
            fontSize: '1.8rem',
            marginBottom: '3rem',
            opacity: 0.95,
            lineHeight: 1.6,
            textShadow: '0 2px 4px rgba(0,0,0,0.3)',
            background: 'linear-gradient(to right, #cbd5e1, #94a3b8)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text'
          }}>
            {siteConfig.tagline}
          </p>
          
          <Link
            className="button button--primary button--lg"
            to="/docs/intro"
            style={{
              fontSize: '1.3rem',
              padding: '1.2rem 3rem',
              borderRadius: '50px',
              background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
              color: 'white',
              border: 'none',
              fontWeight: 'bold',
              boxShadow: `
                0 10px 30px rgba(59, 130, 246, 0.4),
                0 0 0 1px rgba(255,255,255,0.1),
                inset 0 1px 0 rgba(255,255,255,0.2)
              `,
              position: 'relative',
              overflow: 'hidden',
              zIndex: 1,
              transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-5px) scale(1.05)';
              e.currentTarget.style.boxShadow = `
                0 20px 40px rgba(59, 130, 246, 0.6),
                0 0 0 1px rgba(255,255,255,0.2),
                inset 0 1px 0 rgba(255,255,255,0.3)
              `;
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0) scale(1)';
              e.currentTarget.style.boxShadow = `
                0 10px 30px rgba(59, 130, 246, 0.4),
                0 0 0 1px rgba(255,255,255,0.1),
                inset 0 1px 0 rgba(255,255,255,0.2)
              `;
            }}
          >
            {/* BUTTON GLOW EFFECT */}
            <span style={{
              position: 'absolute',
              top: 0,
              left: '-100%',
              width: '100%',
              height: '100%',
              background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
              transition: 'left 0.7s ease'
            }}> 
            </span>
            
            <span style={{ position: 'relative', zIndex: 2 }}>
              🚀 Start Reading Now
            </span>
          </Link>

          {/* SCROLL INDICATOR */}
          <div style={{
            marginTop: '5rem',
            animation: 'bounce 2s infinite'
          }}>
            <div style={{
              width: '30px',
              height: '50px',
              border: '2px solid rgba(255,255,255,0.3)',
              borderRadius: '20px',
              margin: '0 auto',
              position: 'relative'
            }}>
              <div style={{
                width: '6px',
                height: '6px',
                background: 'white',
                borderRadius: '50%',
                position: 'absolute',
                left: '50%',
                top: '10px',
                transform: 'translateX(-50%)',
                animation: 'scroll 2s infinite'
              }}></div>
            </div>
          </div>
        </div>

        {/* ANIMATION STYLES INLINE */}
        <style>{`
          @keyframes gridMove {
            0% { background-position: 0 0; }
            100% { background-position: 50px 50px; }
          }
          
          @keyframes orbFloat {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(30px, -30px) scale(1.1); }
          }
          
          @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
          }
          
          @keyframes scroll {
            0% { opacity: 1; top: 10px; }
            100% { opacity: 0; top: 30px; }
          }
        `}</style>
      </div>

      {/* 📚 BOOK MODULES SECTION */}
      <div style={{ padding: '5rem 2rem', background: '#f8f9fa' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <h2 style={{
            textAlign: 'center',
            fontSize: '2.5rem',
            marginBottom: '3rem',
            color: '#333'
          }}>
            What You'll Learn
          </h2>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr', // ✅ 2 equal columns
            gap: '2rem',
            maxWidth: '1000px',
            margin: '0 auto'
          }}>
            {[
              {
                title: 'Module 1: The Robotic Nervous System',
                desc: 'Learn ROS 2 middleware for humanoid robots, nodes, topics, services, and actions.',
                color: '#667eea'
              },
              {
                title: 'Module 2: The Digital Twin',
                desc: 'Explore Gazebo and Unity simulations for safe robot development and testing.',
                color: '#764ba2'
              },
              {
                title: 'Module 3: The AI-Robot Brain',
                desc: 'Discover NVIDIA Isaac for perception, navigation, and sim-to-real transfer.',
                color: '#e53e3e'
              },
              {
                title: 'Module 4: Vision-Language-Action',
                desc: 'Integrate LLMs with robotics for natural human-robot interaction.',
                color: '#38a169'
              }
            ].map((module, index) => (
              <div key={index} style={{
                background: 'white',
                padding: '2rem',
                borderRadius: '12px',
                boxShadow: '0 5px 20px rgba(0,0,0,0.08)',
                transition: 'all 0.3s ease',
                borderTop: `4px solid ${module.color}`
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-10px)';
                e.currentTarget.style.boxShadow = '0 15px 30px rgba(0,0,0,0.15)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 5px 20px rgba(0,0,0,0.08)';
              }}
              >
                <div style={{
                  width: '50px',
                  height: '50px',
                  background: module.color,
                  borderRadius: '12px',
                  display: 'flex',
                  gridTemplateColumns: 'repeat(2, 1fr)', // 2 columns
                  gap: '3rem',
                  maxWidth: '1000px',
                  margin: '0 auto',
                  alignItems: 'center', // ✅ Center
                  justifyContent: 'center',
                  marginBottom: '1rem',
                  color: 'white',
                  fontSize: '1.5rem',
                  fontWeight: 'bold'
                }}>
                  {index + 1}
                </div>
                <h3 style={{
                  fontSize: '1.5rem',
                  marginBottom: '1rem',
                  color: '#333'
                }}>
                  {module.title}
                </h3>
                <p style={{ color: '#666', lineHeight: 1.6 }}>
                  {module.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
}