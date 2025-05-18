import React from 'react';

const AnimatedBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 -z-10">
      <div className="absolute inset-0 bg-gradient-to-br from-dark-primary via-dark-secondary to-dark-accent opacity-90" />
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_120%,rgba(120,119,198,0.15),rgba(255,255,255,0))] transition-opacity duration-1000" />
        <div className="absolute inset-0 animate-gradient-xy">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 animate-gradient-x" />
          <div className="absolute inset-0 bg-gradient-to-b from-blue-500/10 via-purple-500/10 to-pink-500/10 animate-gradient-y" />
        </div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_0%_0%,rgba(120,119,198,0.15),rgba(255,255,255,0))] transition-opacity duration-1000" />
      </div>
      <div className="absolute inset-0 backdrop-blur-[120px]" />
    </div>
  );
};

export default AnimatedBackground; 