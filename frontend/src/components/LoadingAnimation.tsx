import React from 'react';
import { motion } from 'framer-motion';

interface LoadingAnimationProps {
  isLoading: boolean;
}

const LoadingAnimation: React.FC<LoadingAnimationProps> = ({ isLoading }) => {
  if (!isLoading) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
    >
      <div className="relative">
        {/* Outer rotating ring */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          className="w-32 h-32 rounded-full border-4 border-t-blue-500 border-r-blue-400 border-b-blue-300 border-l-blue-200"
        />
        
        {/* Inner rotating ring */}
        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="absolute inset-0 m-4 rounded-full border-4 border-t-purple-500 border-r-purple-400 border-b-purple-300 border-l-purple-200"
        />

        {/* Center pulsing circle */}
        <motion.div
          animate={{ 
            scale: [1, 1.2, 1],
            opacity: [0.5, 1, 0.5]
          }}
          transition={{ 
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute inset-0 m-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full shadow-lg"
        />

        {/* Loading text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            animate={{ 
              opacity: [0.5, 1, 0.5]
            }}
            transition={{ 
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="text-white font-mono text-sm"
          >
            Analyzing...
          </motion.div>
        </div>

        {/* Floating particles */}
        <div className="absolute inset-0">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              initial={{ 
                x: Math.random() * 100 - 50,
                y: Math.random() * 100 - 50,
                opacity: 0
              }}
              animate={{ 
                x: Math.random() * 100 - 50,
                y: Math.random() * 100 - 50,
                opacity: [0, 1, 0]
              }}
              transition={{ 
                duration: Math.random() * 2 + 1,
                repeat: Infinity,
                repeatType: "reverse"
              }}
              className="absolute w-2 h-2 bg-blue-400 rounded-full"
            />
          ))}
        </div>
      </div>

      {/* Progress messages */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute bottom-20 text-center"
      >
        <div className="text-blue-400 font-mono text-sm mb-2">
          Processing your content
        </div>
        <div className="text-blue-300/60 text-xs">
          This may take a few moments...
        </div>
      </motion.div>
    </motion.div>
  );
};

export default LoadingAnimation; 