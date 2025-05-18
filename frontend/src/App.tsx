import React, { useState, useEffect, useRef } from 'react';
import { DeepfakeResult } from './types';
import AnimatedBackground from './components/AnimatedBackground';
import Navbar from './components/Navbar';
import DynamicDetectionGallery from './components/DynamicDetectionGallery';
import { motion } from 'framer-motion';
import { AlertTriangle, Database, Lock, UserX, Wifi, BarChart3, Server, FileLock2 } from 'lucide-react';
import ReportDialog from './components/ReportDialog';
import LoadingAnimation from './components/LoadingAnimation';

function DeepDefendHero() {
  // State for typewriter effect
  const [displayText, setDisplayText] = useState('');
  const [showEndSequence, setShowEndSequence] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  const heroRef = useRef<HTMLDivElement>(null);
  const deepfakeText = `> INITIATING DEEPFAKE ANALYSIS
> SCANNING FACIAL FEATURES...
> DETECTING FACE SWAP PATTERNS
> ANALYZING VOICE SYNTHESIS
> CHECKING FRAME CONSISTENCY
> IDENTIFYING MANIPULATION ARTIFACTS
> COMPARING FACIAL LANDMARKS
> VERIFYING SKIN TEXTURE PATTERNS
> ANALYZING EYE BLINK PATTERNS
> DETECTING LIP-SYNC ANOMALIES
> SCANNING FOR GAN ARTIFACTS
> VERIFYING LIGHTING CONSISTENCY
> CHECKING SHADOW INTEGRITY
> ANALYZING HAIR AND EDGE DETAILS
> DEEPFAKE DETECTED: 98.7% CONFIDENCE`;

  // Reset animation when component comes into view
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          setShowEndSequence(false);
          setDisplayText('');
          // Reset typewriter effect
          let currentIndex = 0;
          const interval = setInterval(() => {
            if (currentIndex <= deepfakeText.length) {
              setDisplayText(deepfakeText.substring(0, currentIndex));
              currentIndex++;
            } else {
              clearInterval(interval);
            }
          }, 60);

          // Reset end sequence timer
          const timer = setTimeout(() => {
            setShowEndSequence(true);
          }, 13600); // Total duration of typewriter + display

          return () => {
            clearInterval(interval);
            clearTimeout(timer);
          };
        } else {
          setIsVisible(false);
        }
      },
      {
        threshold: 0.5 // Trigger when 50% of the component is visible
      }
    );

    if (heroRef.current) {
      observer.observe(heroRef.current);
    }

    return () => {
      if (heroRef.current) {
        observer.unobserve(heroRef.current);
      }
    };
  }, []);

  // Random data points for visualization
  const dataPoints = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    delay: Math.random() * 8,
    size: Math.random() * 1.5 + 0.5,
  }));

  // Animation sequences - deepfake analysis progress
  const analysisSequences = [
    { label: "FACIAL FEATURE ANALYSIS", value: 100, color: "from-blue-400 to-blue-500", icon: <UserX className="w-4 h-4 text-blue-400" />, delay: 0.2 },
    { label: "VOICE PATTERN VERIFICATION", value: 100, color: "from-blue-400 to-blue-500", icon: <Wifi className="w-4 h-4 text-blue-400" />, delay: 0.8 },
    { label: "FRAME CONSISTENCY CHECK", value: 89, color: "from-blue-400 to-blue-500", icon: <Database className="w-4 h-4 text-blue-400" />, delay: 1.4 },
    { label: "GAN ARTIFACT DETECTION", value: 72, color: "from-blue-400 to-blue-500", icon: <BarChart3 className="w-4 h-4 text-blue-400" />, delay: 2.0 },
    { label: "DEEPFAKE CONFIRMATION", value: 98, color: "from-blue-400 to-blue-500", icon: <AlertTriangle className="w-4 h-4 text-blue-400" />, delay: 2.6 },
  ];

  return (
    <div ref={heroRef} className="relative h-screen w-full overflow-hidden flex items-center justify-center">
      {/* Animated data nodes */}
      {dataPoints.map((point) => (
        <motion.div
          key={point.id}
          className="absolute rounded-full bg-blue-500/30 backdrop-blur-sm"
          style={{
            top: `${point.y}%`,
            left: `${point.x}%`,
            width: `${point.size * 3}px`,
            height: `${point.size * 3}px`,
          }}
          initial={{ opacity: 0 }}
          animate={{ 
            opacity: [0, 0.4, 0],
            scale: [0, 1.2, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            delay: point.delay,
            repeatType: 'loop',
            ease: "easeInOut"
          }}
        />
      ))}
      
      {/* Data connection lines - horizontal */}
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={`h-line-${i}`}
          className="absolute h-px bg-gradient-to-r from-transparent via-blue-500/20 to-transparent"
          style={{
            top: `${12 + i * 12}%`,
            left: 0,
            right: 0,
          }}
          initial={{ opacity: 0, scaleX: 0 }}
          animate={{ 
            opacity: [0, 0.3, 0],
            scaleX: [0, 1, 0],
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            delay: i * 0.5,
            repeatType: 'loop',
            ease: "easeInOut"
          }}
        />
      ))}
      
      {/* Data connection lines - vertical */}
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={`v-line-${i}`}
          className="absolute w-px bg-gradient-to-b from-transparent via-blue-500/20 to-transparent"
          style={{
            left: `${12 + i * 12}%`,
            top: 0,
            bottom: 0,
          }}
          initial={{ opacity: 0, scaleY: 0 }}
          animate={{ 
            opacity: [0, 0.3, 0],
            scaleY: [0, 1, 0],
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            delay: i * 0.5,
            repeatType: 'loop',
            ease: "easeInOut"
          }}
        />
      ))}
      
      {/* Scanning beam */}
      <motion.div
        className="absolute w-full h-2 bg-gradient-to-r from-transparent via-blue-500/30 to-transparent blur-lg"
        animate={{ 
          y: ['-100%', '100%'],
          opacity: [0, 0.5, 0]
        }}
        transition={{ 
          y: { repeat: Infinity, duration: 5, ease: "linear" },
          opacity: { repeat: Infinity, duration: 5, ease: "easeInOut" }
        }}
      />
      
      {/* Terminal screen on the left */}
      <motion.div 
        className="absolute top-1/2 left-16 transform -translate-y-1/2 w-96 h-96 bg-black/80 border border-blue-500/50 rounded-lg overflow-hidden p-4 z-10"
        initial={{ opacity: 0, x: -50 }}
        animate={{ 
          opacity: showEndSequence ? 0 : 1, 
          x: showEndSequence ? -200 : 0 
        }}
        transition={{ 
          opacity: showEndSequence ? { duration: 1.2 } : { duration: 0.8, delay: 0.5 },
          x: showEndSequence ? { duration: 1.5 } : { duration: 0.8, delay: 0.5 }
        }}
      >
        <div className="flex items-center gap-2 mb-2 border-b border-blue-500/30 pb-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-blue-400 text-sm font-mono">deepfake_analyzer.exe</span>
        </div>
        <pre className="text-blue-500 font-mono text-xs whitespace-pre-wrap h-full overflow-y-auto">
          {displayText}
        </pre>
      </motion.div>
      
      {/* Central analysis warning container */}
      <motion.div 
        className="relative z-10 w-96 bg-black/90 border-2 border-blue-500/50 rounded-xl p-6 shadow-2xl shadow-blue-500/30"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ 
          opacity: showEndSequence ? 0 : 1, 
          scale: showEndSequence ? 0.8 : 1,
          y: showEndSequence ? 50 : 0 
        }}
        transition={{ 
          opacity: showEndSequence ? { duration: 1.2 } : { duration: 0.8 },
          scale: showEndSequence ? { duration: 1.5 } : { duration: 0.8 },
          y: showEndSequence ? { duration: 1.5 } : { duration: 0.8 }
        }}
      >
        {/* Flashing warning header */}
        <motion.div 
          className="flex items-center gap-3 mb-6"
          animate={{ opacity: [1, 0.7, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <AlertTriangle className="w-6 h-6 text-blue-500" />
          <h2 className="text-2xl font-bold text-blue-500">DEEPFAKE DETECTED</h2>
        </motion.div>
        
        {/* Progress bars showing the analysis */}
        <div className="space-y-4">
          {analysisSequences.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: item.delay, duration: 0.5 }}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  {item.icon}
                  <span className="text-sm font-mono text-blue-300">{item.label}</span>
                </div>
                <motion.span 
                  className="text-sm font-mono text-blue-400"
                  animate={{ opacity: [1, 0.5, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  {item.value}%
                </motion.span>
              </div>
              <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full bg-gradient-to-r ${item.color} rounded-full`}
                  initial={{ width: '0%' }}
                  animate={{ width: `${item.value}%` }}
                  transition={{ delay: item.delay, duration: 1.5 }}
                />
              </div>
            </motion.div>
          ))}
        </div>
        
        {/* Analysis results */}
        <motion.div
          className="mt-6 pt-4 border-t border-blue-500/30"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 3.2, duration: 0.8 }}
        >
          <div className="flex justify-between items-center mb-2">
            <span className="text-blue-300 font-mono text-sm">DETECTION CONFIDENCE:</span>
            <motion.span 
              className="text-xl font-bold text-blue-500"
              animate={{ opacity: [1, 0.7, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              98.7%
            </motion.span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-blue-300 font-mono text-sm">MANIPULATION TYPE:</span>
            <motion.div 
              className="text-xl font-bold text-blue-500 font-mono"
              animate={{ opacity: [1, 0.7, 1] }}
              transition={{ duration: 0.5, repeat: Infinity }}
            >
              FACE SWAP
            </motion.div>
          </div>
        </motion.div>
      </motion.div>
      
      {/* Right side analysis panel */}
      <motion.div 
        className="absolute top-1/2 right-16 transform -translate-y-1/2 w-80 bg-black/80 border border-blue-500/40 rounded-lg p-4 z-10"
        initial={{ opacity: 0, x: 50 }}
        animate={{ 
          opacity: showEndSequence ? 0 : 1, 
          x: showEndSequence ? 200 : 0 
        }}
        transition={{ 
          opacity: showEndSequence ? { duration: 1.2 } : { duration: 0.8, delay: 0.7 },
          x: showEndSequence ? { duration: 1.5 } : { duration: 0.8, delay: 0.7 }
        }}
      >
        <h3 className="text-lg font-mono text-blue-400 mb-4 border-b border-blue-500/30 pb-2">Analysis Details</h3>
        
        <div className="space-y-3">
          {[
            { name: "Facial Landmarks", icon: <Server className="w-4 h-4" />, progress: 98 },
            { name: "Voice Patterns", icon: <FileLock2 className="w-4 h-4" />, progress: 95 },
            { name: "Frame Consistency", icon: <UserX className="w-4 h-4" />, progress: 92 },
            { name: "GAN Artifacts", icon: <Database className="w-4 h-4" />, progress: 97 },
          ].map((system, index) => (
            <motion.div 
              key={index} 
              className="flex items-center gap-3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1 + (index * 0.3) }}
            >
              <div className="text-blue-400">{system.icon}</div>
              <div className="flex-1">
                <div className="flex justify-between mb-1">
                  <span className="text-xs text-blue-300">{system.name}</span>
                  <span className="text-xs text-blue-400">{system.progress}%</span>
                </div>
                <div className="h-1 w-full bg-gray-800 rounded-full">
                  <motion.div 
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full"
                    initial={{ width: '0%' }}
                    animate={{ width: `${system.progress}%` }}
                    transition={{ delay: 1.2 + (index * 0.3), duration: 1 }}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
      
      {/* Title Overlay - moves to center after fading */}
      
      <motion.div
        className="absolute w-full text-center z-30"
        initial={{ opacity: 0, y: -20, top: "8px" }}
        animate={{ 
          opacity: 1, 
          y: 0,
          top: showEndSequence ? "50%" : "8px",
          translateY: showEndSequence ? "-50%" : "0%",
          scale: showEndSequence ? 1.5 : 1
        }}
        transition={{ 
          opacity: { duration: 0.7 },
          y: { duration: 0.7 },
          top: showEndSequence ? { duration: 1.8, ease: "easeInOut" } : { duration: 0 },
          translateY: showEndSequence ? { duration: 1.8, ease: "easeInOut" } : { duration: 0 },
          scale: showEndSequence ? { duration: 1.8, ease: "easeInOut" } : { duration: 0 }
        }}
      >
        <h1 className="text-5xl sm:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-blue-700 drop-shadow-lg">
          DeepDefend
        </h1>
        <motion.p 
          className="text-blue-300/90 text-xl mt-2 drop-shadow-sm"
          animate={{ 
            opacity: showEndSequence ? 0 : 1
          }}
          transition={{ 
            opacity: showEndSequence ? { duration: 0.8 } : { duration: 0.7 }
          }}
        >
        </motion.p>
      </motion.div>
      
      {/* Blue alert flashes */}
      <motion.div
        className="absolute inset-0 bg-blue-900/10 z-5 pointer-events-none"
        animate={{ 
          opacity: showEndSequence ? 0 : [0, 0.2, 0]
        }}
        transition={{ 
          duration: 3, 
          repeat: showEndSequence ? 0 : Infinity 
        }}
      />
      
      {/* Final animation background glow for DeepDefend title */}
      {showEndSequence && (
        <motion.div
          className="absolute inset-0 z-20 pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2 }}
          style={{
            background: 'radial-gradient(circle at center, rgba(59, 130, 246, 0.2) 0%, rgba(0, 0, 0, 0) 70%)'
          }}
        />
      )}
    </div>
  );
}

// Global background component that matches the hero section
function GlobalBackground() {
  // Random data points for visualization
  const dataPoints = Array.from({ length: 30 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    delay: Math.random() * 5,
    size: Math.random() * 2 + 1,
  }));

  return (
    <div className="fixed inset-0 z-0 pointer-events-none">
      {/* Advanced grid background with depth effect */}
      <div className="absolute inset-0 z-0">
        <div 
          className="absolute inset-0 z-0" 
          style={{
            backgroundImage: `linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)`,
            backgroundSize: '20px 20px',
          }}
        />
        <div 
          className="absolute inset-0 z-0" 
          style={{
            backgroundImage: `linear-gradient(rgba(59, 130, 246, 0.05) 2px, transparent 2px),
                            linear-gradient(90deg, rgba(59, 130, 246, 0.05) 2px, transparent 2px)`,
            backgroundSize: '100px 100px',
          }}
        />
      </div>
      
      {/* Vignette overlay */}
      <div className="absolute inset-0 z-0" 
        style={{
          background: 'radial-gradient(circle at center, transparent 10%, rgba(0, 0, 0, 0.8) 100%)'
        }}
      />
      
      {/* Animated data nodes */}
      {dataPoints.map((point) => (
        <motion.div
          key={point.id}
          className="absolute rounded-full bg-blue-500"
          style={{
            top: `${point.y}%`,
            left: `${point.x}%`,
            width: `${point.size * 3}px`,
            height: `${point.size * 3}px`,
          }}
          initial={{ opacity: 0 }}
          animate={{ 
            opacity: [0, 0.8, 0],
            scale: [0, 1.5, 0],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            delay: point.delay,
            repeatType: 'loop',
          }}
        />
      ))}
      
      {/* Data connection lines - horizontal */}
      {[...Array(12)].map((_, i) => (
        <motion.div
          key={`h-line-${i}`}
          className="absolute h-px bg-gradient-to-r from-transparent via-blue-500/30 to-transparent"
          style={{
            top: `${8 + i * 8}%`,
            left: 0,
            right: 0,
          }}
          initial={{ opacity: 0, scaleX: 0 }}
          animate={{ 
            opacity: [0, 0.6, 0],
            scaleX: [0, 1, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            delay: i * 0.3,
            repeatType: 'loop',
          }}
        />
      ))}
      
      {/* Data connection lines - vertical */}
      {[...Array(12)].map((_, i) => (
        <motion.div
          key={`v-line-${i}`}
          className="absolute w-px bg-gradient-to-b from-transparent via-blue-500/30 to-transparent"
          style={{
            left: `${8 + i * 8}%`,
            top: 0,
            bottom: 0,
          }}
          initial={{ opacity: 0, scaleY: 0 }}
          animate={{ 
            opacity: [0, 0.6, 0],
            scaleY: [0, 1, 0],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            delay: i * 0.3,
            repeatType: 'loop',
          }}
        />
      ))}
      
      {/* Scanning beam */}
      <motion.div
        className="absolute w-full h-3 bg-gradient-to-r from-transparent via-blue-500/50 to-transparent blur-md"
        animate={{ 
          y: ['-100%', '100%'],
          opacity: 1 
        }}
        transition={{ 
          y: { repeat: Infinity, duration: 3, ease: "linear" },
          opacity: { duration: 1.5, ease: "easeOut" }
        }}
      />
    </div>
  );
}

function App() {
  const [results, setResults] = useState<DeepfakeResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [activeTab, setActiveTab] = useState<'search' | 'report'>('search'); // Changed default to 'search'
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [timeLeft, setTimeLeft] = useState(5);
  const [isReportDialogOpen, setIsReportDialogOpen] = useState(false);
  const [selectedResult, setSelectedResult] = useState<DeepfakeResult | null>(null);

  // Sample images (you should replace these with actual images)
  const images = [
    "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80",
    "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80",
    "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80",
    "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80",
    "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80"
  ];

  const getVisibleImages = () => {
    const visible = [];
    for (let i = 0; i < 3; i++) {
      visible.push(images[(currentImageIndex + i) % images.length]);
    }
    return visible;
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          setCurrentImageIndex((prevIndex) => (prevIndex + 1) % images.length);
          return 5;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);

    if (activeTab === 'report') {
      // Report mode logic
      setTimeout(() => {
        const newResult: DeepfakeResult = {
          id: Math.random().toString(36).substr(2, 9),
          imageUrl: imageUrl,
          confidence: Math.random() * 100,
          isSelected: false,
          timestamp: new Date().toISOString(),
        };
        setResults(prev => [...prev, newResult]);
        setHasMore(true);
        setIsLoading(false);
      }, 1500);
    } else {
      // Search mode logic: call backend API
      try {
        const formData = new FormData();
        formData.append('file', file);
        
        console.log('Sending request to backend...');
        const response = await fetch('http://localhost:5002/api/search', {
          method: 'POST',
          body: formData,
        });
        
        console.log('Response received:', response.status);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: 'Unknown error occurred' }));
          throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Data received:', data);
        console.log('YouTube videos:', data.youtube_videos);
        console.log('Vimeo videos:', data.vimeo_videos);
        console.log('Dailymotion videos:', data.dailymotion_videos);
        
        // Transform the response to match our frontend's expected format
        const transformedResults: DeepfakeResult[] = data.map((result: any) => ({
          id: result.id,
          imageUrl: result.imageUrl,  // Use the imageUrl from the API response
          confidence: result.confidence,
          timestamp: result.timestamp,
          celebrity: result.celebrity,
          images: result.images || [],
          youtube_videos: result.youtube_videos || [],
          vimeo_videos: result.vimeo_videos || [],
          dailymotion_videos: result.dailymotion_videos || [],
          isSelected: false
        }));
        
        setResults(transformedResults);
        setHasMore(false);
      } catch (error) {
        console.error('Search error:', error);
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
          alert('Could not connect to the server. Please make sure the backend is running at http://localhost:5002');
        } else {
          alert(error instanceof Error ? error.message : 'Failed to search deepfakes. Please try again.');
        }
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleSearch = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!uploadedImage && !searchQuery) return;
    setIsLoading(true);
    
    if (activeTab === 'search' && searchQuery) {
      // Search by URL logic: call backend API
      try {
        console.log('Sending URL search request...');
        const response = await fetch('http://localhost:5002/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: searchQuery }),
        });
        
        console.log('Response received:', response.status);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: 'Unknown error occurred' }));
          throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Data received:', data);
        console.log('YouTube videos:', data.youtube_videos);
        console.log('Vimeo videos:', data.vimeo_videos);
        console.log('Dailymotion videos:', data.dailymotion_videos);
        
        // Transform the response to match our frontend's expected format
        const transformedResults: DeepfakeResult[] = data.map((result: any) => ({
          id: result.id,
          imageUrl: result.imageUrl,  // Use the imageUrl from the API response
          confidence: result.confidence,
          timestamp: result.timestamp,
          celebrity: result.celebrity,
          images: result.images || [],
          youtube_videos: result.youtube_videos || [],
          vimeo_videos: result.vimeo_videos || [],
          dailymotion_videos: result.dailymotion_videos || [],
          isSelected: false
        }));
        
        setResults(transformedResults);
        setHasMore(false);
      } catch (error) {
        console.error('Search error:', error);
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
          alert('Could not connect to the server. Please make sure the backend is running at http://localhost:5002');
        } else {
          alert(error instanceof Error ? error.message : 'Failed to search deepfakes. Please try again.');
        }
      } finally {
        setIsLoading(false);
      }
    } else {
      setIsLoading(false);
    }
  };

  const toggleSelection = (id: string) => {
    setResults(prev =>
      prev.map(result =>
        result.id === id ? { ...result, isSelected: !result.isSelected } : result
      )
    );
  };

  const handleReportSelected = () => {
    const selectedResults = results.filter(result => result.isSelected);
    console.log('Reporting selected results:', selectedResults);
    // TODO: Implement actual reporting functionality
  };

  const handleLoadMore = () => {
    // TODO: Implement actual load more functionality
    console.log('Loading more results...');
  };

  const handleGuess = (guess: 'real' | 'fake') => {
    // TODO: Implement actual answer checking
    const isCorrect = guess === 'fake'; // This is just a mock implementation
    alert(isCorrect ? 'Correct! This is indeed a deepfake video.' : 'Incorrect! This is actually a deepfake video.');
  };

  const handleReportContent = (result: DeepfakeResult) => {
    setSelectedResult(result);
    setIsReportDialogOpen(true);
  };

  const handleImageUpload = async (file: File) => {
    setIsLoading(true);
    try {
      // Your existing image upload and processing logic
      // ... existing code ...

      // After processing is complete
      setIsLoading(false);
    } catch (error) {
      console.error('Error processing image:', error);
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-primary via-dark-secondary to-dark-accent">
      <LoadingAnimation isLoading={isLoading} />
      <GlobalBackground />
      <AnimatedBackground />
      <Navbar />
      
      {/* Replace old hero section with new DeepDefendHero */}
      <DeepDefendHero />

      {/* Main Content */}
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Stats Section */}
        <div className="mb-20">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Stat 1 */}
            <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-6 transform transition-all duration-300 hover:scale-105 border border-blue-500/20">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
              <div className="relative">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-blue-400">96%</h3>
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500/80 to-blue-600/80 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                    </svg>
                  </div>
                </div>
                <p className="text-blue-300/80 text-sm font-mono">
                  of deepfake videos are used for malicious purposes, including fraud and misinformation
                </p>
              </div>
            </div>

            {/* Stat 2 */}
            <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-6 transform transition-all duration-300 hover:scale-105 border border-blue-500/20">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
              <div className="relative">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-blue-400">300%</h3>
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500/80 to-blue-600/80 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                    </svg>
                  </div>
                </div>
                <p className="text-blue-300/80 text-sm font-mono">
                  increase in deepfake content creation in the last year alone
                </p>
              </div>
            </div>

            {/* Stat 3 */}
            <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-6 transform transition-all duration-300 hover:scale-105 border border-blue-500/20">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
              <div className="relative">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-2xl font-bold text-blue-400">$250M</h3>
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500/80 to-blue-600/80 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                  </div>
                </div>
                <p className="text-blue-300/80 text-sm font-mono">
                  estimated annual losses due to deepfake-related fraud and scams
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Tabs */}
        <div className="mb-16">
          <div className="flex justify-center mb-8">
            <div className="inline-flex rounded-xl bg-black/40 backdrop-blur-sm p-1 border border-blue-500/20">
              <button
                onClick={() => setActiveTab('search')}
                className={`px-6 py-3 rounded-lg transition-all duration-300 ${
                  activeTab === 'search'
                    ? 'bg-gradient-to-r from-blue-500/70 to-blue-600/70 text-white/90 shadow-lg shadow-blue-500/5'
                    : 'text-blue-300/80 hover:text-white/90'
                }`}
              >
                Search Deepfakes
              </button>
              <button
                onClick={() => setActiveTab('report')}
                className={`px-6 py-3 rounded-lg transition-all duration-300 ${
                  activeTab === 'report'
                    ? 'bg-gradient-to-r from-blue-500/70 to-blue-600/70 text-white/90'
                    : 'text-blue-300/80 hover:text-white/90'
                }`}
              >
                Report Deepfakes
              </button>
            </div>
          </div>

          {activeTab === 'search' ? (
            // Search Deepfakes Section
            <div className="space-y-8">
              <div className="text-center mb-16">
                <h1 className="text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-500/80 to-blue-400/80 mb-4 font-mono">
                  Search Deepfakes
                </h1>
                <p className="text-blue-300/80 text-lg max-w-2xl mx-auto font-mono">
                  Upload an image to search for similar deepfakes across the internet
                </p>
              </div>

              <div className="max-w-3xl mx-auto">
                <form onSubmit={handleSearch} className="space-y-6">
                  <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-6 border border-blue-500/20">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
                    <div className="relative">
                      <div className="flex flex-col space-y-4">
                        <label
                          htmlFor="file-upload-search"
                          className="group relative flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-blue-500/30 rounded-2xl cursor-pointer bg-black/20 backdrop-blur-sm hover:bg-blue-500/10 transition-all duration-300 ease-in-out"
                        >
                          {uploadedImage ? (
                            <div className="relative w-full h-full">
                              <img
                                src={uploadedImage}
                                alt="Uploaded content"
                                className="w-full h-full object-contain rounded-xl"
                              />
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setUploadedImage(null);
                                }}
                                className="absolute top-2 right-2 p-2 bg-red-500/80 rounded-full hover:bg-red-600/80 transition-colors duration-300"
                              >
                                <svg className="w-5 h-5 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                                </svg>
                              </button>
                            </div>
                          ) : (
                            <div className="flex flex-col items-center justify-center pt-5 pb-6">
                              <div className="relative">
                                <div className="absolute -inset-1 rounded-full bg-gradient-to-r from-blue-500/10 to-blue-600/10 opacity-0 group-hover:opacity-20 blur-[2px] transition duration-500"></div>
                                <div className="relative">
                                  <svg className="w-12 h-12 mb-4 text-gray-400/80 group-hover:text-blue-400/80 transition-colors duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                                  </svg>
                                </div>
                              </div>
                              <p className="mb-2 text-lg text-gray-300/80 group-hover:text-white/90 transition-colors duration-300">
                                <span className="font-semibold">Click to upload</span> or drag and drop
                              </p>
                              <p className="text-sm text-gray-500/80 group-hover:text-gray-400/80 transition-colors duration-300">PNG, JPG or MP4</p>
                            </div>
                          )}
                          <input
                            id="file-upload-search"
                            type="file"
                            className="hidden"
                            accept="image/*,video/*"
                            onChange={handleFileUpload}
                          />
                        </label>

                        <div className="flex items-center space-x-4">
                          <div className="flex-1 relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                              <svg className="w-5 h-5 text-gray-400/80" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                              </svg>
                            </div>
                            <input
                              type="text"
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              placeholder="Or enter image URL..."
                              className="w-full bg-gray-900/30 border border-gray-700/30 rounded-xl pl-10 pr-4 py-3 text-white/90 placeholder-gray-400/80 focus:outline-none focus:border-blue-500/50 transition-colors duration-300"
                            />
                          </div>
                          <button
                            type="submit"
                            disabled={!uploadedImage && !searchQuery}
                            className={`group relative px-6 py-3 rounded-xl transition-all duration-300 ease-in-out transform hover:scale-105 ${
                              uploadedImage || searchQuery
                                ? 'bg-gradient-to-r from-blue-500/70 to-blue-600/70 hover:from-blue-600/70 hover:to-blue-700/70'
                                : 'bg-gray-700/50 cursor-not-allowed'
                            }`}
                          >
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-xl blur opacity-50 group-hover:opacity-70 transition duration-300"></div>
                            <div className="relative flex items-center space-x-2">
                              <svg className="w-5 h-5 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                              </svg>
                              <span className="text-white/90">Search</span>
                            </div>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="text-center">
                    <p className="text-blue-300/80 text-sm">
                      Upload an image or enter an image URL to search for similar deepfakes
                    </p>
                  </div>
                </form>
              </div>

              {isLoading && (
                <div className="text-center text-blue-400 font-mono my-8">Loading results...</div>
              )}

              {results.length > 0 && (
                <div className="grid grid-cols-1 gap-8 mt-8 max-w-[95%] mx-auto">
                  {results.map((result) => (
                    <div key={result.id} className="bg-black/40 rounded-xl p-8 border border-blue-500/20">
                      {/* Main image */}
                      {result.imageUrl && (
                        <div className="mb-8">
                          <img
                            src={result.imageUrl}
                            alt="Detection result"
                            className="w-full h-[600px] object-cover rounded-lg"
                          />
                        </div>
                      )}
                      {/* Additional images */}
                      {result.images && result.images.length > 0 && (
                        <div className="mb-8">
                          <h3 className="text-blue-400 font-mono mb-4 text-2xl">Related Images</h3>
                          <div className="grid grid-cols-2 gap-8">
                            {result.images.map((img: string, idx: number) => (
                              <img
                                key={idx}
                                src={img}
                                alt={`Related image ${idx+1}`}
                                className="w-full h-[500px] object-cover rounded-lg"
                              />
                            ))}
                          </div>
                        </div>
                      )}
                      {/* YouTube videos */}
                      {result.youtube_videos && result.youtube_videos.length > 0 && (
                        <div className="mb-8">
                          <h3 className="text-blue-400 font-mono mb-4 text-2xl">YouTube Videos</h3>
                          <div className="grid grid-cols-1 gap-8">
                            {result.youtube_videos.map((url: string, idx: number) => {
                              const videoId = url.split('v=')[1]?.split('&')[0];
                              const embedUrl = videoId ? `https://www.youtube.com/embed/${videoId}?rel=0` : url;
                              return (
                                <div key={idx} className="relative pb-[56.25%] h-0">
                                  <iframe
                                    src={embedUrl}
                                    title={`YouTube video ${idx+1}`}
                                    className="absolute top-0 left-0 w-full h-full rounded-lg"
                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                    allowFullScreen
                                  />
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      {/* Vimeo videos */}
                      {result.vimeo_videos && result.vimeo_videos.length > 0 && (
                        <div className="mb-8">
                          <h3 className="text-blue-400 font-mono mb-4 text-2xl">Vimeo Videos</h3>
                          <div className="grid grid-cols-1 gap-8">
                            {result.vimeo_videos.map((url: string, idx: number) => {
                              const videoId = url.split('vimeo.com/')[1]?.split('?')[0];
                              const embedUrl = videoId ? `https://player.vimeo.com/video/${videoId}?title=0&byline=0&portrait=0` : url;
                              return (
                                <div key={idx} className="relative pb-[56.25%] h-0">
                                  <iframe
                                    src={embedUrl}
                                    title={`Vimeo video ${idx+1}`}
                                    className="absolute top-0 left-0 w-full h-full rounded-lg"
                                    allow="autoplay; fullscreen; picture-in-picture"
                                    allowFullScreen
                                  />
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      {/* Dailymotion videos */}
                      {result.dailymotion_videos && result.dailymotion_videos.length > 0 && (
                        <div className="mb-8">
                          <h3 className="text-blue-400 font-mono mb-4 text-2xl">Dailymotion Videos</h3>
                          <div className="grid grid-cols-1 gap-8">
                            {result.dailymotion_videos.map((url: string, idx: number) => {
                              const videoId = url.split('dailymotion.com/video/')[1]?.split('?')[0];
                              const embedUrl = videoId ? `https://www.dailymotion.com/embed/video/${videoId}?autoplay=0&controls=1` : url;
                              return (
                                <div key={idx} className="relative pb-[56.25%] h-0">
                                  <iframe
                                    src={embedUrl}
                                    title={`Dailymotion video ${idx+1}`}
                                    className="absolute top-0 left-0 w-full h-full rounded-lg"
                                    allow="autoplay"
                                    allowFullScreen
                                  />
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      <div className="flex justify-between items-center mt-6">
                        <div className="text-blue-300 font-mono text-lg">
                          Confidence: {result.confidence?.toFixed(2)}%
                        </div>
                        <div className="text-gray-400 text-sm">
                          {new Date(result.timestamp).toLocaleString()}
                        </div>
                      </div>
                      {result.celebrity && (
                        <div className="text-blue-400 font-bold mt-3 text-xl">{result.celebrity}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            // Report Deepfakes Section
            <div className="space-y-8">
              <div className="text-center mb-16">
                <h1 className="text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-500/80 to-blue-400/80 mb-4 font-mono">
                  Report Deepfakes
                </h1>
                <p className="text-blue-300/80 text-lg max-w-2xl mx-auto font-mono">
                  Upload an image to detect and report potential deepfakes
                </p>
              </div>

              <div className="max-w-3xl mx-auto">
                <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-6 border border-blue-500/20">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
                  <div className="relative">
                    <label
                      htmlFor="file-upload-report"
                      className="group relative flex flex-col items-center justify-center w-full h-80 border-2 border-dashed border-blue-500/30 rounded-2xl cursor-pointer bg-black/20 backdrop-blur-sm hover:bg-blue-500/10 transition-all duration-300 ease-in-out"
                    >
                      {uploadedImage ? (
                        <div className="relative w-full h-full">
                          <img
                            src={uploadedImage}
                            alt="Uploaded content"
                            className="w-full h-full object-contain rounded-xl"
                          />
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setUploadedImage(null);
                            }}
                            className="absolute top-2 right-2 p-2 bg-red-500/80 rounded-full hover:bg-red-600/80 transition-colors duration-300"
                          >
                            <svg className="w-5 h-5 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                          </button>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <div className="relative">
                            <div className="absolute -inset-1 rounded-full bg-gradient-to-r from-blue-500/10 to-blue-600/10 opacity-0 group-hover:opacity-20 blur-[2px] transition duration-500"></div>
                            <div className="relative">
                              <svg className="w-16 h-16 mb-4 text-gray-400/80 group-hover:text-blue-400/80 transition-colors duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                              </svg>
                            </div>
                          </div>
                          <p className="mb-2 text-lg text-gray-300/80 group-hover:text-white/90 transition-colors duration-300">
                            <span className="font-semibold">Click to upload</span> or drag and drop
                          </p>
                          <p className="text-sm text-gray-500/80 group-hover:text-gray-400/80 transition-colors duration-300">PNG, JPG or MP4</p>
                        </div>
                      )}
                      <input
                        id="file-upload-report"
                        type="file"
                        className="hidden"
                        accept="image/*,video/*"
                        onChange={handleFileUpload}
                      />
                    </label>
                  </div>
                </div>
              </div>

              <div className="flex justify-between items-center mb-8">
                <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500/80 to-blue-400/80 font-mono">
                  Detection Results
                </h2>
                <button
                  onClick={handleReportSelected}
                  className="group relative px-6 py-3 bg-gradient-to-r from-blue-500/70 to-blue-600/70 hover:from-blue-600/70 hover:to-blue-700/70 rounded-xl transition-all duration-300 ease-in-out transform hover:scale-105"
                >
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-xl blur opacity-50 group-hover:opacity-70 transition duration-300"></div>
                  <div className="relative flex items-center">
                    <svg className="w-5 h-5 mr-2 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2zm9-13.5V9"></path>
                    </svg>
                    Report Selected
                  </div>
                </button>
              </div>

              <div className="grid grid-cols-1 gap-8 mt-8 max-w-[95%] mx-auto">
                {results.map((result) => (
                  <div key={result.id} className="bg-black/40 rounded-xl p-8 border border-blue-500/20">
                    {/* Main image */}
                    {result.imageUrl && (
                      <div className="mb-8">
                        <img
                          src={result.imageUrl}
                          alt="Detection result"
                          className="w-full h-[600px] object-cover rounded-lg"
                        />
                      </div>
                    )}
                    {/* Additional images */}
                    {result.images && result.images.length > 0 && (
                      <div className="mb-8">
                        <h3 className="text-blue-400 font-mono mb-4 text-2xl">Related Images</h3>
                        <div className="grid grid-cols-2 gap-8">
                          {result.images.map((img: string, idx: number) => (
                            <img
                              key={idx}
                              src={img}
                              alt={`Related image ${idx+1}`}
                              className="w-full h-[500px] object-cover rounded-lg"
                            />
                          ))}
                        </div>
                      </div>
                    )}
                    {/* YouTube videos */}
                    {result.youtube_videos && result.youtube_videos.length > 0 && (
                      <div className="mb-8">
                        <h3 className="text-blue-400 font-mono mb-4 text-2xl">YouTube Videos</h3>
                        <div className="grid grid-cols-1 gap-8">
                          {result.youtube_videos.map((url: string, idx: number) => {
                            const videoId = url.split('v=')[1]?.split('&')[0];
                            const embedUrl = videoId ? `https://www.youtube.com/embed/${videoId}?rel=0` : url;
                            return (
                              <div key={idx} className="relative pb-[56.25%] h-0">
                                <iframe
                                  src={embedUrl}
                                  title={`YouTube video ${idx+1}`}
                                  className="absolute top-0 left-0 w-full h-full rounded-lg"
                                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                  allowFullScreen
                                />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    {/* Vimeo videos */}
                    {result.vimeo_videos && result.vimeo_videos.length > 0 && (
                      <div className="mb-8">
                        <h3 className="text-blue-400 font-mono mb-4 text-2xl">Vimeo Videos</h3>
                        <div className="grid grid-cols-1 gap-8">
                          {result.vimeo_videos.map((url: string, idx: number) => {
                            const videoId = url.split('vimeo.com/')[1]?.split('?')[0];
                            const embedUrl = videoId ? `https://player.vimeo.com/video/${videoId}?title=0&byline=0&portrait=0` : url;
                            return (
                              <div key={idx} className="relative pb-[56.25%] h-0">
                                <iframe
                                  src={embedUrl}
                                  title={`Vimeo video ${idx+1}`}
                                  className="absolute top-0 left-0 w-full h-full rounded-lg"
                                  allow="autoplay; fullscreen; picture-in-picture"
                                  allowFullScreen
                                />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    {/* Dailymotion videos */}
                    {result.dailymotion_videos && result.dailymotion_videos.length > 0 && (
                      <div className="mb-8">
                        <h3 className="text-blue-400 font-mono mb-4 text-2xl">Dailymotion Videos</h3>
                        <div className="grid grid-cols-1 gap-8">
                          {result.dailymotion_videos.map((url: string, idx: number) => {
                            const videoId = url.split('dailymotion.com/video/')[1]?.split('?')[0];
                            const embedUrl = videoId ? `https://www.dailymotion.com/embed/video/${videoId}?autoplay=0&controls=1` : url;
                            return (
                              <div key={idx} className="relative pb-[56.25%] h-0">
                                <iframe
                                  src={embedUrl}
                                  title={`Dailymotion video ${idx+1}`}
                                  className="absolute top-0 left-0 w-full h-full rounded-lg"
                                  allow="autoplay"
                                  allowFullScreen
                                />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    <div className="flex justify-between items-center mt-6">
                      <div className="text-blue-300 font-mono text-lg">
                        Confidence: {result.confidence?.toFixed(2)}%
                      </div>
                      <div className="text-gray-400 text-sm">
                        {new Date(result.timestamp).toLocaleString()}
                      </div>
                    </div>
                    {result.celebrity && (
                      <div className="text-blue-400 font-bold mt-3 text-xl">{result.celebrity}</div>
                    )}
                    
                    {/* Add this button near the bottom of the result card */}
                    <div className="mt-6 flex justify-end">
                      <button
                        onClick={() => handleReportContent(result)}
                        className="group relative px-6 py-3 bg-gradient-to-r from-red-500/70 to-red-600/70 hover:from-red-600/70 hover:to-red-700/70 rounded-xl transition-all duration-300 ease-in-out transform hover:scale-105"
                      >
                        <div className="absolute -inset-0.5 bg-gradient-to-r from-red-500/20 to-red-600/20 rounded-xl blur opacity-50 group-hover:opacity-70 transition duration-300"></div>
                        <div className="relative flex items-center space-x-2">
                          <AlertTriangle className="w-4 h-4 text-white/90" />
                          <span className="text-white/90">Report Content</span>
                        </div>
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {hasMore && (
                <button
                  onClick={handleLoadMore}
                  className="group relative w-full py-4 bg-gradient-to-r from-blue-500/70 to-blue-600/70 hover:from-blue-600/70 hover:to-blue-700/70 rounded-xl transition-all duration-300 ease-in-out transform hover:scale-[1.02] mt-8"
                >
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-xl blur opacity-50 group-hover:opacity-70 transition duration-300"></div>
                  <div className="relative flex items-center justify-center">
                    <svg className="w-5 h-5 mr-2 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    Load More
                  </div>
                </button>
              )}
            </div>
          )}
        </div>

        {/* Image Gallery Section */}
        <div className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-500/80 to-blue-400/80 mb-4 font-mono">
              Recent Detections
            </h2>
            <p className="text-blue-300/80 text-lg max-w-2xl mx-auto font-mono">
              Explore our latest deepfake detections and analysis
            </p>
            <DynamicDetectionGallery />
          </div>
          <div className="relative bg-black/40 backdrop-blur-sm rounded-2xl p-6 border border-blue-500/20">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
            <div className="relative">
              {/* Optionally, you can keep the gallery for recent detections or static data here */}
            </div>
          </div>
        </div>

        {/* About Us Section */}
        <div className="mt-32 mb-16">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-500/80 to-blue-400/80 mb-4 font-mono">
              About DeepGuard
            </h2>
            <p className="text-blue-300/80 text-lg max-w-3xl mx-auto font-mono">
              Empowering users with advanced AI technology to combat deepfake threats
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Mission */}
            <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-8 transform transition-all duration-300 hover:scale-105 border border-blue-500/20">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
              <div className="relative">
                <div className="w-12 h-12 mb-6 rounded-xl bg-gradient-to-r from-blue-500/80 to-blue-600/80 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-blue-400 mb-4 font-mono">Our Mission</h3>
                <p className="text-blue-300/80 font-mono">
                  To provide cutting-edge deepfake detection technology that helps maintain trust and authenticity in digital media.
                </p>
              </div>
            </div>

            {/* Technology */}
            <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-8 transform transition-all duration-300 hover:scale-105 border border-blue-500/20">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
              <div className="relative">
                <div className="w-12 h-12 mb-6 rounded-xl bg-gradient-to-r from-blue-500/80 to-blue-600/80 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-blue-400 mb-4 font-mono">Advanced Technology</h3>
                <p className="text-blue-300/80 font-mono">
                  Leveraging state-of-the-art AI algorithms and machine learning to detect sophisticated deepfake content with high accuracy.
                </p>
              </div>
            </div>

            {/* Impact */}
            <div className="group relative bg-black/40 backdrop-blur-sm rounded-2xl p-8 transform transition-all duration-300 hover:scale-105 border border-blue-500/20">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-2xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
              <div className="relative">
                <div className="w-12 h-12 mb-6 rounded-xl bg-gradient-to-r from-blue-500/80 to-blue-600/80 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-blue-400 mb-4 font-mono">Our Impact</h3>
                <p className="text-blue-300/80 font-mono">
                  Helping individuals and organizations maintain digital trust and combat misinformation in an increasingly complex digital landscape.
                </p>
              </div>
            </div>
          </div>

          {/* Team Section */}
          <div className="mt-16 text-center">
            <button className="group relative px-8 py-4 bg-gradient-to-r from-blue-500/70 to-blue-600/70 hover:from-blue-600/70 hover:to-blue-700/70 rounded-xl transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg shadow-blue-500/5">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-xl opacity-0 group-hover:opacity-30 blur-[1px] transition duration-500"></div>
              <div className="relative flex items-center justify-center space-x-2">
                <svg className="w-5 h-5 text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
                </svg>
                <span className="text-white/90 font-mono">Meet Our Team</span>
              </div>
            </button>
          </div>
        </div>
      </div>
      
      {selectedResult && (
        <ReportDialog
          isOpen={isReportDialogOpen}
          onClose={() => {
            setIsReportDialogOpen(false);
            setSelectedResult(null);
          }}
          imageUrls={[
            selectedResult.imageUrl,
            ...(selectedResult.images || [])
          ]}
          videoUrls={{
            youtube: selectedResult.youtube_videos || [],
            vimeo: selectedResult.vimeo_videos || [],
            dailymotion: selectedResult.dailymotion_videos || []
          }}
          additionalInfo={{
            confidence: selectedResult.confidence,
            timestamp: selectedResult.timestamp,
            celebrity: selectedResult.celebrity
          }}
        />
      )}
    </div>
  );
}

export default App;


