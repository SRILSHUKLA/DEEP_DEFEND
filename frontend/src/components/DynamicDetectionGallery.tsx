import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Shield, AlertTriangle, ChevronLeft, ChevronRight, Info } from 'lucide-react';

// Define types for the detection data
type ThreatLevel = 'Low' | 'Medium' | 'High' | 'Critical';

interface DetectionData {
  id: number;
  image: string;
  title: string;
  detectionScore: number;
  platform: string;
  detectedOn: string;
  category: string;
  description: string;
  threatLevel: ThreatLevel;
}

export default function DynamicDetectionGallery() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isHovering, setIsHovering] = useState(false);
  const [autoplayEnabled, setAutoplayEnabled] = useState(true);
  const [expandedInfo, setExpandedInfo] = useState(false);

  // Sample detection data - in a real app, this would come from a backend API
  const detectionData: DetectionData[] = [
    {
      id: 1,
      image: "https://warroom.armywarcollege.edu/wp-content/uploads/21-057-Deep_fake_Tom_Cruise.jpeg",
      title: "Celebrity Impersonation",
      detectionScore: 99.7,
      platform: "TikTok",
      detectedOn: "May 14, 2025",
      category: "Synthetic Media",
      description: "AI-generated content impersonating a celebrity, detected using advanced facial inconsistency analysis and voice pattern recognition.",
      threatLevel: "Medium"
    },
    {
      id: 2,
      image: "https://dam.mediacorp.sg/image/upload/s--Ncxdh58M--/c_fill,g_auto,h_468,w_830/f_auto,q_auto/v1/mediacorp/cna/image/2024/09/12/Deepfake%20Why%20It%20Matters%20screengrab.jpg?itok=gOmtbEp3",
      title: "Synthetic Financial Advisor",
      detectionScore: 97.3,
      platform: "YouTube",
      detectedOn: "May 11, 2025",
      category: "Disinformation",
      description: "Fake financial advisor created using generative AI to promote fraudulent investment schemes targeting retirement accounts.",
      threatLevel: "High"
    },
    {
      id: 3,
      image: "https://static01.nyt.com/images/2023/02/01/video/01vid-deepfake-disinfo-man-split-COVER/01vid-deepfake-disinfo-man-split-COVER-superJumbo-v2.png",
      title: "Political Deepfake",
      detectionScore: 98.4,
      platform: "Facebook",
      detectedOn: "May 15, 2025",
      category: "Misinformation",
      description: "Manipulated video content of a political figure making fabricated statements about sensitive policy issues.",
      threatLevel: "Critical"
    },
    {
      id: 4,
      image: "https://miro.medium.com/v2/resize:fit:1400/0*VYFviTLDQKHvAJez",
      title: "Corporate Identity Fraud",
      detectionScore: 95.8,
      platform: "LinkedIn",
      detectedOn: "May 9, 2025",
      category: "Identity Theft",
      description: "Synthetic corporate executive profile using AI-generated face and fabricated credentials to facilitate business email compromise.",
      threatLevel: "High"
    },
    {
      id: 5,
      image: "https://www.snopes.com/tachyon/2021/12/oprah-marjorie-taylor-greene-deepfake-comparison-2.jpeg",
      title: "Manipulated Interview",
      detectionScore: 96.2,
      platform: "Instagram",
      detectedOn: "May 13, 2025",
      category: "Media Manipulation",
      description: "Altered interview footage using neural voice synthesis and facial reenactment to fabricate controversial statements.",
      threatLevel: "Medium"
    }
  ];

  // Auto-advance carousel
  useEffect(() => {
    if (!autoplayEnabled) return;
    
    const interval = setInterval(() => {
      if (!isHovering) {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % detectionData.length);
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [autoplayEnabled, isHovering, detectionData.length]);

  // Navigate to previous image
  const goToPrevious = () => {
    setCurrentIndex((prevIndex) => (prevIndex === 0 ? detectionData.length - 1 : prevIndex - 1));
  };

  // Navigate to next image
  const goToNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % detectionData.length);
  };

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        goToPrevious();
      } else if (e.key === 'ArrowRight') {
        goToNext();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Current detection
  const currentDetection = detectionData[currentIndex];

  // Threat level color mapping
  const getThreatLevelColor = (level: ThreatLevel): string => {
    switch (level) {
      case 'Low': return 'bg-blue-500';
      case 'Medium': return 'bg-yellow-500';
      case 'High': return 'bg-orange-500';
      case 'Critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div 
      className="relative overflow-hidden rounded-xl border border-blue-500/30 bg-blue-950/20 backdrop-blur-md shadow-2xl"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      {/* Animated background grid */}
      <div className="absolute inset-0 z-0">
        <div 
          className="absolute inset-0 z-0" 
          style={{
            backgroundImage: `linear-gradient(rgba(59, 130, 246, 0.05) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(59, 130, 246, 0.05) 1px, transparent 1px)`,
            backgroundSize: '20px 20px',
          }}
        />
        <div 
          className="absolute inset-0 z-0" 
          style={{
            backgroundImage: `linear-gradient(rgba(59, 130, 246, 0.03) 2px, transparent 2px),
                            linear-gradient(90deg, rgba(59, 130, 246, 0.03) 2px, transparent 2px)`,
            backgroundSize: '100px 100px',
          }}
        />
      </div>

      {/* Main carousel section */}
      <div className="relative z-10 h-[600px] flex flex-col lg:flex-row">
        {/* Image container */}
        <div className="relative h-full lg:w-2/3 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentDetection.id}
              className="absolute inset-0"
              initial={{ opacity: 0, scale: 1.05 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="relative h-full w-full">
                {/* Image */}
                <img 
                  src={currentDetection.image} 
                  alt={currentDetection.title}
                  className="h-full w-full object-cover"
                />
                
                {/* Scanning effect */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/20 to-blue-500/0 w-full h-full pointer-events-none"
                  animate={{ 
                    x: ['-100%', '100%'],
                  }}
                  transition={{ 
                    repeat: Infinity, 
                    duration: 3, 
                    ease: "linear",
                  }}
                />
                
                {/* Analysis overlay - active on hover */}
                <motion.div 
                  className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300"
                  whileHover={{ opacity: 1 }}
                >
                  <div className="absolute inset-x-0 bottom-0 p-6">
                    <div className="mb-4 flex justify-between items-center">
                      <div className="flex items-center">
                        <div className={`w-3 h-3 rounded-full ${getThreatLevelColor(currentDetection.threatLevel)} animate-pulse mr-2`}></div>
                        <span className="text-sm font-mono text-white/90">THREAT LEVEL: {currentDetection.threatLevel}</span>
                      </div>
                      <div className="flex items-center gap-1 bg-blue-500/20 rounded-full px-3 py-1">
                        <Search className="w-3 h-3 text-blue-300" />
                        <span className="text-xs font-mono text-blue-300">DeepScan™ Analysis</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
                
                {/* Detection score badge */}
                <div className="absolute top-6 left-6 flex items-center gap-2 bg-black/50 backdrop-blur-sm border border-blue-500/30 rounded-lg px-3 py-2">
                  <Shield className="w-4 h-4 text-blue-400" />
                  <div>
                    <div className="text-xs text-blue-300 font-mono">DETECTION CONFIDENCE</div>
                    <div className="text-lg font-bold text-blue-100">{currentDetection.detectionScore}%</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Previous/Next buttons */}
          <div className="absolute inset-y-0 left-0 flex items-center">
            <motion.button
              className="flex justify-center items-center w-10 h-16 bg-black/30 backdrop-blur-sm hover:bg-blue-900/40 rounded-r-lg text-white/70 hover:text-white"
              onClick={goToPrevious}
              whileHover={{ x: 3 }}
              whileTap={{ scale: 0.95 }}
            >
              <ChevronLeft className="w-6 h-6" />
            </motion.button>
          </div>
          
          <div className="absolute inset-y-0 right-0 flex items-center">
            <motion.button
              className="flex justify-center items-center w-10 h-16 bg-black/30 backdrop-blur-sm hover:bg-blue-900/40 rounded-l-lg text-white/70 hover:text-white"
              onClick={goToNext}
              whileHover={{ x: -3 }}
              whileTap={{ scale: 0.95 }}
            >
              <ChevronRight className="w-6 h-6" />
            </motion.button>
          </div>
        </div>
        
        {/* Detection info panel */}
        <div className="relative lg:w-1/3 bg-black/50 backdrop-blur-sm p-6 flex flex-col border-l border-blue-500/20">
          {/* Detection title with animated accent */}
          <div className="relative mb-6 border-b border-blue-500/20 pb-4">
            <motion.div 
              className="absolute left-0 top-0 h-6 w-1.5 bg-blue-500"
              animate={{ height: ["24px", "36px", "24px"] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <h3 className="text-2xl font-bold text-white ml-3">{currentDetection.title}</h3>
            <div className="ml-3 text-blue-300 text-sm mt-1 font-mono">Detection #{currentDetection.id}</div>
          </div>
          
          {/* Detection metadata */}
          <div className="space-y-4 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-blue-300 font-mono">PLATFORM</span>
              <span className="font-medium text-white bg-blue-900/30 px-2 py-0.5 rounded">
                {currentDetection.platform}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-blue-300 font-mono">DETECTED ON</span>
              <span className="font-medium text-white">{currentDetection.detectedOn}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-blue-300 font-mono">CATEGORY</span>
              <span className="font-medium text-white bg-blue-900/30 px-2 py-0.5 rounded">
                {currentDetection.category}
              </span>
            </div>
            
            <div>
              <div className="text-blue-300 font-mono mb-2">DESCRIPTION</div>
              <p className="text-white/80 leading-relaxed">{currentDetection.description}</p>
            </div>
          </div>
          
          {/* Expanded technical analysis */}
          <div className="mt-auto pt-6">
            <div 
              className="flex items-center cursor-pointer" 
              onClick={() => setExpandedInfo(!expandedInfo)}
            >
              <Info className="w-4 h-4 text-blue-400 mr-2" />
              <span className="text-blue-400 font-medium">
                {expandedInfo ? "Hide Technical Analysis" : "Show Technical Analysis"}
              </span>
            </div>
            
            <AnimatePresence>
              {expandedInfo && (
                <motion.div 
                  className="mt-4 bg-blue-900/20 border border-blue-500/20 rounded-lg p-4 text-sm"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  {/* Detection metrics */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <div className="text-blue-300 text-xs mb-1 font-mono">VISUAL INCONSISTENCY</div>
                      <div className="h-1.5 bg-blue-900/40 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-gradient-to-r from-blue-500 to-blue-400" 
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.random() * 30 + 70}%` }}
                          transition={{ duration: 1 }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-blue-300 text-xs mb-1 font-mono">AUDIO ARTIFACTS</div>
                      <div className="h-1.5 bg-blue-900/40 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-gradient-to-r from-blue-500 to-blue-400" 
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.random() * 30 + 70}%` }}
                          transition={{ duration: 1 }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-blue-300 text-xs mb-1 font-mono">GAN PATTERNS</div>
                      <div className="h-1.5 bg-blue-900/40 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-gradient-to-r from-blue-500 to-blue-400" 
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.random() * 30 + 70}%` }}
                          transition={{ duration: 1 }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-blue-300 text-xs mb-1 font-mono">BEHAVIORAL ANOMALIES</div>
                      <div className="h-1.5 bg-blue-900/40 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-gradient-to-r from-blue-500 to-blue-400" 
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.random() * 30 + 70}%` }}
                          transition={{ duration: 1 }}
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center">
                      <AlertTriangle className="w-3 h-3 text-yellow-400 mr-1" />
                      <span className="text-yellow-300">Generated with AI Model ID: SD-4.8x</span>
                    </div>
                    <span className="text-blue-300 font-mono">TS-{Math.floor(Math.random() * 10000)}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          {/* Bottom controls */}
          <div className="pt-4 mt-4 border-t border-blue-500/20 flex justify-between">
            {/* Pagination indicators */}
            <div className="flex items-center gap-1">
              {detectionData.map((_, index) => (
                <button 
                  key={index} 
                  onClick={() => setCurrentIndex(index)}
                  className={`w-2 h-2 rounded-full transition-all duration-300 ${
                    index === currentIndex ? 'bg-blue-400 w-6' : 'bg-blue-700'
                  }`}
                  aria-label={`Go to slide ${index + 1}`}
                />
              ))}
            </div>
            
            {/* Autoplay toggle */}
            <button
              onClick={() => setAutoplayEnabled(!autoplayEnabled)}
              className={`text-xs font-medium px-3 py-1 rounded-full transition-colors ${
                autoplayEnabled 
                  ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40' 
                  : 'bg-blue-900/20 text-blue-400/60 border border-blue-500/20'
              }`}
            >
              {autoplayEnabled ? 'Auto-advance ON' : 'Auto-advance OFF'}
            </button>
          </div>
        </div>
      </div>
      
      {/* Bottom animation line */}
      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-black">
        <motion.div
          className="h-full bg-gradient-to-r from-blue-400 via-blue-500 to-blue-400"
          animate={{
            x: ['-100%', '100%'],
          }}
          transition={{
            repeat: Infinity,
            duration: 3,
            ease: 'linear',
          }}
        />
      </div>
    </div>
  );
} 