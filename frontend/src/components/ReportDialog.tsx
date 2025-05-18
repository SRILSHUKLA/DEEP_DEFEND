import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, X, Send, Copy, Check } from 'lucide-react';

interface ReportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  imageUrls: string[];
  videoUrls: {
    youtube: string[];
    vimeo: string[];
    dailymotion: string[];
  };
  additionalInfo?: {
    confidence?: number;
    timestamp?: string;
    celebrity?: string;
  };
}

const ReportDialog: React.FC<ReportDialogProps> = ({
  isOpen,
  onClose,
  imageUrls,
  videoUrls,
  additionalInfo
}) => {
  const [reportStatus, setReportStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [reportDetails, setReportDetails] = useState('');
  const [selectedAuthority, setSelectedAuthority] = useState('');
  const [copiedStates, setCopiedStates] = useState<{ [key: string]: boolean }>({});

  const authorities = [
    { id: 'fbi', name: 'FBI Cyber Division' },
    { id: 'ic3', name: 'Internet Crime Complaint Center (IC3)' },
    { id: 'local', name: 'Local Law Enforcement' },
    { id: 'platform', name: 'Platform Authority' },
  ];

  const handleCopyToClipboard = async (text: string, key: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedStates({ ...copiedStates, [key]: true });
      setTimeout(() => {
        setCopiedStates({ ...copiedStates, [key]: false });
      }, 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setReportStatus('submitting');

    // Prepare the report data
    const reportData = {
      imageUrls,
      videoUrls,
      additionalInfo,
      reportDetails,
      selectedAuthority,
      timestamp: new Date().toISOString(),
    };

    try {
      // Here you would typically make an API call to your backend
      // For now, we'll simulate an API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      console.log('Report submitted:', reportData);
      setReportStatus('success');
      
      // Reset form after success
      setTimeout(() => {
        onClose();
        setReportStatus('idle');
        setReportDetails('');
        setSelectedAuthority('');
      }, 2000);
    } catch (error) {
      console.error('Error submitting report:', error);
      setReportStatus('error');
    }
  };

  // Function to format links for copying
  const formatLinksForCopy = () => {
    let text = '=== DEEPFAKE CONTENT REPORT ===\n\n';
    
    if (imageUrls.length > 0) {
      text += '--- IMAGE LINKS ---\n';
      imageUrls.forEach((url, i) => text += `${i + 1}. ${url}\n`);
      text += '\n';
    }
    
    if (Object.values(videoUrls).some(arr => arr.length > 0)) {
      text += '--- VIDEO LINKS ---\n';
      if (videoUrls.youtube.length > 0) {
        text += '\nYouTube Videos:\n';
        videoUrls.youtube.forEach((url, i) => text += `${i + 1}. ${url}\n`);
      }
      if (videoUrls.vimeo.length > 0) {
        text += '\nVimeo Videos:\n';
        videoUrls.vimeo.forEach((url, i) => text += `${i + 1}. ${url}\n`);
      }
      if (videoUrls.dailymotion.length > 0) {
        text += '\nDailymotion Videos:\n';
        videoUrls.dailymotion.forEach((url, i) => text += `${i + 1}. ${url}\n`);
      }
    }

    if (additionalInfo) {
      text += '\n--- ADDITIONAL INFORMATION ---\n';
      if (additionalInfo.confidence) text += `Confidence: ${additionalInfo.confidence.toFixed(2)}%\n`;
      if (additionalInfo.timestamp) text += `Detected: ${additionalInfo.timestamp}\n`;
      if (additionalInfo.celebrity) text += `Related to: ${additionalInfo.celebrity}\n`;
    }

    return text;
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Dialog */}
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="relative bg-black/90 border border-blue-500/30 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl"
          >
            {/* Header */}
            <div className="p-6 border-b border-blue-500/20">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <AlertTriangle className="w-6 h-6 text-blue-500" />
                  <h2 className="text-xl font-bold text-blue-400">Report Deepfake Content</h2>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-blue-500/10 rounded-lg transition-colors duration-200"
                >
                  <X className="w-5 h-5 text-blue-400" />
                </button>
              </div>
            </div>

            {/* Content */}
            <form onSubmit={handleSubmit} className="p-6 space-y-6">
              {/* Links Section */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-blue-400 font-mono text-sm">CONTENT LINKS</h3>
                  <button
                    type="button"
                    onClick={() => handleCopyToClipboard(formatLinksForCopy(), 'all')}
                    className="flex items-center space-x-2 text-sm text-blue-400 hover:text-blue-300 transition-colors"
                  >
                    {copiedStates['all'] ? (
                      <>
                        <Check className="w-4 h-4" />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4" />
                        <span>Copy All</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Image Links */}
                {imageUrls.length > 0 && (
                  <div className="bg-blue-950/20 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-mono text-blue-300">Image Links</span>
                      <button
                        type="button"
                        onClick={() => handleCopyToClipboard(imageUrls.join('\n'), 'images')}
                        className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                      >
                        {copiedStates['images'] ? 'Copied!' : 'Copy Images'}
                      </button>
                    </div>
                    <div className="space-y-2">
                      {imageUrls.map((url, index) => (
                        <div key={index} className="text-sm text-blue-300/80 break-all bg-blue-900/20 p-2 rounded">
                          {url}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Video Links */}
                {Object.values(videoUrls).some(arr => arr.length > 0) && (
                  <div className="bg-blue-950/20 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-mono text-blue-300">Video Links</span>
                      <button
                        type="button"
                        onClick={() => handleCopyToClipboard(
                          Object.entries(videoUrls)
                            .filter(([_, urls]) => urls.length > 0)
                            .map(([platform, urls]) => `${platform.toUpperCase()}:\n${urls.join('\n')}`)
                            .join('\n\n'),
                          'videos'
                        )}
                        className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                      >
                        {copiedStates['videos'] ? 'Copied!' : 'Copy Videos'}
                      </button>
                    </div>
                    <div className="space-y-4">
                      {Object.entries(videoUrls).map(([platform, urls]) => urls.length > 0 && (
                        <div key={platform}>
                          <div className="text-xs text-blue-400 mb-1">{platform.toUpperCase()}</div>
                          {urls.map((url, index) => (
                            <div key={index} className="text-sm text-blue-300/80 break-all bg-blue-900/20 p-2 rounded mb-2">
                              {url}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Authority Selection */}
              <div className="space-y-2">
                <label className="text-blue-400 font-mono text-sm">Select Authority to Report to:</label>
                <select
                  value={selectedAuthority}
                  onChange={(e) => setSelectedAuthority(e.target.value)}
                  required
                  className="w-full bg-blue-950/20 border border-blue-500/20 rounded-lg px-4 py-3 text-blue-300/80 focus:outline-none focus:border-blue-500/50 transition-colors duration-200"
                >
                  <option value="">Select an authority...</option>
                  {authorities.map(authority => (
                    <option key={authority.id} value={authority.id}>
                      {authority.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Additional Details */}
              <div className="space-y-2">
                <label className="text-blue-400 font-mono text-sm">Additional Details:</label>
                <textarea
                  value={reportDetails}
                  onChange={(e) => setReportDetails(e.target.value)}
                  placeholder="Provide any additional context or details about this content..."
                  rows={4}
                  className="w-full bg-blue-950/20 border border-blue-500/20 rounded-lg px-4 py-3 text-blue-300/80 placeholder-blue-300/40 focus:outline-none focus:border-blue-500/50 transition-colors duration-200"
                />
              </div>

              {/* Submit Button */}
              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={reportStatus === 'submitting'}
                  className={`group relative px-6 py-3 bg-gradient-to-r from-blue-500/70 to-blue-600/70 hover:from-blue-600/70 hover:to-blue-700/70 rounded-xl transition-all duration-300 ease-in-out transform hover:scale-105 ${
                    reportStatus === 'submitting' ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-blue-600/20 rounded-xl blur opacity-50 group-hover:opacity-70 transition duration-300"></div>
                  <div className="relative flex items-center space-x-2">
                    <Send className="w-4 h-4 text-white/90" />
                    <span className="text-white/90">
                      {reportStatus === 'submitting' ? 'Submitting...' : 'Submit Report'}
                    </span>
                  </div>
                </button>
              </div>
            </form>

            {/* Status Messages */}
            <AnimatePresence>
              {reportStatus === 'success' && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/95"
                >
                  <div className="text-center space-y-3">
                    <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center mx-auto">
                      <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <p className="text-green-400 font-mono">Report submitted successfully!</p>
                  </div>
                </motion.div>
              )}

              {reportStatus === 'error' && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute bottom-0 inset-x-0 p-4"
                >
                  <div className="bg-red-500/20 text-red-400 px-4 py-3 rounded-lg text-center">
                    An error occurred while submitting the report. Please try again.
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default ReportDialog; 