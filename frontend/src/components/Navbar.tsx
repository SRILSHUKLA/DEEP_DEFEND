import React from 'react';

const Navbar: React.FC = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="relative group">
                <div className="absolute -inset-1 rounded-xl bg-gradient-to-r from-indigo-700/30 to-blue-600/30 opacity-0 group-hover:opacity-50 blur-[2px] transition duration-500"></div>
                <div className="relative bg-black/50 backdrop-blur-sm px-6 py-3 rounded-xl">
                  <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-500 to-blue-400">
                    DeepDefend
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-center space-x-8">
              <a href="#" className="relative group px-4 py-2 text-gray-300 hover:text-white transition-colors duration-200">
                <span className="relative z-10">Home</span>
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-700/0 to-blue-600/0 rounded-lg opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              </a>
              <a href="#" className="relative group px-4 py-2 text-gray-300 hover:text-white transition-colors duration-200">
                <span className="relative z-10">History</span>
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-700/0 to-blue-600/0 rounded-lg opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              </a>
              <a href="#" className="relative group px-4 py-2 text-gray-300 hover:text-white transition-colors duration-200">
                <span className="relative z-10">Settings</span>
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-700/0 to-blue-600/0 rounded-lg opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              </a>
            </div>
          </div>

          {/* User Profile */}
          <div className="flex items-center">
            <button className="group relative px-4 py-2 rounded-xl bg-black/50 backdrop-blur-sm hover:bg-black/70 transition-all duration-300">
              <div className="absolute -inset-0.5 rounded-xl bg-gradient-to-r from-indigo-700/20 to-blue-600/20 opacity-0 group-hover:opacity-50 blur-[2px] transition duration-500"></div>
              <div className="relative flex items-center space-x-3">
                <div className="w-9 h-9 rounded-lg bg-gradient-to-r from-indigo-700/90 to-blue-600/90 flex items-center justify-center shadow-lg shadow-indigo-500/5">
                  <span className="text-white text-sm font-medium">JD</span>
                </div>
                <span className="text-gray-300 text-sm font-medium">John Doe</span>
                <svg className="w-4 h-4 text-gray-400 group-hover:text-gray-300 transition-colors duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                </svg>
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Glass effect overlay */}
      <div className="absolute inset-0 bg-black/80 backdrop-blur-md -z-10" />
    </nav>
  );
};

export default Navbar; 