import React from 'react'

function TypingIndicator() {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="bg-white border border-gray-200 text-gray-900 rounded-2xl px-6 py-4 shadow-sm">
        <div className="flex space-x-2">
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse-light"></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse-light" style={{ animationDelay: '0.2s' }}></div>
          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse-light" style={{ animationDelay: '0.4s' }}></div>
        </div>
      </div>
    </div>
  )
}

export default TypingIndicator
