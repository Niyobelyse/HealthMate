import React from 'react'

function Message({ message }) {
  const isBot = message.sender === 'bot'

  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} animate-fade-in`}>
      <div
        className={`max-w-2xl px-6 py-4 rounded-2xl ${
          isBot
            ? 'bg-white border border-gray-200 text-gray-900 shadow-sm'
            : 'bg-medical-600 text-white shadow-md'
        }`}
      >
        <p className="text-base leading-relaxed whitespace-pre-wrap">
          {message.text}
        </p>
        <span className={`text-xs mt-2 block ${
          isBot ? 'text-gray-500' : 'text-blue-100'
        }`}>
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </span>
      </div>
    </div>
  )
}

export default Message
