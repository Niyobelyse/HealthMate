import React, { useState } from 'react'

function InputForm({ onSend, disabled }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim()) {
      onSend(input)
      setInput('')
    }
  }

  const exampleQuestions = [
    "What are the symptoms of diabetes?",
    "How do I manage hypertension?",
    "Tell me about COVID-19 treatments"
  ]

  return (
    <div className="border-t border-gray-200 bg-white">
      <div className="max-w-4xl mx-auto px-6 py-6">
        {/* Example questions */}
        {input === '' && (
          <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-2">
            {exampleQuestions.map((question, idx) => (
              <button
                key={idx}
                onClick={() => setInput(question)}
                disabled={disabled}
                className="p-3 text-sm text-left rounded-lg border border-gray-200 hover:bg-gray-50 hover:border-medical-500 transition-colors disabled:opacity-50"
              >
                 {question}
              </button>
            ))}
          </div>
        )}

        {/* Input form */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={disabled}
            placeholder="Ask me anything about health and medicine..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-medical-500 focus:border-transparent disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={disabled || !input.trim()}
            className="px-6 py-3 bg-medical-600 text-white rounded-lg hover:bg-medical-700 transition-colors disabled:bg-gray-300 font-medium flex items-center gap-2"
          >
            {disabled ? (
              <>
                <span className="inline-block animate-spin">⏳</span>
                Sending...
              </>
            ) : (
              <>
                <span>Send</span>
                <span>→</span>
              </>
            )}
          </button>
        </form>


      </div>
    </div>
  )
}

export default InputForm
