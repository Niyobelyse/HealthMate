import React from 'react'

function Header({ modelType, setModelType, onClear }) {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-medical-700">
            HealthMate
          </h1>

        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-medical-500"
            >
              <option value="fine-tuned">Fine-tuned</option>
              <option value="base">Base Model</option>
            </select>
          </div>

          {/* <button
            onClick={onClear}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            Clear Chat
          </button> */}
        </div>
      </div>
    </header>
  )
}

export default Header
