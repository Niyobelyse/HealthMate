import React, { useState } from 'react'
import ChatWindow from './components/ChatWindow'
import InputForm from './components/InputForm'
import Header from './components/Header'

// Backend API configuration
const API_URL = 'https://belyseniyo-healthmate-backend.hf.space/query'

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your medical assistant. I can help answer questions about diabetes, hypertension, treatments, medications, and more. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ])
  const [loading, setLoading] = useState(false)
  const [modelType, setModelType] = useState('fine-tuned')

  const sendMessage = async (userMessage) => {
    if (!userMessage.trim()) return

    // Add user message to chat
    const newUserMessage = {
      id: messages.length + 1,
      text: userMessage,
      sender: 'user',
      timestamp: new Date()
    }
    setMessages(prev => [...prev, newUserMessage])
    setLoading(true)

    // Retry logic with exponential backoff
    const maxRetries = 3
    let lastError = null

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        // Call FastAPI backend
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: userMessage,
            model_type: modelType
          }),
          timeout: 30000 // 30 second timeout
        })

        const data = await response.json()
        const botReply = data.response || "I couldn't generate a response. Please try again."

        // Add bot message
        const newBotMessage = {
          id: messages.length + 2,
          text: botReply,
          sender: 'bot',
          timestamp: new Date()
        }
        setMessages(prev => [...prev, newBotMessage])
        setLoading(false)
        return // Success - exit retry loop
      } catch (error) {
        lastError = error
        console.error(`Attempt ${attempt + 1} failed:`, error)
        
        // Wait before retrying (exponential backoff: 1s, 2s, 4s)
        if (attempt < maxRetries - 1) {
          const waitTime = Math.pow(2, attempt) * 1000
          await new Promise(resolve => setTimeout(resolve, waitTime))
        }
      }
    }

    // All retries failed
    console.error('All retries exhausted. Last error:', lastError)
    const errorMessage = {
      id: messages.length + 2,
      text: "I'm having trouble connecting to the backend. The server might be starting up. Please try again in a moment.",
      sender: 'bot',
      timestamp: new Date()
    }
    setMessages(prev => [...prev, errorMessage])
    setLoading(false)
  }

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        text: "Hello! I'm your medical assistant. How can I help you today?",
        sender: 'bot',
        timestamp: new Date()
      }
    ])
  }

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-blue-50 to-indigo-50">
      <Header modelType={modelType} setModelType={setModelType} onClear={clearChat} />
      <ChatWindow messages={messages} loading={loading} />
      <InputForm onSend={sendMessage} disabled={loading} />
    </div>
  )
}

export default App
