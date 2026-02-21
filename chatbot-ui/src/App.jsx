import React, { useState } from 'react'
import ChatWindow from './components/ChatWindow'
import InputForm from './components/InputForm'
import Header from './components/Header'

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

    try {
      // Call FastAPI backend
      const response = await fetch('http://localhost:8001/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          model_type: modelType
        })
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
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        id: messages.length + 2,
        text: "I'm having trouble connecting. Make sure the FastAPI server is running on localhost:8001. Error: " + error.message,
        sender: 'bot',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
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
