# Medical Chatbot UI - React + Tailwind

Modern web interface for the medical chatbot built with React and Tailwind CSS.

## ✨ Features

- **Clean, responsive design** with Tailwind CSS
- **Real-time chat interface** with message history
- **Model comparison** - Switch between base and fine-tuned models
- **Typing indicators** for better UX
- **Example questions** for quick testing
- **Dark mode support** (easily configurable)
- **Mobile-friendly** responsive layout

## 📦 Installation

### 1. Install Node.js
Download from https://nodejs.org/ (LTS version recommended)

### 2. Install Dependencies
```bash
cd chatbot-ui
npm install
```

## 🚀 Running the App

### Terminal 1: Start the Gradio Backend
```bash
# From your main chatbot directory
cd /home/belysetag/Desktop/chatbot
jupyter notebook medical_chatbot_final_(2).ipynb
# Run the Gradio cell (last cell) - it will start on http://localhost:7860
```

### Terminal 2: Start React Dev Server
```bash
cd /home/belysetag/Desktop/chatbot/chatbot-ui
npm run dev
```

Open http://localhost:3000 in your browser

## 📁 Project Structure

```
chatbot-ui/
├── src/
│   ├── components/
│   │   ├── Header.jsx           # Top navigation and model selector
│   │   ├── ChatWindow.jsx       # Message display area
│   │   ├── Message.jsx          # Individual message component
│   │   ├── TypingIndicator.jsx  # Loading animation
│   │   └── InputForm.jsx        # Text input and buttons
│   ├── App.jsx                  # Main app component
│   ├── main.jsx                 # React entry point
│   └── index.css                # Tailwind + custom styles
├── index.html                   # HTML template
├── package.json                 # Dependencies
├── vite.config.js              # Vite build config
├── tailwind.config.js          # Tailwind customization
└── postcss.config.js           # PostCSS setup
```

## 🛠️ Customization

### Change Colors
Edit `tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      medical: {
        50: '#your-color',
        600: '#your-color',
        700: '#your-color',
      }
    }
  }
}
```

### Change Gradio Backend URL
Edit `src/App.jsx` fetch URL:
```javascript
const response = await fetch('http://YOUR_SERVER:PORT/api/predict/', {
```

### Add New Components
```bash
# Create new component
touch src/components/YourComponent.jsx
```

## 🌐 Deployment

### Build for Production
```bash
npm run build
```

This creates `dist/` folder ready to deploy.

### Deploy to GitHub Pages
```bash
npm install gh-pages --save-dev
```

Edit `package.json`:
```json
"homepage": "https://YOUR_USERNAME.github.io/medical-chatbot-ui",
"scripts": {
  "predeploy": "npm run build",
  "deploy": "gh-pages -d dist"
}
```

Deploy:
```bash
npm run deploy
```

### Deploy to Vercel (Recommended)
1. Push to GitHub
2. Go to https://vercel.com
3. Import your repository
4. Deploy (one click)
5. Add environment variable for Gradio URL

## 🔧 Troubleshooting

### "Cannot connect to Gradio server"
- ✅ Make sure Jupyter notebook cell is running (terminal shows Gradio URL)
- ✅ Check URL is `http://localhost:7860`
- ✅ Try disabling firewall temporarily

### "Messages not sending"
- ✅ Check browser console (F12) for errors
- ✅ Verify Gradio server is responding
- ✅ Try refreshing the page

### "Styling looks broken"
- ✅ Run `npm install` again
- ✅ Clear browser cache (Ctrl+Shift+R)
- ✅ Check `tailwind.config.js` is in root

## 📚 Next Steps

1. **Customize colors** in `tailwind.config.js`
2. **Add your logo** to the Header component
3. **Deploy to Vercel** for public access
4. **Add analytics** (Google Analytics, Mixpanel)
5. **Dark mode** toggle in settings
6. **Chat history** persistence with localStorage

## 📝 Notes

- Vite provides fast HMR (Hot Module Replacement)
- Tailwind classes compile only for used styles (small bundle size)
- React 18.2 with latest features
- Easy to add TypeScript if needed

## 🤝 Contributing

Feel free to modify and improve!

## 📄 License

MIT - Use freely in your projects
