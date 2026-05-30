import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
      <Toaster
        position="top-right"
        toastOptions={{
          style: { background:'#111f33', color:'#e0eeff', border:'1px solid rgba(255,255,255,0.08)' },
          success: { iconTheme: { primary:'#00ff88', secondary:'#060d1a' } },
          error:   { iconTheme: { primary:'#ff6b35', secondary:'#060d1a' } },
        }}
      />
    </BrowserRouter>
  </React.StrictMode>
)
