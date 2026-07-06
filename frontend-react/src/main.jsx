import React, { Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useTranslation } from 'react-i18next'
import App from './App'
import './i18n'
import './index.css'

/** Toaster that follows the active language: position + RTL rendering. */
function AppToaster() {
  const { i18n } = useTranslation()
  const rtl = i18n.dir() === 'rtl'
  return (
    <Toaster
      position={rtl ? 'top-left' : 'top-right'}
      toastOptions={{
        style: { background:'#111f33', color:'#e0eeff', border:'1px solid rgba(255,255,255,0.08)', direction: rtl ? 'rtl' : 'ltr' },
        success: { iconTheme: { primary:'#00ff88', secondary:'#060d1a' } },
        error:   { iconTheme: { primary:'#ff6b35', secondary:'#060d1a' } },
      }}
    />
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Suspense fallback={null}>
        <App />
        <AppToaster />
      </Suspense>
    </BrowserRouter>
  </React.StrictMode>
)
