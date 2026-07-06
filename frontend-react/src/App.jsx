import { useEffect } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import useStore from './store/useStore'
import { authAPI } from './lib/api'
import Layout from './components/layout/Layout'
import Landing  from './pages/Landing'
import Login    from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import Chatbot   from './pages/Chatbot'
import Predictions from './pages/Predictions'
import Vision    from './pages/Vision'
import Progress  from './pages/Progress'
import Pricing   from './pages/Pricing'
import Profile   from './pages/Profile'

/**
 * AuthBootstrap — runs once on app mount.
 *
 * Asks the backend "is this cookie still valid?" via /auth/me, and updates
 * the store. Until that probe finishes, `authReady` is false, which lets
 * PrivateRoute show a tiny spinner instead of bouncing the user to /login
 * just because the store hasn't hydrated yet.
 */
function AuthBootstrap({ children }) {
  const setUser   = useStore((s) => s.setUser)
  const clearUser = useStore((s) => s.clearUser)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const { data } = await authAPI.me()
        if (cancelled) return
        const userObj = {
          user_id: data.user_id,
          name:    data.name,
          email:   data.email,
          gender:  data.gender,
          role:    data.role,
        }
        setUser(userObj, data.profile || {})
      } catch {
        if (!cancelled) clearUser()
      }
    })()
    return () => { cancelled = true }
  }, [setUser, clearUser])

  return children
}

/**
 * Theme effect — keeps the <html> data-theme attribute in sync with the
 * store. CSS variables in index.css read off [data-theme="feminine"] etc.
 */
function ThemeEffect() {
  const theme = useStore((s) => s.theme)
  useEffect(() => {
    const root = document.documentElement
    root.setAttribute('data-theme', theme || 'masculine')
  }, [theme])
  return null
}

function PrivateRoute({ children }) {
  const { t }     = useTranslation()
  const user      = useStore((s) => s.user)
  const authReady = useStore((s) => s.authReady)
  const location  = useLocation()

  // Show a quiet placeholder while the cookie probe is in flight.
  if (!authReady) {
    return (
      <div className="min-h-screen flex items-center justify-center"
           style={{ background: 'var(--bg-primary)' }}>
        <div className="flex items-center gap-3 text-white/60">
          <div className="w-4 h-4 rounded-full border-2 border-white/20 border-t-white animate-spin" />
          <span className="text-sm">{t('common:loading')}</span>
        </div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }
  return children
}

export default function App() {
  return (
    <AuthBootstrap>
      <ThemeEffect />
      <Routes>
        <Route path="/"         element={<Landing />} />
        <Route path="/login"    element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/pricing"  element={<Pricing />} />
        <Route element={<PrivateRoute><Layout /></PrivateRoute>}>
          <Route path="/dashboard"   element={<Dashboard />} />
          <Route path="/chat"        element={<Chatbot />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/vision"      element={<Vision />} />
          <Route path="/progress"    element={<Progress />} />
          <Route path="/profile"     element={<Profile />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthBootstrap>
  )
}
