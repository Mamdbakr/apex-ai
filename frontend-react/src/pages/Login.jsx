import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Zap, Eye, EyeOff } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { authAPI } from '../lib/api'
import { apiErrorMessage } from '../i18n/format'
import useStore from '../store/useStore'
import toast from 'react-hot-toast'
import LanguageSwitcher from '../components/LanguageSwitcher'

export default function Login() {
  const { t } = useTranslation(['auth', 'common'])
  const [form, setForm]       = useState({ email: '', password: '' })
  const [showPw, setShowPw]   = useState(false)
  const [loading, setLoading] = useState(false)
  const setUser = useStore((s) => s.setUser)
  const navigate = useNavigate()
  const location = useLocation()

  async function handleSubmit(e) {
    e.preventDefault()
    if (loading) return
    setLoading(true)
    try {
      // The cookie is set by the backend's Set-Cookie response header.
      // We just consume the user payload to populate the store.
      const { data } = await authAPI.signin(form)

      const userObj = {
        user_id: data.user_id,
        name:    data.name,
        email:   data.email,
        gender:  data.gender,
        role:    data.role,
      }
      setUser(userObj, data.profile || {})

      const firstName =
        (data.name && String(data.name).split(' ')[0]) ||
        (data.email && String(data.email).split('@')[0]) ||
        t('auth:there')
      toast.success(t('auth:welcomeBackName', { name: firstName }))

      // Navigate where the user was trying to go, or /dashboard.
      const redirectTo = location.state?.from || '/dashboard'
      navigate(redirectTo, { replace: true })
    } catch (err) {
      toast.error(apiErrorMessage(t, err.detail, 'auth:signInFailed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 relative"
         style={{ background: 'var(--bg-primary)' }}>
      {/* Aurora background glows */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="aurora w-[520px] h-[520px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
             style={{ background:'var(--accent)', opacity:0.10 }} />
        <div className="aurora w-[300px] h-[300px] top-[15%] right-[10%]"
             style={{ background:'var(--accent3)', opacity:0.18 }} />
        <div className="aurora w-[260px] h-[260px] bottom-[8%] left-[8%]"
             style={{ background:'var(--accent2)', opacity:0.16 }} />
      </div>

      <div className="absolute top-4 end-4">
        <LanguageSwitcher />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md relative"
      >
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2 font-bold text-2xl font-display mb-6">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg"
                 style={{ background: 'var(--grad-accent)', boxShadow: '0 8px 24px var(--shadow-accent)' }}>
              <Zap size={18} style={{ color: 'var(--bg-primary)' }} />
            </div>
            APEX<span style={{ color: 'var(--accent)' }}>AI</span>
          </Link>
          <h1 className="text-3xl font-bold font-display">{t('auth:welcomeBack')}</h1>
          <p className="text-white/50 text-sm mt-2">{t('auth:signInSubtitle')}</p>
        </div>

        <form onSubmit={handleSubmit} className="card space-y-5">
          <div>
            <label className="text-xs font-semibold text-white/60 uppercase tracking-wider mb-2 block">
              {t('auth:email')}
            </label>
            <input
              className="input"
              type="email"
              placeholder={t('auth:emailPlaceholder')}
              autoComplete="email"
              required
              value={form.email}
              onChange={(e) => setForm((p) => ({ ...p, email: e.target.value }))}
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-white/60 uppercase tracking-wider mb-2 block">
              {t('auth:password')}
            </label>
            <div className="relative">
              <input
                className="input pe-11"
                type={showPw ? 'text' : 'password'}
                placeholder="••••••••"
                autoComplete="current-password"
                required
                value={form.password}
                onChange={(e) => setForm((p) => ({ ...p, password: e.target.value }))}
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute end-3 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/80 transition-colors"
                aria-label={t('auth:togglePassword')}
              >
                {showPw ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          <button type="submit" disabled={loading} className="btn-primary w-full py-3 text-base">
            {loading ? t('auth:signingIn') : t('auth:signInCta')}
          </button>

          <p className="text-center text-sm text-white/50 pt-2">
            {t('auth:noAccount')}{' '}
            <Link to="/register" className="font-semibold hover:underline" style={{ color: 'var(--accent)' }}>
              {t('auth:createOneFree')}
            </Link>
          </p>
        </form>
      </motion.div>
    </div>
  )
}
