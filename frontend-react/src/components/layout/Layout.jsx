import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { LayoutDashboard, MessageSquare, Brain, Eye, TrendingUp, User, LogOut,
         ChevronLeft, Menu, Zap, Sparkles, Flame } from 'lucide-react'
import useStore from '../../store/useStore'
import { authAPI } from '../../lib/api'
import toast from 'react-hot-toast'

const NAV = [
  { to: '/dashboard',   icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/chat',        icon: MessageSquare,   label: 'Apex Coach' },
 { to: '/predictions', icon: Brain, label: 'Stats' },
{ to: '/vision',      icon: Eye,   label: 'Camera Trainer' },
  { to: '/progress',    icon: TrendingUp,      label: 'Progress' },
  { to: '/profile',     icon: User,            label: 'Profile' },
]

export default function Layout() {
  const user         = useStore((s) => s.user)
  const profile      = useStore((s) => s.profile)
  const sidebarOpen  = useStore((s) => s.sidebarOpen)
  const toggleSidebar = useStore((s) => s.toggleSidebar)
  const clearUser    = useStore((s) => s.clearUser)
  const theme        = useStore((s) => s.theme)
  const setTheme     = useStore((s) => s.setTheme)
  const resetTheme   = useStore((s) => s.resetTheme)
  const navigate = useNavigate()

  async function handleLogout() {
    try {
      await authAPI.logout()
    } catch {
      // even if the server call failed, clear local state anyway —
      // the user wants out
    }
    clearUser()
    toast.success('See you soon!')
    navigate('/login')
  }

  function handleThemeToggle() {
    // Click toggles between masculine and feminine. Long-press / double-click
    // could reset to gender-default; for now, third click goes back to default.
    if (theme === 'masculine') setTheme('feminine')
    else if (theme === 'feminine') resetTheme()
    else setTheme('masculine')
  }

  const themeIcon = theme === 'feminine' ? Sparkles : Flame
  const ThemeIcon = themeIcon

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
      {/* Sidebar */}
      <motion.aside
        animate={{ width: sidebarOpen ? 240 : 72 }}
        transition={{ duration: 0.25, ease: 'easeInOut' }}
        className="flex-shrink-0 h-full border-r flex flex-col overflow-hidden z-20"
        style={{
          background: 'var(--bg-secondary)',
          borderColor: 'var(--border)',
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 p-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
               style={{ background: 'var(--grad-accent)' }}>
            <Zap size={16} style={{ color: 'var(--bg-primary)' }} />
          </div>
          <AnimatePresence>
            {sidebarOpen && (
              <motion.span
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="font-bold text-lg text-white font-display"
              >
                APEX<span style={{ color: 'var(--accent)' }}>AI</span>
              </motion.span>
            )}
          </AnimatePresence>
          <button onClick={toggleSidebar} className="ml-auto text-white/30 hover:text-white/70 transition-colors"
                  aria-label="Toggle sidebar">
            {sidebarOpen ? <ChevronLeft size={16} /> : <Menu size={16} />}
          </button>
        </div>

        {/* User avatar */}
        <div className="p-3 border-b" style={{ borderColor: 'var(--border)' }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 text-sm font-bold text-white"
                 style={{ background: 'var(--grad-cool)' }}>
              {(user?.name || profile?.name || 'U').charAt(0).toUpperCase()}
            </div>
            <AnimatePresence>
              {sidebarOpen && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="text-sm font-semibold text-white truncate max-w-[140px]">
                    {user?.name || profile?.name || 'Athlete'}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--accent)' }}>Pro Member</div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to} to={to}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={18} className="flex-shrink-0" />
              <AnimatePresence>
                {sidebarOpen && (
                  <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    {label}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          ))}
        </nav>

        {/* Bottom — theme toggle + logout */}
        <div className="p-3 border-t space-y-1" style={{ borderColor: 'var(--border)' }}>
          <button onClick={handleThemeToggle} className="sidebar-link w-full text-left"
                  title={`Theme: ${theme}`}>
            <ThemeIcon size={18} className="flex-shrink-0" style={{ color: 'var(--accent)' }} />
            <AnimatePresence>
              {sidebarOpen && (
                <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  Theme: <span style={{ color: 'var(--accent)' }}>{theme === 'feminine' ? 'Feminine' : 'Athletic'}</span>
                </motion.span>
              )}
            </AnimatePresence>
          </button>

          <button onClick={handleLogout} className="sidebar-link w-full text-left"
                  style={{ color: 'var(--warn)' }}>
            <LogOut size={18} className="flex-shrink-0" />
            <AnimatePresence>
              {sidebarOpen && (
                <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  Logout
                </motion.span>
              )}
            </AnimatePresence>
          </button>
        </div>
      </motion.aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto">
        <div className="p-6 max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
