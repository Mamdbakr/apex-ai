/**
 * src/store/useStore.js — APEX AI v14 zustand store.
 *
 * Auth model: cookie-based. The store ONLY caches the user/profile object
 * for fast first-paint. The actual session lives in the HttpOnly cookie.
 *
 * Theme: derived from gender by default ('m' → masculine, 'f' → feminine),
 * but the user can override it. The override is persisted; if no override is
 * set, the theme follows whatever gender the user signed up with.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

function deriveTheme(gender, override) {
  if (override === 'masculine' || override === 'feminine') return override
  return gender === 'f' ? 'feminine' : 'masculine'
}

const useStore = create(
  persist(
    (set, get) => ({
      // ── Auth (just a cache; the real session is in the cookie) ────────────
      user:    null,    // { user_id, name, email, gender, role }
      profile: null,    // { age, weight_kg, height_cm, ... }
      authReady: false, // becomes true after the first /auth/me probe

      setUser: (user, profile) => {
        const gender = user?.gender || 'm'
        const override = get().themeOverride
        set({
          user, profile,
          authReady: true,
          theme: deriveTheme(gender, override),
        })
      },
      setProfile: (profile) => set({ profile }),
      setAuthReady: (v) => set({ authReady: !!v }),
      clearUser: () => set({
        user: null, profile: null, authReady: true,
        mlData: null, intelligence: null,
        chatHistory: [],
      }),

      // ── Theme system (the headline UI requirement) ────────────────────────
      // 'masculine' = dark athletic (green/cyan/purple)
      // 'feminine'  = elegant feminine (rose/violet/peach)
      theme: 'masculine',
      themeOverride: null, // null | 'masculine' | 'feminine'
      setTheme: (theme) => {
        const override = (theme === 'masculine' || theme === 'feminine') ? theme : null
        const gender = get().user?.gender || 'm'
        set({ themeOverride: override, theme: deriveTheme(gender, override) })
      },
      resetTheme: () => {
        const gender = get().user?.gender || 'm'
        set({ themeOverride: null, theme: deriveTheme(gender, null) })
      },

      // ── ML & dashboard caches ─────────────────────────────────────────────
      mlData:       null,
      intelligence: null,
      insights:     [],
      forecast:     null,
      cluster:      null,
      setMLData:    (d) => set({ mlData: d }),
      setIntelligence: (d) => set({
        intelligence: d,
        insights:  d?.insights  || [],
        forecast:  d?.forecast_30d || null,
        cluster:   d?.cluster   || null,
      }),

      dashboardData:    null,
      setDashboardData: (d) => set({ dashboardData: d }),

      // ── Chat history (session-scoped, not persisted) ──────────────────────
      chatHistory: [],
      addMessage:  (msg) => set(s => ({ chatHistory: [...s.chatHistory, msg] })),
      clearChat:   ()    => set({ chatHistory: [] }),

      // ── UI state ──────────────────────────────────────────────────────────
      sidebarOpen: true,
      toggleSidebar: () => set(s => ({ sidebarOpen: !s.sidebarOpen })),
    }),
    {
      name: 'apex-store-v14',
      // Only persist what's safe to keep in localStorage. The user object is a
      // *cache* — the source of truth is /auth/me, which we re-call on app boot.
      partialize: s => ({
        user: s.user,
        profile: s.profile,
        themeOverride: s.themeOverride,
        theme: s.theme,
      }),
    }
  )
)

export default useStore
