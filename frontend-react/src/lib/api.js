/**
 * src/lib/api.js — APEX AI v14 frontend HTTP client.
 *
 * Auth model: cookie-based sessions.
 *   • The browser stores nothing — the backend sets an HttpOnly cookie.
 *   • Every request goes out with `withCredentials: true` so the cookie is sent.
 *   • There are no access/refresh tokens, no Authorization headers, no token
 *     storage in localStorage. A 401 just means "not signed in" — it doesn't
 *     trigger a refresh, doesn't loop, doesn't race the dashboard.
 *
 * To talk to the backend from a different origin, set CORS_ORIGINS in the
 * backend .env to your frontend URL (NOT '*' — you can't credential a wildcard).
 */
import axios from 'axios'

const BASE =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) ||
  'http://localhost:8000'

const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
  withCredentials: true,        // ← THE important line
})

// ── Response interceptor ─────────────────────────────────────────────────────
// On 401, we DON'T try to refresh anything (there's nothing to refresh). We
// just bubble the error up so the caller can decide what to do (usually
// redirect to /login). This is the most predictable behaviour we can give.
api.interceptors.response.use(
  (res) => res,
  (err) => {
    // Surface a friendlier shape so callers can do:
    //   try { ... } catch (e) { toast.error(e.detail) }
    if (err.response?.data?.detail) {
      err.detail = err.response.data.detail
    } else if (err.response?.data?.message) {
      err.detail = err.response.data.message
    } else {
      err.detail = err.message
    }
    return Promise.reject(err)
  },
)

// ── Auth ─────────────────────────────────────────────────────────────────────
export const authAPI = {
  signup:        (d)   => api.post('/auth/signup', d),
  signin:        (d)   => api.post('/auth/signin', d),
  logout:        ()    => api.post('/auth/logout'),
  logoutAll:     ()    => api.post('/auth/logout-all'),
  me:            ()    => api.get('/auth/me'),
  sessions:      ()    => api.get('/auth/sessions'),
  revokeSession: (sid) => api.delete(`/auth/sessions/${sid}`),
  checkEmail:    (email) =>
    api.get('/auth/check-email', { params: { email } }),
}

// ── Chat ─────────────────────────────────────────────────────────────────────
export const chatAPI = {
  send:    (d)   => api.post('/chat', d),
  history: (uid) => api.get(`/chat/history/${uid}`),
  clear:   (uid) => api.delete(`/chat/history/${uid}`),
  stats:   ()    => api.get('/chat/stats'),
}

// ── ML predictions ───────────────────────────────────────────────────────────
export const predictAPI = {
  all:          (overrides = {}) => api.post('/predict/all',           overrides),
  calories:     (overrides = {}) => api.post('/predict/calories',      overrides),
  weightChange: (overrides = {}) => api.post('/predict/weight-change', overrides),
  fitnessLevel: (overrides = {}) => api.post('/predict/fitness-level', overrides),
  explain:      (overrides = {}) => api.post('/predict/explain',       overrides),
  history:      ()               => api.get('/predict/history'),
}

// ── Recommendations ──────────────────────────────────────────────────────────
export const recommendAPI = {
  get:     ({ top_k = 5 } = {}) => api.get('/recommend', { params: { top_k } }),
  history: ()                   => api.get('/recommend/history'),
}

// ── User data + dashboard ────────────────────────────────────────────────────
export const userAPI = {
  getProfile:       ()  => api.get('/user-data/profile'),
  updateProfile:    (d) => api.post('/user-data/profile', d),
  logWorkout:       (d) => api.post('/user-data/workout', d),
  getWorkouts:      ()  => api.get('/user-data/workouts'),
  getWeightHistory: ()  => api.get('/user-data/weight-history'),
  getDashboard:     ()  => api.get('/user-data/dashboard'),
  getDashboardFull: ()  => api.get('/user-data/dashboard-full'),
}

// ── Data pipeline ────────────────────────────────────────────────────────────
export const dataAPI = {
  workout:   (d) => api.post('/data/workout',   d),
  weight:    (d) => api.post('/data/weight',    d),
  nutrition: (d) => api.post('/data/nutrition', d),
}

// ── Vision ───────────────────────────────────────────────────────────────────
// The browser sends the apex_session cookie automatically on the WS handshake
// when the connection is same-origin. For cross-origin dev, we don't have any
// good way to attach the signed cookie (browsers won't let JS read HttpOnly
// cookies), so the ?session=… fallback is intentionally not exposed here —
// run the frontend behind the same host (or a Vite proxy) for live streaming.
export const visionAPI = {
  analyze: (formData, sessionId = 'default') =>
    api.post(`/vision/analyze?session_id=${encodeURIComponent(sessionId)}`,
             formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
  reset:   (sessionId = 'default') =>
    api.post(`/vision/reset?session_id=${encodeURIComponent(sessionId)}`),
  finish:  ({ session_id, sets, duration_min, notes }) =>
    api.post('/vision/session/finish', null, {
      params: { session_id, sets, duration_min, notes },
    }),
  history: () => api.get('/vision/history'),
  streamUrl: (sessionId = 'default') => {
    const proto = BASE.startsWith('https') ? 'wss' : 'ws'
    const host  = BASE.replace(/^https?:\/\//, '')
    return `${proto}://${host}/vision/stream?sid=${encodeURIComponent(sessionId)}`
  },
}

// ── System ───────────────────────────────────────────────────────────────────
export const systemAPI = { health: () => api.get('/health') }

// ── AI Dashboard insights ────────────────────────────────────────────────────
export const insightsAPI = {
  dashboard: ()       => api.get('/insights/dashboard'),
  forecast:  (days)   => api.get('/insights/forecast', { params: { days: days || 90 } }),
  anomalies: ()       => api.get('/insights/anomalies'),
  cohort:    ()       => api.get('/insights/cohort'),
  refresh:   ()       => api.post('/insights/refresh'),
}

export default api
