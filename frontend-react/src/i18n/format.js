import i18n from './index'

/**
 * Locale-aware number/date helpers. Components call these instead of raw
 * toLocaleString / toLocaleDateString so output follows the active language
 * (en-US for English, ar for Arabic).
 */

export function currentLocale() {
  return i18n.resolvedLanguage === 'ar' ? 'ar' : 'en-US'
}

export function formatNumber(value, options) {
  if (value == null || Number.isNaN(Number(value))) return value ?? '—'
  return new Intl.NumberFormat(currentLocale(), options).format(value)
}

export function formatDate(value, options = { month: 'short', day: 'numeric' }) {
  const d = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(d.getTime())) return '—'
  return new Intl.DateTimeFormat(currentLocale(), options).format(d)
}

export function formatTime(value) {
  const d = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(d.getTime())) return '—'
  return new Intl.DateTimeFormat(currentLocale(), { hour: 'numeric', minute: '2-digit' }).format(d)
}

/**
 * Backend error messages arrive as fixed English strings in `detail`.
 * Known ones are mapped to translation keys; unknown ones fall back to the
 * raw string (better than hiding the reason), then to a generic key.
 */
const API_ERROR_KEYS = {
  'Email already registered': 'emailRegistered',
  'Incorrect email or password': 'incorrectCredentials',
  'User no longer exists': 'userGone',
  'Not authenticated — please sign in.': 'notAuthenticated',
  'Session not found': 'sessionNotFound',
  "Cannot read another user's history": 'forbiddenHistory',
  "Cannot clear another user's history": 'forbiddenHistory',
}

export function apiErrorMessage(t, detail, fallbackKey) {
  const key = detail && API_ERROR_KEYS[detail]
  if (key) return t(`common:apiErrors.${key}`)
  return detail || t(fallbackKey)
}
