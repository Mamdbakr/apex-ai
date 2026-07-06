import { useTranslation } from 'react-i18next'
import { Languages } from 'lucide-react'

const LANGS = [
  { code: 'en', label: 'English' },
  { code: 'ar', label: 'العربية' },
]

/**
 * LanguageSwitcher — instant language toggle. Preference is persisted to
 * localStorage by the i18next language detector and restored on next visit.
 *
 * variant="pills"    — two labelled buttons side by side (landing nav, auth pages)
 * variant="sidebar"  — full-width row matching .sidebar-link (app layout)
 */
export default function LanguageSwitcher({ variant = 'pills', expanded = true }) {
  const { i18n, t } = useTranslation()
  const current = i18n.resolvedLanguage === 'ar' ? 'ar' : 'en'

  if (variant === 'sidebar') {
    const next = current === 'ar' ? 'en' : 'ar'
    return (
      <button
        onClick={() => i18n.changeLanguage(next)}
        className="sidebar-link w-full text-start"
        title={t('common:language')}
        aria-label={t('common:language')}
      >
        <Languages size={18} className="flex-shrink-0" style={{ color: 'var(--accent2)' }} />
        {expanded && (
          <span>
            {t('common:language')}:{' '}
            <span style={{ color: 'var(--accent2)' }}>
              {LANGS.find((l) => l.code === next)?.label}
            </span>
          </span>
        )}
      </button>
    )
  }

  return (
    <div className="flex items-center gap-1 rounded-xl border p-1"
         style={{ borderColor: 'var(--border)', background: 'var(--bg-glass)' }}
         role="group" aria-label={t('common:language')}>
      {LANGS.map(({ code, label }) => (
        <button
          key={code}
          onClick={() => i18n.changeLanguage(code)}
          className="px-2.5 py-1 rounded-lg text-xs font-semibold transition-all"
          style={current === code
            ? { background: 'var(--accent)', color: 'var(--bg-primary)' }
            : { color: 'var(--text-muted)' }}
          aria-pressed={current === code}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
