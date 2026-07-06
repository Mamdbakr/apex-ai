import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

/**
 * i18n bootstrap — singleton instance shared by the whole app.
 *
 * Translation JSON files live in ./locales/<lng>/<ns>.json and are
 * lazy-loaded through a tiny dynamic-import backend, so a namespace is only
 * fetched when a page that uses it mounts (Vite code-splits each JSON file).
 *
 * Language preference is persisted in localStorage ("apex-lang") by the
 * detector, and restored automatically on the next visit.
 */

// Lazy backend: each locale JSON becomes its own Vite chunk.
const lazyImportBackend = {
  type: 'backend',
  init() {},
  read(lng, ns, callback) {
    import(`./locales/${lng}/${ns}.json`)
      .then((mod) => callback(null, mod.default))
      .catch((err) => callback(err, null))
  },
}

/** Keep <html> dir/lang in sync so the whole document flips RTL/LTR. */
function applyDocumentDirection(lng) {
  const root = document.documentElement
  // Detector may report a region variant ("en-US"); normalise to the base
  // language actually being served ("en" / "ar").
  const base = (i18n.resolvedLanguage || lng || 'en').split('-')[0]
  root.dir = i18n.dir(base)
  root.lang = base
}

i18n
  .use(lazyImportBackend)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    supportedLngs: ['en', 'ar'],
    fallbackLng: 'en',
    load: 'languageOnly',
    ns: ['common'],
    defaultNS: 'common',
    interpolation: { escapeValue: false }, // React already escapes
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
      lookupLocalStorage: 'apex-lang',
    },
    react: { useSuspense: true },
  })

i18n.on('languageChanged', applyDocumentDirection)
if (i18n.resolvedLanguage) applyDocumentDirection(i18n.resolvedLanguage)

export default i18n
