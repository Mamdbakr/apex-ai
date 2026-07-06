import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Check, Zap, ArrowLeft, Star } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import LanguageSwitcher from '../components/LanguageSwitcher'

/**
 * Pricing.jsx — all copy comes from the landing i18n namespace
 * (tiers.*.fullFeatures / limits are arrays translated per language).
 */
const PLAN_DEFS = [
  { key: 'free',  price: 0,  color: '#7b5cff', hasLimits: true,  ctaKey: 'cta' },
  { key: 'pro',   price: 12, color: '#00ff88', popular: true,    ctaKey: 'ctaArrow' },
  { key: 'elite', price: 29, color: '#00d4ff',                   ctaKey: 'ctaSales' },
]

export default function Pricing() {
  const { t } = useTranslation(['landing', 'common'])

  const plans = PLAN_DEFS.map(p => ({
    ...p,
    tier:     t(`landing:tiers.${p.key}.name`),
    features: t(`landing:tiers.${p.key}.fullFeatures`, { returnObjects: true }),
    limits:   p.hasLimits ? t(`landing:tiers.${p.key}.limits`, { returnObjects: true }) : [],
    cta:      t(`landing:tiers.${p.key}.${p.ctaKey}`),
  }))

  return (
    <div className="min-h-screen text-white px-6 py-20" style={{ background: 'var(--bg-primary)' }}>
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-10">
          <Link to="/" className="inline-flex items-center gap-2 text-white/40 hover:text-white text-sm transition-colors">
            <ArrowLeft size={14} className="rtl-flip" /> {t('landing:backToHome')}
          </Link>
          <LanguageSwitcher />
        </div>
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 badge badge-green mb-6">
            <Zap size={12} /> {t('landing:simplePricing')}
          </div>
          <h1 className="text-5xl font-bold font-display mb-4">
            {t('landing:pricingHeroTitle')} <span className="gradient-text">{t('landing:pricingHeroAccent')}</span>
          </h1>
          <p className="text-white/40 text-lg max-w-xl mx-auto">
            {t('landing:pricingHeroSubtitle')}
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {plans.map((p, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }} viewport={{ once: true }}
              className={`card relative flex flex-col ${p.popular ? 'card-accent' : ''}`}>
              {p.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 badge badge-green flex items-center gap-1">
                  <Star size={11} /> {t('landing:mostPopular')}
                </div>
              )}
              <div className="mb-6">
                <div className="text-xs font-bold tracking-widest uppercase mb-2" style={{ color: p.color }}>{p.tier}</div>
                <div className="flex items-end gap-1 mb-1">
                  <span className="text-4xl font-bold font-display">${p.price}</span>
                  <span className="text-white/30 text-sm mb-1">{t('common:units.month')}</span>
                </div>
              </div>
              <ul className="space-y-2.5 flex-1 mb-6">
                {p.features.map((f, j) => (
                  <li key={j} className="flex items-start gap-2 text-sm">
                    <Check size={14} className="flex-shrink-0 mt-0.5" style={{ color: p.color }} />
                    <span className="text-white/70">{f}</span>
                  </li>
                ))}
                {p.limits.map((f, j) => (
                  <li key={`l${j}`} className="flex items-start gap-2 text-sm opacity-40">
                    <span className="flex-shrink-0 mt-0.5 w-3.5 text-center">✗</span>
                    <span className="text-white/40">{f}</span>
                  </li>
                ))}
              </ul>
              <Link to="/register"
                className={`btn text-center ${p.popular ? 'btn-primary' : 'btn-ghost'}`}
                style={p.popular ? undefined : { color: 'var(--text-primary)' }}>
                {p.cta}
              </Link>
            </motion.div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <p className="text-white/30 text-sm">{t('landing:pricingFootnote')}</p>
        </div>
      </div>
    </div>
  )
}
