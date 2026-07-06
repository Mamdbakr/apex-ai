import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Zap, Brain, TrendingUp, Eye, Star, ArrowRight, Check } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import LanguageSwitcher from '../components/LanguageSwitcher'

/**
 * Landing.jsx — all copy comes from the landing i18n namespace; layout uses
 * logical (start/end) utilities so the page mirrors correctly in RTL.
 */

const FEATURES = [
  { key: 'coach',       icon: Brain,      color: '#00ff88' },
  { key: 'camera',      icon: Eye,        color: '#00d4ff' },
  { key: 'predictions', icon: TrendingUp, color: '#7b5cff' },
  { key: 'progress',    icon: Zap,        color: '#ffd93d' },
]

export default function Landing() {
  const { t } = useTranslation(['landing', 'common'])

  const STATS = [
    { n: t('landing:stats.live'),  label: t('landing:stats.cameraCoaching') },
    { n: t('landing:stats.apex'),  label: t('landing:stats.chatbot') },
    { n: '∞',                      label: t('landing:stats.personalization') },
    { n: t('landing:stats.twentyFourSeven'), label: t('landing:stats.alwaysOn') },
  ]

  const PRICING = [
    {
      tier: t('landing:tiers.free.name'), price: '$0', color: '#7b5cff', desc: t('landing:tiers.free.desc'),
      features: t('landing:tiers.free.features', { returnObjects: true }),
      cta: t('landing:tiers.free.cta'),
    },
    {
      tier: t('landing:tiers.pro.name'), price: '$12', color: '#00ff88', desc: t('landing:tiers.pro.desc'), popular: true,
      features: t('landing:tiers.pro.features', { returnObjects: true }),
      cta: t('landing:tiers.pro.cta'),
    },
    {
      tier: t('landing:tiers.elite.name'), price: '$29', color: '#00d4ff', desc: t('landing:tiers.elite.desc'),
      features: t('landing:tiers.elite.features', { returnObjects: true }),
      cta: t('landing:tiers.elite.cta'),
    },
  ]

  return (
    <div className="min-h-screen text-white overflow-x-hidden" style={{ background: 'var(--bg-primary)' }}>
      {/* Nav */}
      <nav className="fixed top-0 inset-x-0 z-50 glass" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl font-display">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#00ff88] to-[#00d4ff] flex items-center justify-center">
              <Zap size={14} className="text-[#060d1a]" />
            </div>
            APEX<span className="text-[#00ff88]">AI</span>
          </div>
          <div className="flex items-center gap-3">
            <LanguageSwitcher />
            <Link to="/pricing" className="text-sm text-white/50 hover:text-white transition-colors">{t('common:pricing')}</Link>
            <Link to="/login"   className="btn btn-ghost text-sm px-4 py-2" style={{ color: 'var(--text-primary)' }}>{t('common:signIn')}</Link>
            <Link to="/register" className="btn-primary text-sm px-4 py-2">{t('common:startFree')}</Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-40 pb-28 px-6 text-center relative">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="aurora w-[600px] h-[600px] top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2"
               style={{ background: 'var(--accent)', opacity: 0.06 }} />
          <div className="aurora w-[400px] h-[400px] top-1/2 left-1/4"
               style={{ background: 'var(--accent3)', opacity: 0.06 }} />
        </div>
        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <div className="inline-flex items-center gap-2 badge badge-green mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00ff88] animate-pulse" />
            {t('landing:badge')}
          </div>
          <h1 className="text-6xl md:text-7xl font-bold font-display leading-tight mb-6">
            {t('landing:heroLine1')}<br />
            <span className="gradient-text">{t('landing:heroLine2')}</span>
          </h1>
          <p className="text-xl text-white/50 max-w-2xl mx-auto mb-10 leading-relaxed">
            {t('landing:heroSubtitle')}
          </p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link to="/register" className="btn-primary flex items-center gap-2 text-base px-7 py-3">
              {t('landing:startForFree')} <ArrowRight size={16} className="rtl-flip" />
            </Link>
            <Link to="/login" className="btn btn-ghost flex items-center gap-2 text-base px-7 py-3" style={{ color: 'var(--text-primary)' }}>
              {t('common:signIn')}
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Stats */}
      <section className="py-10 px-6" style={{ borderBlock: '1px solid var(--border)' }}>
        <div className="max-w-4xl mx-auto grid grid-cols-4 gap-6 text-center">
          {STATS.map((s, i) => (
            <motion.div key={i} initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
              transition={{ delay: i * 0.1 }} viewport={{ once: true }}>
              <div className="text-3xl font-bold font-display gradient-text">{s.n}</div>
              <div className="text-sm text-white/40 mt-1">{s.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="eyebrow justify-center" style={{ display: 'inline-flex' }}>{t('landing:coreFeatures')}</div>
            <h2 className="text-4xl font-bold font-display mt-2">{t('landing:featuresTitle')} <span className="gradient-text">{t('landing:featuresTitleAccent')}</span></h2>
          </div>
          <div className="grid md:grid-cols-2 gap-5">
            {FEATURES.map((f, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                className="card"
              >
                <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-4"
                     style={{ background: f.color + '18', border: `1px solid ${f.color}30` }}>
                  <f.icon size={20} style={{ color: f.color }} />
                </div>
                <h3 className="font-bold text-lg font-display mb-2">{t(`landing:features.${f.key}.title`)}</h3>
                <p className="text-white/50 text-sm leading-relaxed">{t(`landing:features.${f.key}.desc`)}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section className="py-24 px-6" style={{ background: 'var(--bg-secondary)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <div className="eyebrow justify-center" style={{ display: 'inline-flex' }}>{t('landing:pricingEyebrow')}</div>
            <h2 className="text-4xl font-bold font-display mt-2">{t('landing:pricingTitle')}</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {PRICING.map((p, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }} viewport={{ once: true }}
                className={`card relative ${p.popular ? 'card-accent' : ''}`}
              >
                {p.popular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 badge badge-green flex items-center gap-1">
                    <Star size={11} /> {t('landing:mostPopular')}
                  </div>
                )}
                <div className="text-xs font-bold tracking-widest uppercase mb-1" style={{ color: p.color }}>{p.tier}</div>
                <div className="text-4xl font-bold font-display mb-1">{p.price}<span className="text-sm text-white/40 font-normal">{t('common:units.perMonth')}</span></div>
                <div className="text-sm text-white/40 mb-6">{p.desc}</div>
                <ul className="space-y-2 mb-8">
                  {p.features.map((f, j) => (
                    <li key={j} className="flex items-center gap-2 text-sm">
                      <Check size={14} style={{ color: p.color }} />
                      <span className="text-white/70">{f}</span>
                    </li>
                  ))}
                </ul>
                <Link
                  to="/register"
                  className={`btn w-full text-center ${p.popular ? 'btn-primary' : 'btn-ghost'}`}
                  style={p.popular ? undefined : { color: 'var(--text-primary)' }}
                >
                  {p.cta}
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-6 text-center" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="font-bold font-display mb-2">APEX<span className="text-[#00ff88]">AI</span></div>
        <div className="text-xs text-white/30">{t('landing:footer', { year: new Date().getFullYear() })}</div>
      </footer>
    </div>
  )
}
