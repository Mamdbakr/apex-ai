import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Check, Zap, ArrowLeft, Star } from 'lucide-react'

/**
 * Pricing.jsx — UI/UX-polished drop-in replacement.
 *
 * No logic changed. Polish:
 *   • Replaced broken `neon-glow` (undefined in index.css) with real `card-accent`.
 *   • "Most Popular" badge gets a star; cards lift on hover via .card system.
 *   • Background uses theme tokens instead of hardcoded hex.
 */
const PLANS = [
  {
    tier: 'Free', price: 0, color: '#7b5cff',
    features: ['5 AI predictions/day','Chatbot (5 messages/day)','Basic dashboard','Progress tracking'],
    limits:   ['No CV Trainer','No weight forecasting','No plateau detection'],
    cta: 'Start Free', to: '/register',
  },
  {
    tier: 'Pro', price: 12, color: '#00ff88', popular: true,
    features: ['Unlimited ML predictions','Unlimited AI coaching chat','Computer Vision Trainer','30-day weight forecast','Plateau detection','User clustering & insights','Full analytics dashboard','Priority model inference'],
    limits: [],
    cta: 'Get Pro →', to: '/register',
  },
  {
    tier: 'Elite', price: 29, color: '#00d4ff',
    features: ['Everything in Pro','Multi-user management','REST API access','Custom model fine-tuning','Webhook integrations','Dedicated support','SLA guarantee'],
    limits: [],
    cta: 'Contact Sales', to: '/register',
  },
]

export default function Pricing() {
  return (
    <div className="min-h-screen text-white px-6 py-20" style={{ background: 'var(--bg-primary)' }}>
      <div className="max-w-5xl mx-auto">
        <Link to="/" className="inline-flex items-center gap-2 text-white/40 hover:text-white text-sm mb-10 transition-colors">
          <ArrowLeft size={14} /> Back to home
        </Link>
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 badge badge-green mb-6">
            <Zap size={12} /> Simple pricing
          </div>
          <h1 className="text-5xl font-bold font-display mb-4">
            Start free, scale as you <span className="gradient-text">grow</span>
          </h1>
          <p className="text-white/40 text-lg max-w-xl mx-auto">
            Every plan includes the full AI stack. Upgrade when you need more.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {PLANS.map((p, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }} viewport={{ once: true }}
              className={`card relative flex flex-col ${p.popular ? 'card-accent' : ''}`}>
              {p.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 badge badge-green flex items-center gap-1">
                  <Star size={11} /> Most Popular
                </div>
              )}
              <div className="mb-6">
                <div className="text-xs font-bold tracking-widest uppercase mb-2" style={{ color: p.color }}>{p.tier}</div>
                <div className="flex items-end gap-1 mb-1">
                  <span className="text-4xl font-bold font-display">${p.price}</span>
                  <span className="text-white/30 text-sm mb-1">/month</span>
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
              <Link to={p.to}
                className={`btn text-center ${p.popular ? 'btn-primary' : 'btn-ghost'}`}
                style={p.popular ? undefined : { color: 'var(--text-primary)' }}>
                {p.cta}
              </Link>
            </motion.div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <p className="text-white/30 text-sm">All plans include: cookie-session auth · SQLite/PostgreSQL · Full API · Self-hostable</p>
        </div>
      </div>
    </div>
  )
}
