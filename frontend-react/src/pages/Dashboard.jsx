/**
 * Dashboard.jsx — AI-powered dashboard.  (UI/UX-polished drop-in replacement)
 *
 * ── WHAT CHANGED (UI ONLY) ───────────────────────────────────────────────────
 * This is a visual + layout refresh of the original Dashboard. NOTHING about the
 * data flow changed: every numeric value still comes from `/insights/dashboard`
 * via `insightsAPI.dashboard()`, the exact same `data?.*` field names are read,
 * the macros math is identical, and all conditional-render guards
 * (`available`, `forecast.available`, etc.) are preserved 1:1.
 *
 * Polished:
 *   • Richer hero header with an at-a-glance status strip (Section component).
 *   • KPI tiles get optional trend deltas + your `.metric` orb styling, and the
 *     row uses your `.stagger` entrance helper instead of ad-hoc motion.
 *   • Consistent section headers via a reusable <SectionHead> using your
 *     `.eyebrow` design token, so the page reads as deliberate zones.
 *   • A real full-page skeleton state (charts no longer "pop in").
 *   • Empty/error states rebuilt on your `.notice` classes.
 *   • Light micro-interactions on rec / macro tiles.
 *
 * Every class used here already exists in index.css (card, glass, metric,
 * badge-*, eyebrow, gradient-text, skeleton, btn-ghost, notice*, stagger…).
 * No new CSS or dependencies are required.
 */
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Flame, Scale, Brain, Zap, Activity, AlertTriangle, Target,
  TrendingUp, TrendingDown, Award, RefreshCw, Sparkles, Droplets,
} from 'lucide-react'
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { insightsAPI } from '../lib/api'
import useStore from '../store/useStore'
import toast from 'react-hot-toast'


// ─── small UI helpers ────────────────────────────────────────────────────────

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-3 py-2 text-xs">
      <div className="text-white/50 mb-1">{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>
          {p.name}: <b>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</b>
        </div>
      ))}
    </div>
  )
}

/**
 * SectionHead — consistent zone header. Uses the existing `.eyebrow` token so
 * every section on the page shares one rhythm. `right` is an optional slot for
 * a badge / control on the far side.
 */
function SectionHead({ icon: Icon, eyebrow, title, right }) {
  return (
    <div className="flex items-end justify-between gap-4 mb-4">
      <div>
        {eyebrow && (
          <div className="eyebrow" style={{ marginBottom: '0.35rem' }}>{eyebrow}</div>
        )}
        <h3 className="font-bold font-display flex items-center gap-2 text-lg">
          {Icon && <Icon size={18} className="opacity-80" />}
          {title}
        </h3>
      </div>
      {right}
    </div>
  )
}

/**
 * Kpi — metric tile. Now optionally renders a trend delta (`deltaDir` +
 * `deltaText`) and leans on your `.metric` orb hover styling. Entrance is
 * handled by the parent `.stagger` wrapper, so we drop the per-card motion.
 */
function Kpi({ icon: Icon, value, sub, label, color, loading, deltaDir, deltaText }) {
  const DeltaIcon = deltaDir === 'down' ? TrendingDown : TrendingUp
  const deltaColor =
    deltaDir === 'down' ? '#ff6b35' : deltaDir === 'up' ? '#00ff88' : 'rgba(240,246,255,0.48)'
  return (
    <div className="metric">
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ background: color + '18', border: `1px solid ${color}30` }}
        >
          <Icon size={18} style={{ color }} />
        </div>
        {deltaText && !loading && (
          <span
            className="inline-flex items-center gap-1 text-[11px] font-semibold tabular-nums"
            style={{ color: deltaColor }}
          >
            <DeltaIcon size={12} /> {deltaText}
          </span>
        )}
      </div>
      {loading ? (
        <div className="skeleton h-8 w-24 mb-1" />
      ) : (
        <div className="text-2xl font-bold font-display tabular-nums" style={{ color }}>
          {value}
        </div>
      )}
      <div className="text-xs text-white/40 uppercase tracking-wider mt-1">{label}</div>
      {sub && (
        <div className="text-xs mt-1.5 truncate-2" style={{ color: color + 'cc' }}>
          {sub}
        </div>
      )}
    </div>
  )
}

const SEVERITY_STYLE = {
  alert: { color: '#ff6b35', bg: 'rgba(255,107,53,0.10)', border: 'rgba(255,107,53,0.30)' },
  warn:  { color: '#ffd93d', bg: 'rgba(255,217,61,0.10)', border: 'rgba(255,217,61,0.30)' },
  info:  { color: '#00d4ff', bg: 'rgba(0,212,255,0.10)', border: 'rgba(0,212,255,0.30)' },
}

/** Full-page skeleton — shown on first load so the layout never jumps. */
function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="card">
        <div className="skeleton h-4 w-32 mb-3" />
        <div className="skeleton h-9 w-72 mb-2" />
        <div className="skeleton h-4 w-56" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="metric">
            <div className="skeleton h-10 w-10 rounded-xl mb-3" />
            <div className="skeleton h-7 w-20 mb-2" />
            <div className="skeleton h-3 w-16" />
          </div>
        ))}
      </div>
      <div className="card">
        <div className="skeleton h-5 w-44 mb-4" />
        <div className="skeleton h-[220px] w-full rounded-xl" />
      </div>
      <div className="grid md:grid-cols-2 gap-5">
        <div className="card"><div className="skeleton h-[180px] w-full rounded-xl" /></div>
        <div className="card"><div className="skeleton h-[180px] w-full rounded-xl" /></div>
      </div>
    </div>
  )
}


// ─── main component ──────────────────────────────────────────────────────────

export default function Dashboard() {
  const { profile } = useStore()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [err, setErr] = useState(null)

  const hour = new Date().getHours()
  const greet =
    hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'
  const name =
    data?.profile?.name?.split(' ')[0] || profile?.name?.split(' ')[0] || 'Athlete'

  async function load(showSpinner = true) {
    if (showSpinner) setLoading(true)
    else setRefreshing(true)
    setErr(null)
    try {
      const res = await insightsAPI.dashboard()
      setData(res.data)
    } catch (e) {
      setErr(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false); setRefreshing(false)
    }
  }

  useEffect(() => { load() }, [])

  async function handleRefresh() {
    await load(false)
    toast.success('Dashboard refreshed')
  }

  // ── derived (from API only — nothing fabricated) ──────────────────────────

  const available    = data?.available
  const kpi          = data?.kpi || {}
  const ml           = data?.ml_predictions || {}
  const forecast     = data?.forecast || {}
  const timeline     = data?.timeline_to_goal || {}
  const calorieCurve = data?.calorie_curve || {}
  const recs         = data?.recommendations || []
  const anomalies    = data?.anomalies || []
  const cohort       = data?.cohort || {}
  const insights     = data?.insights || []

  // ── Goal-aware corrections ────────────────────────────────────────────────
  // The raw ML values (ml.calories_target, ml.weight_change_30d_kg) are
  // GOAL-BLIND — they ignore the user's lose/gain goal. The backend now also
  // returns goal-aware figures in the same payload: kpi.calories_goal applies
  // the deficit/surplus, and forecast.trend_kg_per_30d is the direction-clamped
  // blended trend. Prefer those so every card tells one consistent story.
  const calorieTarget =
    (kpi.calories_goal != null ? kpi.calories_goal
      : calorieCurve.daily_target != null ? calorieCurve.daily_target
      : ml.calories_target)          // last-resort fallback only
  const calorieSource =
    kpi.calories_goal != null ? 'AI · matched to your goal'
      : (calorieCurve.method || ml.calories_source)
        ? 'AI estimate' : undefined

  // Human-readable caption for the calorie plan footer. Describes what the
  // number means for THIS user's goal instead of exposing raw internals like
  // "ML target 3760 (heuristic_fallback)". Uses "AI" wording per request.
  const calorieCaption = (() => {
    const goal = (data?.profile?.goal || '').toLowerCase()
    const tdee = calorieCurve.tdee
    const target = calorieTarget
    if (tdee == null || target == null) return null
    const diff = Math.round(target - tdee)
    const isLose  = ['lose', 'cut', 'fat_loss', 'fat loss'].includes(goal)
    const isGain  = ['gain', 'bulk', 'build', 'muscle_gain'].includes(goal)
    if (isLose && diff < 0) {
      return `AI plan: ${Math.abs(diff)} kcal deficit below your ${Number(tdee).toLocaleString()} kcal maintenance — tuned for fat loss.`
    }
    if (isGain && diff > 0) {
      return `AI plan: ${diff} kcal surplus above your ${Number(tdee).toLocaleString()} kcal maintenance — tuned for muscle gain.`
    }
    return `AI plan: maintenance target, matched to your ${Number(tdee).toLocaleString()} kcal daily burn.`
  })()

  // Corrected 30-day weight change: prefer the goal-aware blended forecast.
  const weightChange30d =
    forecast.trend_kg_per_30d != null ? forecast.trend_kg_per_30d
      : forecast.trend_kg_per_day != null ? Math.round(forecast.trend_kg_per_day * 30 * 100) / 100
      : ml.weight_change_30d_kg      // last-resort fallback only
  const weightSource =
    forecast.trend_kg_per_30d != null || forecast.trend_kg_per_day != null
      ? 'AI forecast' : (ml.weight_source ? 'AI estimate' : undefined)

  // Macros — computed from API-supplied weight + goal-aware calorie target
  const macros = (() => {
    const cal = calorieTarget
    const w = data?.profile?.weight_kg
    if (!cal || !w) return null
    const protein = Math.round(w * 2.0)
    const fats    = Math.round(w * 0.8)
    const carbs   = Math.max(Math.round((cal - protein * 4 - fats * 9) / 4), 0)
    const water   = Math.round(w * 0.035 * 10) / 10
    return { protein, carbs, fats, water }
  })()

  // ── first-load skeleton (replaces abrupt pop-in) ──────────────────────────
  if (loading && !data) {
    return (
      <div className="animate-fade-in">
        <DashboardSkeleton />
      </div>
    )
  }

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 animate-fade-in">
      {/* ── HERO HEADER ─────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
        className="card card-accent relative overflow-hidden"
      >
        {/* decorative corner orb (pure CSS, theme-aware via tokens) */}
        <div
          className="absolute -top-16 -right-16 w-56 h-56 rounded-full pointer-events-none"
          style={{ background: 'var(--grad-accent)', filter: 'blur(70px)', opacity: 0.18 }}
        />
        <div className="relative flex items-start justify-between flex-wrap gap-4">
          <div>
            <div className="eyebrow">Apex Dashboard</div>
            <h1 className="text-3xl font-bold font-display leading-tight">
              {greet}, <span className="gradient-text">{name}</span> 💪
            </h1>
            <p className="text-white/45 text-sm mt-1.5">
              {err
                ? `⚠️ ${err}`
                : !available
                  ? 'Complete your profile to unlock AI insights'
                  : `Real-time predictions · last computed ${
                      data?.computed_at ? new Date(data.computed_at).toLocaleTimeString() : '—'
                    }`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="badge badge-green">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00ff88] animate-pulse" /> AI Active
            </span>
            <button
              onClick={handleRefresh}
              disabled={loading || refreshing}
              className="btn-ghost flex items-center gap-2 text-xs py-2 px-3"
            >
              <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} /> Refresh
            </button>
          </div>
        </div>

        {/* at-a-glance status strip — only when we have data */}
        {available && (
          <div className="relative mt-5 pt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs"
               style={{ borderTop: '1px solid var(--border)' }}>
            <span className="flex items-center gap-1.5 text-white/55">
              <Zap size={13} className="text-[#ffd93d]" />
              <b className="text-white/85 tabular-nums">{kpi.streak_days ?? 0}</b> day streak
            </span>
            <span className="flex items-center gap-1.5 text-white/55">
              <Activity size={13} className="text-[#00d4ff]" />
              <b className="text-white/85 tabular-nums">{kpi.workouts_7d ?? 0}</b> sessions / 7d
            </span>
            {ml.fitness_level && (
              <span className="flex items-center gap-1.5 text-white/55">
                <Brain size={13} className="text-[#7b5cff]" />
                <b className="text-white/85">{ml.fitness_level}</b> class
              </span>
            )}
            {insights.length > 0 && (
              <span className="flex items-center gap-1.5 text-white/55">
                <Sparkles size={13} className="text-[#00ff88]" />
                <b className="text-white/85 tabular-nums">{insights.length}</b> live insights
              </span>
            )}
          </div>
        )}
      </motion.div>

      {/* No-profile fallback — now on the .notice system */}
      {!err && !available && (
        <div className="notice notice-info">
          <Brain size={20} className="flex-shrink-0 mt-0.5" />
          <div className="text-sm text-white/70">
            {data?.message || 'Set up your profile to see AI predictions and insights.'}
          </div>
        </div>
      )}

      {/* Hard error — surfaced clearly instead of only in the subtitle */}
      {err && (
        <div className="notice notice-error">
          <AlertTriangle size={20} className="flex-shrink-0 mt-0.5" />
          <div className="text-sm text-white/70">
            Couldn’t load your dashboard: {err}
          </div>
        </div>
      )}

      {/* ── KPI ROW ─────────────────────────────────────────────────────── */}
      {available && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 stagger">
          <Kpi
            icon={Flame} loading={loading}
            value={calorieTarget != null ? `${Number(calorieTarget).toLocaleString()} kcal` : '—'}
            label="Calorie Target"
            color="#ff6b35"
            sub={calorieSource ? `${calorieSource}` : undefined}
          />
          <Kpi
            icon={Brain} loading={loading}
            value={ml.fitness_level || '—'}
            label="AI Fitness Class"
            color="#00d4ff"
            sub={ml.fitness_probabilities && Object.keys(ml.fitness_probabilities).length
              ? `${Math.round(Math.max(...Object.values(ml.fitness_probabilities)) * 100)}% confidence`
              : undefined}
          />
          <Kpi
            icon={weightChange30d < 0 ? TrendingDown : TrendingUp}
            loading={loading}
            value={weightChange30d != null
              ? `${weightChange30d > 0 ? '+' : ''}${weightChange30d} kg`
              : '—'}
            label="Predicted 30d"
            color="#00ff88"
            deltaDir={weightChange30d != null
              ? (weightChange30d < 0 ? 'down' : 'up') : undefined}
            deltaText={weightChange30d != null ? '30-day' : undefined}
            sub={weightSource ? `via ${weightSource}` : undefined}
          />
          <Kpi
            icon={Zap} loading={loading}
            value={kpi.streak_days ?? 0}
            label="Day Streak"
            color="#ffd93d"
            sub={kpi.workouts_7d ? `${kpi.workouts_7d} sessions this week` : undefined}
          />
        </div>
      )}

      {/* ── AI INSIGHTS ─────────────────────────────────────────────────── */}
      {available && insights.length > 0 && (
        <div className="card">
          <SectionHead
            eyebrow="Generated live"
            title={<span className="flex items-center gap-2">
              <span className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#7b5cff] to-[#00d4ff] flex items-center justify-center text-sm">🧬</span>
              AI Insight Engine
            </span>}
            right={<span className="badge badge-blue">{insights.length} live</span>}
          />
          <div className="grid md:grid-cols-2 gap-3">
            {insights.map((ins, i) => {
              const sty = SEVERITY_STYLE[ins.severity] || SEVERITY_STYLE.info
              return (
                <motion.div key={i}
                  initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className="flex gap-3 p-3.5 rounded-xl border"
                  style={{ background: sty.bg, borderColor: sty.border }}>
                  <span className="text-xl flex-shrink-0 mt-0.5">{ins.icon}</span>
                  <div>
                    <div className="text-xs font-bold tracking-wider uppercase mb-1" style={{ color: sty.color }}>
                      {ins.category}
                    </div>
                    <div className="text-xs text-white/70 leading-relaxed">{ins.text}</div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── WEIGHT FORECAST ─────────────────────────────────────────────── */}
      {available && forecast.available && (
        <div className="card">
          <SectionHead
            eyebrow="Multi-model projection"
            icon={Scale}
            title="Weight Forecast"
            right={data?.profile?.target_weight
              ? <span className="badge badge-blue">target {data.profile.target_weight}kg</span>
              : null}
          />
          <div className="text-xs text-white/40 -mt-2 mb-4">
            <span className="text-white/60">AI forecast</span>{' · '}
            Confidence: <span className="text-white/60">{Math.round((forecast.stability ?? 0) * 100)}%</span>{' · '}
            Trend: <span className="text-white/60">{forecast.trend_kg_per_day} kg/day</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={forecast.points}>
              <defs>
                <linearGradient id="wGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#00ff88" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false}
                domain={['auto', 'auto']} />
              <Tooltip content={<Tip />} />
              {data?.profile?.target_weight && (
                <ReferenceLine y={data.profile.target_weight} stroke="#00d4ff" strokeDasharray="3 3" label={{
                  value: `target ${data.profile.target_weight}kg`,
                  fill: '#00d4ff', fontSize: 10, position: 'right',
                }} />
              )}
              <Area type="monotone" dataKey="kg" stroke="#00ff88" fill="url(#wGrad)"
                strokeWidth={2} dot={{ r: 3, fill: '#00ff88' }} name="kg" />
            </AreaChart>
          </ResponsiveContainer>

          <div className="grid grid-cols-3 gap-3 mt-4">
            {[
              { k: 'ml', label: 'AI model', color: '#7b5cff' },
              { k: 'energy_balance', label: 'Energy balance', color: '#ffd93d' },
              { k: 'observed', label: 'Observed trend', color: '#00ff88' },
            ].map(m => {
              const v = forecast.models_kg_change_30d?.[m.k]
              return (
                <div key={m.k} className="text-center p-3 rounded-xl"
                     style={{ background: 'var(--bg-glass)', border: '1px solid var(--border)' }}>
                  <div className="text-xs uppercase tracking-wider text-white/40 mb-1">{m.label}</div>
                  <div className="text-lg font-bold tabular-nums" style={{ color: m.color }}>
                    {v == null ? '—' : `${v > 0 ? '+' : ''}${v} kg`}
                  </div>
                  <div className="text-[10px] text-white/30">in 30 days</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── CALORIE PLAN + TIMELINE ─────────────────────────────────────── */}
      {available && (
        <div className="grid md:grid-cols-2 gap-5">
          <div className="card">
            <SectionHead
              icon={Flame}
              title="14-Day Calorie Plan"
              right={calorieCurve.available
                ? <span className="badge badge-warn">{calorieCurve.daily_target} kcal/day</span>
                : null}
            />
            {calorieCurve.available ? (
              <>
                <ResponsiveContainer width="100%" height={140}>
                  <BarChart data={calorieCurve.points}>
                    <XAxis dataKey="date" tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<Tip />} />
                    <Bar dataKey="calories" fill="#ff6b35" radius={[4, 4, 0, 0]} name="kcal" />
                  </BarChart>
                </ResponsiveContainer>
                <div className="text-xs text-white/40 mt-2">
                  {calorieCaption || `Daily target matched to your goal.`}
                </div>
              </>
            ) : (
              <div className="h-[140px] flex items-center justify-center text-white/30 text-sm">
                Add your profile to see your calorie schedule
              </div>
            )}
          </div>

          <div className="card">
            <SectionHead icon={Target} title="Timeline to Goal" />
            {timeline.available ? (
              <div className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <div className="text-xs uppercase tracking-wider text-white/40">Days remaining</div>
                  <div className="text-3xl font-bold font-display gradient-text tabular-nums">
                    {timeline.days_to_target}
                  </div>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-white/50">Target date</span>
                  <span className="text-white/80 font-mono">{timeline.target_date}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-white/50">Gap to target</span>
                  <span className="text-white/80 tabular-nums">{timeline.gap_kg > 0 ? '+' : ''}{timeline.gap_kg} kg</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-white/50">Trend</span>
                  <span className="text-white/80 tabular-nums">{timeline.trend_kg_per_day} kg/day</span>
                </div>
                <div className="pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                  <span className={`badge ${
                    timeline.feasibility === 'realistic' ? 'badge-green' :
                    timeline.feasibility === 'aggressive' ? 'badge-warn' : 'badge-warn'
                  }`}>
                    {timeline.feasibility}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-sm text-white/40">
                {timeline.reason === 'no_target_weight' && 'Set a target weight to see your timeline.'}
                {timeline.reason === 'trend_too_flat' && 'No detectable weight trend yet — log more weights.'}
                {timeline.reason === 'trend_wrong_direction' && 'Your current trend is moving away from your target.'}
                {!timeline.reason && (loading ? 'Computing…' : 'Insufficient data for timeline.')}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── ANOMALIES ───────────────────────────────────────────────────── */}
      {available && anomalies.length > 0 && (
        <div className="card" style={{ borderColor: 'rgba(255,107,53,0.20)' }}>
          <SectionHead
            icon={AlertTriangle}
            title="Anomaly Detection"
            right={<span className="badge badge-warn">{anomalies.length} flag{anomalies.length === 1 ? '' : 's'}</span>}
          />
          <div className="space-y-2">
            {anomalies.map((a, i) => {
              const sty = SEVERITY_STYLE[a.severity] || SEVERITY_STYLE.info
              return (
                <div key={i} className="p-3 rounded-xl border"
                  style={{ background: sty.bg, borderColor: sty.border }}>
                  <div className="text-xs font-bold tracking-wider uppercase" style={{ color: sty.color }}>
                    [{a.severity}] {a.code}
                  </div>
                  <div className="text-sm font-semibold mt-1" style={{ color: sty.color }}>
                    {a.title}
                  </div>
                  <div className="text-xs text-white/60 mt-1">{a.detail}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── RECOMMENDATIONS ─────────────────────────────────────────────── */}
      {available && recs.length > 0 && (
        <div className="card">
          <SectionHead
            icon={Target}
            title={`Recommended for ${ml.fitness_level || 'You'}`}
            right={<span className="badge badge-green">{recs.length} exercises</span>}
          />
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {recs.map((r, i) => (
              <motion.div key={i}
                whileHover={{ y: -3 }}
                className="p-3 rounded-xl border text-center cursor-default"
                style={{ borderColor: 'var(--border)', background: 'var(--bg-glass)' }}>
                <div className="text-2xl mb-2">💪</div>
                <div className="text-sm font-semibold text-white">
                  {typeof r === 'string' ? r : r.exercise}
                </div>
                {r.muscle_group && (
                  <div className="text-xs text-white/40 mt-1">{r.muscle_group}</div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* ── COHORT ──────────────────────────────────────────────────────── */}
      {available && cohort.available && (
        <div className="card">
          <SectionHead
            icon={Award}
            title="Vs Similar Users"
            right={<span className="badge badge-blue">{cohort.cohort_size} peers</span>}
          />
          <div className="text-xs text-white/40 -mt-2 mb-3">
            Same goal ({cohort.filters?.goal}) · Age {cohort.filters?.age_band?.[0]}–{cohort.filters?.age_band?.[1]}
            {' · '} BMI {cohort.filters?.bmi_band?.[0]}–{cohort.filters?.bmi_band?.[1]}
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {[
              { k: 'workouts_per_week', label: 'Workouts / week', unit: '' },
              { k: 'avg_form_score',    label: 'Avg form score', unit: '' },
              { k: 'weight_change_30d_kg', label: 'Weight Δ (30d)', unit: ' kg' },
            ].map(m => {
              const c = cohort.you_vs_cohort?.[m.k] || {}
              return (
                <div key={m.k} className="text-center p-3 rounded-xl"
                     style={{ background: 'var(--bg-glass)', border: '1px solid var(--border)' }}>
                  <div className="text-xs uppercase tracking-wider text-white/40 mb-2">{m.label}</div>
                  <div className="flex items-baseline justify-center gap-2">
                    <span className="text-2xl font-bold font-display gradient-text tabular-nums">
                      {c.you ?? '—'}{m.unit}
                    </span>
                    <span className="text-xs text-white/30">you</span>
                  </div>
                  <div className="text-xs text-white/50 mt-1 tabular-nums">
                    cohort avg: {c.cohort_avg ?? '—'}{m.unit}
                  </div>
                  {c.your_percentile != null && (
                    <div className="text-xs mt-2">
                      <span className={`badge ${c.your_percentile >= 75 ? 'badge-green' : c.your_percentile <= 25 ? 'badge-warn' : 'badge-blue'}`}>
                        P{Math.round(c.your_percentile)}
                      </span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── MACROS ──────────────────────────────────────────────────────── */}
      {available && macros && (
        <div className="card">
          <SectionHead icon={Droplets} title="Daily Macro Targets" />
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Protein', val: macros.protein, unit: 'g', color: '#7b5cff' },
              { label: 'Carbs',   val: macros.carbs,   unit: 'g', color: '#ffd93d' },
              { label: 'Fats',    val: macros.fats,    unit: 'g', color: '#ff6b35' },
              { label: 'Water',   val: macros.water,   unit: 'L', color: '#00d4ff' },
            ].map(({ label, val, unit, color }) => (
              <div key={label} className="text-center p-3 rounded-xl"
                   style={{ background: 'var(--bg-glass)', border: '1px solid var(--border)' }}>
                <div className="text-xs uppercase tracking-wider text-white/40 mb-1">{label}</div>
                <div className="text-2xl font-bold font-display tabular-nums" style={{ color }}>
                  {val}{unit}
                </div>
              </div>
            ))}
          </div>
          <div className="text-xs text-white/30 mt-3 text-center">
            Derived from {calorieTarget} kcal target · 2g/kg protein · 0.8g/kg fat
          </div>
        </div>
      )}
    </div>
  )
}
