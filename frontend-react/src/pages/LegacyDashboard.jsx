import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Flame, Scale, Zap, Brain, Activity } from 'lucide-react'
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { userAPI } from '../lib/api'
import useStore from '../store/useStore'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-3 py-2 text-xs">
      <div className="text-white/50 mb-1">{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>{p.name}: <b>{p.value}</b></div>
      ))}
    </div>
  )
}

function KpiCard({ icon: Icon, value, label, sub, color, loading }) {
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="card text-center">
      <div className="w-10 h-10 rounded-xl mx-auto mb-3 flex items-center justify-center"
           style={{ background: color + '18', border: `1px solid ${color}30` }}>
        <Icon size={18} style={{ color }} />
      </div>
      {loading
        ? <div className="skeleton h-7 w-20 mx-auto mb-1" />
        : <div className="text-2xl font-bold font-display" style={{ color }}>{value}</div>
      }
      <div className="text-xs text-white/40 uppercase tracking-wider mt-1">{label}</div>
      {sub && <div className="text-xs mt-1.5" style={{ color: color + 'cc' }}>{sub}</div>}
    </motion.div>
  )
}

function InsightCard({ insight }) {
  return (
    <div className="flex gap-3 p-3.5 rounded-xl border transition-colors"
         style={{ background: insight.borderColor, borderColor: insight.borderColor }}>
      <span className="text-xl flex-shrink-0 mt-0.5">{insight.icon}</span>
      <div>
        <div className="text-xs font-bold tracking-wider uppercase mb-1" style={{ color: insight.color }}>
          {insight.category}
        </div>
        <div className="text-xs text-white/60 leading-relaxed">{insight.text}</div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { profile } = useStore()
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const hour = new Date().getHours()
  const greet = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'
  const name  = profile?.name?.split(' ')[0] || data?.profile?.name?.split(' ')[0] || 'Athlete'

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true); setError(null)
      try {
        const res = await userAPI.getDashboardFull()
        if (!cancelled) setData(res.data)
      } catch (e) {
        if (!cancelled) setError(e.response?.data?.detail || e.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  const kpi = data?.kpi || {}
  const insights = data?.insights || []
  const forecast = data?.forecast_30d
  const weightSeries = data?.weight_series || []
  const calorieSeries = data?.calorie_series || []
  const consistencyPct = Math.round((kpi.consistency || 0) * 100)
  const formPct = Math.round((kpi.avg_form_score || 0) * 100)

  // Macros: derive from calories goal using standard fitness ratios on the server-computed value
  const macros = (() => {
    const cal = kpi.calories_goal || 0
    const w = data?.profile?.weight_kg || 0
    if (!cal || !w) return null
    const protein = Math.round(w * 2.0)
    const fats    = Math.round(w * 0.8)
    const carbs   = Math.max(Math.round((cal - protein * 4 - fats * 9) / 4), 0)
    const water   = Math.round(w * 0.035 * 10) / 10
    return { protein, carbs, fats, water }
  })()

  // Goal-progress percentages — only show ones we can derive honestly
  const goalProgress = []
  if (kpi.workouts_30d != null) {
    goalProgress.push({
      label: 'Workout Consistency',
      val: consistencyPct,
      color: '#00ff88',
    })
  }
  if (kpi.avg_form_score) {
    goalProgress.push({
      label: 'Average Form Score',
      val: formPct,
      color: '#7b5cff',
    })
  }
  if (data?.profile?.weight_kg && data?.profile?.target_weight && weightSeries.length) {
    const startW   = weightSeries[0].weight
    const currentW = weightSeries[weightSeries.length - 1].weight
    const targetW  = data.profile.target_weight
    const totalGap = Math.abs(targetW - startW)
    const closed   = Math.abs(targetW - currentW)
    const progress = totalGap > 0 ? Math.max(0, Math.min(100, Math.round((1 - closed / totalGap) * 100))) : 0
    goalProgress.push({ label: 'Weight Progress', val: progress, color: '#00d4ff' })
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="text-xs font-semibold text-[#00ff88] tracking-widest uppercase mb-1">// Overview</div>
          <h1 className="text-3xl font-bold font-display">
            {greet}, <span className="gradient-text">{name}</span> 💪
          </h1>
          <p className="text-white/40 text-sm mt-1">
            {loading
              ? 'Loading your dashboard…'
              : error
                ? '⚠️ Could not reach dashboard API'
                : kpi.streak_days > 0
                  ? `${kpi.streak_days}-day streak · Consistency: ${consistencyPct}%`
                  : 'Log a workout to start your streak'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="badge badge-green">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00ff88] animate-pulse" /> AI Active
          </span>
        </div>
      </div>

      {error && (
        <div className="card border-[#ff6b35]/30 bg-[#ff6b35]/5">
          <div className="text-sm text-[#ff6b35]">Dashboard error: {error}</div>
        </div>
      )}

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard icon={Flame} loading={loading}
          value={kpi.calories_goal ? `${kpi.calories_goal.toLocaleString()} kcal` : '—'}
          label="Daily Target" color="#ff6b35"
          sub={data?.profile?.goal === 'lose' ? 'Cut −500' : (data?.profile?.goal === 'build' || data?.profile?.goal === 'gain') ? 'Bulk +300' : 'Maintenance'} />
        <KpiCard icon={Brain} loading={loading}
          value={kpi.fitness_level || '—'} label="Fitness Level" color="#00d4ff"
          sub={kpi.fitness_confidence ? `${Math.round(kpi.fitness_confidence * 100)}% confidence` : undefined} />
        <KpiCard icon={Scale} loading={loading}
          value={kpi.bmi ? kpi.bmi.toFixed(1) : '—'} label="BMI" color="#00ff88" />
        <KpiCard icon={Zap} loading={loading}
          value={kpi.streak_days ?? 0} label="Day Streak" color="#ffd93d"
          sub={kpi.workouts_7d ? `${kpi.workouts_7d} sessions this week` : undefined} />
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-5">
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold font-display">⚖️ Weight Trend</h3>
            {forecast?.available && (
              <span className="badge badge-blue">
                Forecast: {forecast.predicted_kg}kg in 30d
              </span>
            )}
          </div>
          {weightSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={weightSeries}>
                <defs>
                  <linearGradient id="wGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00ff88" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} domain={['auto','auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="weight" stroke="#00ff88" fill="url(#wGrad)" strokeWidth={2} dot={false} name="kg" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[160px] flex items-center justify-center text-white/30 text-sm">
              {loading ? 'Loading…' : 'Log your weight to see your trend'}
            </div>
          )}
        </div>

        <div className="card card-neon2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold font-display">🔥 Daily Calories (last 7d)</h3>
            {kpi.calories_goal > 0 && (
              <span className="badge badge-blue">target {kpi.calories_goal.toLocaleString()}</span>
            )}
          </div>
          {calorieSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={calorieSeries}>
                <XAxis dataKey="day" tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="calories" radius={[4, 4, 0, 0]} name="kcal">
                  {calorieSeries.map((_, i) => (
                    <Cell key={i} fill={'#00d4ff88'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[160px] flex items-center justify-center text-white/30 text-sm">
              {loading ? 'Loading…' : 'Log a meal to see your calorie history'}
            </div>
          )}
        </div>
      </div>

      {/* Insights */}
      {insights.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#7b5cff] to-[#00d4ff] flex items-center justify-center text-sm">🧬</div>
            <h3 className="font-bold font-display">AI Insight Engine</h3>
            <span className="badge badge-blue ml-auto">{insights.length} insights</span>
          </div>
          <div className="grid md:grid-cols-2 gap-3">
            {insights.map((ins, i) => <InsightCard key={i} insight={ins} />)}
          </div>
        </div>
      )}

      {/* Goal progress + macros */}
      <div className="grid md:grid-cols-3 gap-5">
        <div className="card md:col-span-2">
          <h3 className="font-bold font-display mb-4">📊 Goal Progress</h3>
          {goalProgress.length === 0 ? (
            <div className="text-sm text-white/40">Log workouts and weight to see progress.</div>
          ) : goalProgress.map(({ label, val, color }) => (
            <div key={label} className="mb-3">
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-white/50">{label}</span>
                <span className="font-semibold" style={{ color }}>{val}%</span>
              </div>
              <div className="progress-bar">
                <motion.div className="progress-fill" initial={{ width: 0 }} animate={{ width: `${val}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                  style={{ background: `linear-gradient(90deg, ${color}aa, ${color})` }} />
              </div>
            </div>
          ))}
        </div>

        <div className="card">
          <h3 className="font-bold font-display mb-4">🥗 Macro Targets</h3>
          {!macros ? (
            <div className="text-sm text-white/40">Complete your profile to see macros.</div>
          ) : [
            { label: 'Protein', val: macros.protein, unit: 'g', color: '#7b5cff' },
            { label: 'Carbs',   val: macros.carbs,   unit: 'g', color: '#ffd93d' },
            { label: 'Fats',    val: macros.fats,    unit: 'g', color: '#ff6b35' },
            { label: 'Water',   val: macros.water,   unit: 'L', color: '#00d4ff' },
          ].map(({ label, val, unit, color }) => (
            <div key={label} className="flex justify-between items-center py-2.5 border-b border-white/5 last:border-0">
              <span className="text-sm text-white/50">{label}</span>
              <span className="font-bold text-sm" style={{ color }}>{val}{unit}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Forecast */}
      {forecast?.available && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card card-neon2">
          <div className="flex items-center gap-3">
            <Activity size={20} className="text-[#00d4ff]" />
            <div>
              <div className="text-sm font-bold">Weight Forecast (30 days)</div>
              <div className="text-xs text-white/50 mt-0.5">
                Predicted: <span className="text-[#00d4ff] font-semibold">{forecast.predicted_kg}kg</span>
                · Trend: {forecast.trend_kg_per_30d > 0 ? '+' : ''}{forecast.trend_kg_per_30d}kg/30d
              </div>
            </div>
            <span className={`ml-auto badge ${forecast.trend_kg_per_30d < 0 ? 'badge-green' : forecast.trend_kg_per_30d > 0 ? 'badge-blue' : 'badge-warn'}`}>
              {forecast.trend_kg_per_30d < 0 ? '↓ Losing' : forecast.trend_kg_per_30d > 0 ? '↑ Gaining' : '→ Stable'}
            </span>
          </div>
        </motion.div>
      )}
    </div>
  )
}
