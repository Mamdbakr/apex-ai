import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus, Plus, Scale } from 'lucide-react'
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis,
         Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'
import { userAPI, dataAPI } from '../lib/api'
import useStore from '../store/useStore'
import toast from 'react-hot-toast'

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-3 py-2 text-xs">
      <div className="text-white/40 mb-1">{label}</div>
      {payload.map((p, i) => <div key={i} style={{ color: p.color }}>{p.name}: <b>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</b></div>)}
    </div>
  )
}

export default function Progress() {
  const { user, profile, intelligence } = useStore()
  const userId = user?.user_id
  const [workouts,  setWorkouts]   = useState([])
  const [weightHist,setWeightHist] = useState([])
  const [logWeight, setLogWeight]  = useState('')
  const [loading,   setLoading]    = useState(true)
  const [logging,   setLogging]    = useState(false)

  const fv       = intelligence?.feature_vector || {}
  const forecast = intelligence?.forecast_30d   || {}

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const [wkRes, whRes] = await Promise.all([
          userAPI.getWorkouts(),
          userAPI.getWeightHistory(),
        ])
        setWorkouts(wkRes.data || [])
        setWeightHist(whRes.data || [])
      } catch (e) { console.warn(e.message) }
      finally { setLoading(false) }
    }
    load()
  }, [userId])

  async function handleLogWeight() {
    if (!logWeight || isNaN(parseFloat(logWeight))) return
    setLogging(true)
    try {
      const uid = userId || 1
      await dataAPI.weight({ user_id: uid, weight_kg: parseFloat(logWeight) })
      toast.success(`Weight ${logWeight}kg logged! 📊`)
      setLogWeight('')
      const { data } = await userAPI.getWeightHistory()
      setWeightHist(data || [])
    } catch (e) { toast.error('Failed to log weight') }
    finally { setLogging(false) }
  }

  // Prepare chart data
  const weightData = weightHist.slice(-14).map(w => ({
    date: new Date(w.logged_at).toLocaleDateString('en', { month: 'short', day: 'numeric' }),
    weight: parseFloat(w.weight_kg.toFixed(1)),
  }))

  // Add forecast point
  if (forecast.available && weightData.length > 0) {
    weightData.push({ date: 'Forecast', weight: forecast.predicted_kg, forecast: true })
  }

  const workoutByDay = (() => {
    const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    const counts = Array(7).fill(0)
    workouts.forEach(w => { counts[(new Date(w.logged_at).getDay() + 6) % 7]++ })
    return days.map((d, i) => ({ day: d, sessions: counts[i] }))
  })()

  const formHistory = workouts.slice(0, 20).reverse().map((w, i) => ({
    i: i + 1,
    form: Math.round((w.form_score || 1) * 100),
  }))

  const trend = fv.weight_trend_30d || 0
  const TrendIcon = trend < -0.1 ? TrendingDown : trend > 0.1 ? TrendingUp : Minus
  const trendColor = trend < -0.1 ? '#00ff88' : trend > 0.1 ? '#00d4ff' : '#7a9cbf'

  const stats = [
    { label: 'Total Workouts',    val: fv.total_workouts || workouts.length,       color: '#00ff88' },
    { label: 'Sessions / Month',  val: fv.workouts_30d   || 0,                     color: '#00d4ff' },
    { label: 'Avg Duration',      val: `${Math.round(fv.avg_duration || 0)} min`,  color: '#7b5cff' },
    { label: 'Avg Form Score',    val: `${Math.round((fv.avg_form_score || 1) * 100)}%`, color: '#ffd93d' },
    { label: 'Current Streak',    val: `${fv.streak_days || 0} days`,             color: '#ff6b35' },
    { label: 'Consistency',       val: `${Math.round((fv.consistency_score || 0) * 100)}%`, color: '#00ff88' },
  ]

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="text-xs font-semibold text-[#00ff88] tracking-widest uppercase mb-1">Progress Tracking</div>
          <h1 className="text-3xl font-bold font-display">Your <span className="gradient-text">Progress</span></h1>
          <p className="text-white/40 text-sm mt-1">Time-series analytics · Behavioral patterns · Weight forecasting</p>
        </div>
        {/* Log weight inline */}
        <div className="flex items-center gap-2">
          <input className="input w-28 text-sm" type="number" step="0.1" placeholder="kg"
            value={logWeight} onChange={e => setLogWeight(e.target.value)} />
          <button onClick={handleLogWeight} disabled={logging || !logWeight}
            className="btn-primary flex items-center gap-1.5 text-sm px-4 py-2.5">
            <Scale size={14} /> {logging ? 'Logging…' : 'Log Weight'}
          </button>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {stats.map(({ label, val, color }) => (
          <div key={label} className="card text-center py-4">
            <div className="text-xl font-bold font-display mb-1" style={{ color }}>{val}</div>
            <div className="text-xs text-white/40">{label}</div>
          </div>
        ))}
      </div>

      {/* Weight chart */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold font-display">⚖️ Weight History</h3>
          <div className="flex items-center gap-2">
            <TrendIcon size={16} style={{ color: trendColor }} />
            <span className="text-sm font-semibold" style={{ color: trendColor }}>
              {trend >= 0 ? '+' : ''}{trend.toFixed(2)} kg / 30d
            </span>
            {forecast.available && (
              <span className="badge badge-blue ml-2">→ {forecast.predicted_kg}kg forecast</span>
            )}
          </div>
        </div>
        {weightData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={weightData}>
              <defs>
                <linearGradient id="wGrad2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#00ff88" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#00ff88" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} domain={['auto','auto']} />
              <Tooltip content={<Tip />} />
              <Area type="monotone" dataKey="weight" stroke="#00ff88" fill="url(#wGrad2)"
                strokeWidth={2} dot={d => d.payload.forecast ?
                  <circle cx={d.cx} cy={d.cy} r={5} fill="#00d4ff" stroke="#00d4ff" strokeWidth={2} /> : null}
                name="Weight (kg)" />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-40 text-white/20 text-sm">
            Log your weight above to see your trend
          </div>
        )}
      </div>

      {/* Workout frequency + Form trend */}
      <div className="grid md:grid-cols-2 gap-5">
        <div className="card">
          <h3 className="font-bold font-display text-sm mb-4">📅 Workout Frequency</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={workoutByDay}>
              <XAxis dataKey="day" tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip content={<Tip />} />
              <Bar dataKey="sessions" radius={[4, 4, 0, 0]} name="Sessions">
                {workoutByDay.map((_, i) => (
                  <Cell key={i} fill={_.sessions > 0 ? '#00ff88' : 'rgba(255,255,255,0.04)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="font-bold font-display text-sm mb-4">🎯 Form Score Trend</h3>
          {formHistory.length > 0 ? (
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={formHistory}>
                <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
                <XAxis dataKey="i" tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: '#7a9cbf', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip content={<Tip />} />
                <Line type="monotone" dataKey="form" stroke="#7b5cff" strokeWidth={2}
                  dot={{ fill: '#7b5cff', r: 3 }} name="Form %" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-40 text-white/20 text-sm">
              Log workouts to see form trend
            </div>
          )}
          {fv.form_improvement > 0 && (
            <div className="text-xs text-[#00ff88] text-center mt-2">
              ↑ {fv.form_improvement?.toFixed(1)}% improvement over your history
            </div>
          )}
        </div>
      </div>

      {/* Recent workouts table */}
      {workouts.length > 0 && (
        <div className="card">
          <h3 className="font-bold font-display text-sm mb-4">📋 Recent Sessions</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-white/30 uppercase tracking-wider border-b border-white/05">
                  {['Exercise','Sets','Reps','Weight','Duration','Form','Date'].map(h => (
                    <th key={h} className="text-left py-2 pr-4 font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {workouts.slice(0, 8).map((w, i) => (
                  <tr key={i} className="border-b border-white/03 hover:bg-white/02 transition-colors">
                    <td className="py-2.5 pr-4 font-medium text-white/80">{w.exercise}</td>
                    <td className="py-2.5 pr-4 text-white/50">{w.sets}</td>
                    <td className="py-2.5 pr-4 text-white/50">{w.reps}</td>
                    <td className="py-2.5 pr-4 text-white/50">{w.weight_kg > 0 ? `${w.weight_kg}kg` : '—'}</td>
                    <td className="py-2.5 pr-4 text-white/50">{w.duration_min}min</td>
                    <td className="py-2.5 pr-4">
                      <span style={{ color: w.form_score >= 0.8 ? '#00ff88' : w.form_score >= 0.6 ? '#ffd93d' : '#ff6b35' }}>
                        {Math.round(w.form_score * 100)}%
                      </span>
                    </td>
                    <td className="py-2.5 text-white/30 text-xs">
                      {new Date(w.logged_at).toLocaleDateString('en', { month: 'short', day: 'numeric' })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
