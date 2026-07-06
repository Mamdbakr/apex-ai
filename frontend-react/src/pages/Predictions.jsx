import { useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Flame, Scale, Target, ChevronDown } from 'lucide-react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'
import { useTranslation } from 'react-i18next'
import { predictAPI } from '../lib/api'
import { formatNumber } from '../i18n/format'
import useStore from '../store/useStore'
import toast from 'react-hot-toast'

const ACTIVITY_LEVELS = [1, 2, 3, 4, 5]
const GOALS = ['lose','build','maintain']

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-3 py-2 text-xs">
      <div className="text-white/50 mb-1">{label}</div>
      {payload.map((p, i) => <div key={i} style={{ color: p.color }}>{p.name}: <b>{p.value}</b></div>)}
    </div>
  )
}

function ResultCard({ label, value, sub, color, icon: Icon }) {
  return (
    <div className="card text-center">
      <div className="w-10 h-10 rounded-xl mx-auto mb-3 flex items-center justify-center"
           style={{ background: color + '18', border: `1px solid ${color}30` }}>
        {Icon && <Icon size={18} style={{ color }} />}
      </div>
      <div className="text-2xl font-bold font-display mb-1" style={{ color }}>{value}</div>
      <div className="text-xs text-white/40 uppercase tracking-wider">{label}</div>
      {sub && <div className="text-xs text-white/30 mt-1">{sub}</div>}
    </div>
  )
}

export default function Predictions() {
  const { t } = useTranslation(['predictions', 'common'])
  const { profile, setMLData } = useStore()
  const [form, setForm] = useState({
    age:            profile?.age            || 25,
    weight_kg:      profile?.weight_kg      || 70,
    height_cm:      profile?.height_cm      || 175,
    activity_level: profile?.activity_level || 2,
    gender:         profile?.gender === 'f' ? 0 : 1,
    goal:           profile?.goal           || 'maintain',
    duration_min:   45,
  })
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

 async function runPrediction() {
  setLoading(true)
  try {
    const { data } = await predictAPI.all(form)

    // Normalize nested backend response into flat shape the UI expects
    const cal    = data.calories      || {}
    const wt     = data.weight_change || {}
    const fit    = data.fitness       || {}

    const tdee         = cal.calories            || 0
    const weightChange = wt.weight_change_kg_30d || 0
    const h_m          = form.height_cm / 100
    const bmi          = form.weight_kg / (h_m * h_m)
    const protein_g    = Math.round(form.weight_kg * 2.0)
    const water_l      = (form.weight_kg * 0.033).toFixed(1)
    const goalCalories = form.goal === 'lose'
      ? tdee - 500
      : form.goal === 'build'
      ? tdee + 300
      : tdee

    const normalized = {
      calories_tdee:         tdee,
      goal_calories:         Math.round(goalCalories),
      bmi:                   parseFloat(bmi.toFixed(1)),
      // bmi_category holds a stable key; translated at render time
      bmi_category:          bmi < 18.5 ? 'underweight' : bmi < 25 ? 'normal' : bmi < 30 ? 'overweight' : 'obese',
      fitness_level:         fit.level_name         || null,
      fitness_level_id:      fit.level_id           ?? 1,
      classifier_confidence: fit.confidence         || 0.5,
      probabilities:         fit.probabilities      || null,
      weight_change_30d:     weightChange,
      new_weight_est:        parseFloat((form.weight_kg + weightChange).toFixed(1)),
      protein_g,
      water_l:               parseFloat(water_l),
      macros: {
        protein_g,
        carbs_g: Math.round(goalCalories * 0.45 / 4),
        fat_g:   Math.round(goalCalories * 0.25 / 9),
      },
    }

    setResult(normalized)
    setMLData(normalized)
    toast.success(t('predictions:predictionComplete'))
  } catch (error) {
    toast.error(t('predictions:predictionFailed', { error: error.message }))
  } finally {
    setLoading(false)
  }
  }

  const macroData = result ? [
    { name: t('predictions:macros.protein'), g: result.macros?.protein_g || result.protein_g, fill: '#7b5cff' },
    { name: t('predictions:macros.carbs'),   g: result.macros?.carbs_g,   fill: '#ffd93d' },
    { name: t('predictions:macros.fats'),    g: result.macros?.fat_g,      fill: '#ff6b35' },
  ] : []

  const radarData = result ? [
    { subject: t('predictions:radar.tdee'),      A: Math.min(100, Math.round((result.calories_tdee / 3500) * 100)) },
    { subject: t('predictions:radar.bmiScore'),  A: Math.max(0, 100 - Math.abs(result.bmi - 22) * 5) },
    { subject: t('predictions:radar.fitness'),   A: [25, 50, 75, 100][result.fitness_level_id] || 50 },
    { subject: t('predictions:radar.hydration'), A: Math.min(100, result.water_l * 30) },
    { subject: t('predictions:radar.protein'),   A: Math.min(100, Math.round((result.protein_g / (form.weight_kg * 2.2)) * 100)) },
    { subject: t('predictions:radar.goalAlign'), A: result.weight_change_30d && form.goal === 'lose' && result.weight_change_30d < 0 ? 85 : 60 },
  ] : []

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <div className="text-xs font-semibold text-[#7b5cff] tracking-widest uppercase mb-1">{t('predictions:eyebrow')}</div>
        <h1 className="text-3xl font-bold font-display">{t('predictions:title')} <span className="gradient-text">{t('predictions:titleAccent')}</span></h1>
      </div>

      <div className="grid lg:grid-cols-3 gap-5">
        {/* Input Form */}
        <div className="card lg:col-span-1 space-y-4">
          <h3 className="font-bold font-display mb-2">{t('predictions:yourStats')}</h3>

          {[
            [t('predictions:ageLabel'), 'age', 'number', '25'],
            [t('predictions:weightLabel'),  'weight_kg', 'number', '70'],
            [t('predictions:heightLabel'),  'height_cm', 'number', '175'],
          ].map(([label, key, type, placeholder]) => (
            <div key={key}>
              <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{label}</label>
              <input className="input" type={type} placeholder={placeholder} value={form[key]}
                onChange={e => set(key, parseFloat(e.target.value) || 0)} />
            </div>
          ))}

          <div>
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('predictions:gender')}</label>
            <div className="grid grid-cols-2 gap-2">
              {[[t('common:gender.male'), 1], [t('common:gender.female'), 0]].map(([l, v]) => (
                <button key={v} onClick={() => set('gender', v)} type="button"
                  className={`py-2 rounded-xl border text-sm font-semibold transition-all ${form.gender === v ? 'border-[#00ff88] bg-[#00ff88]/10 text-[#00ff88]' : 'border-white/10 text-white/40'}`}>
                  {l}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('predictions:activityLevel')}</label>
            <select className="select" value={form.activity_level} onChange={e => set('activity_level', parseInt(e.target.value))}>
              {ACTIVITY_LEVELS.map(lvl => <option key={lvl} value={lvl}>{t(`common:activities.${lvl}`)}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('predictions:goal')}</label>
            <div className="space-y-1.5">
              {GOALS.map(g => (
                <button key={g} onClick={() => set('goal', g)} type="button"
                  className={`w-full py-2 px-3 rounded-xl border text-sm font-semibold text-start transition-all ${form.goal === g ? 'border-[#00ff88] bg-[#00ff88]/10 text-[#00ff88]' : 'border-white/10 text-white/40'}`}>
                  {t(`common:goals.${g}`)}
                </button>
              ))}
            </div>
          </div>

          <button onClick={runPrediction} disabled={loading} className="btn-primary w-full py-3 mt-2">
            {loading ? t('predictions:running') : t('predictions:run')}
          </button>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-5">
          {!result ? (
            <div className="card h-full flex items-center justify-center text-center py-20">
              <div>
                <Brain size={48} className="text-white/10 mx-auto mb-4" />
                <p className="text-white/30 text-sm">{t('predictions:emptyState')}</p>
              </div>
            </div>
          ) : (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
              {/* KPI results */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <ResultCard label={t('predictions:tdee')} value={`${formatNumber(result.calories_tdee)} ${t('common:units.kcal')}`}
                  color="#ff6b35" icon={Flame} sub={t('predictions:tdeeSub')} />
                <ResultCard label={t('predictions:bmi')} value={formatNumber(result.bmi)}
                  color="#00ff88" icon={Scale} sub={t(`predictions:bmiCategories.${result.bmi_category}`)} />
                <ResultCard label={t('predictions:fitnessLevel')} value={result.fitness_level || t('predictions:unknown')}
                  color="#00d4ff" icon={Brain}
                  sub={t('predictions:confidenceSub', { percent: formatNumber(Math.round((result.classifier_confidence || 0.5) * 100)) })} />
                <ResultCard label={t('predictions:goalCalories')} value={formatNumber(result.goal_calories)}
                  color="#7b5cff" icon={Target}
                  sub={form.goal === 'lose' ? t('predictions:cut') : form.goal === 'build' ? t('predictions:bulk') : t('predictions:maintenance')} />
              </div>

              {/* Confidence probabilities */}
              {result.probabilities && (
                <div className="card">
                  <h4 className="font-bold font-display text-sm mb-3">{t('predictions:classifierConfidence')}</h4>
                  <div className="space-y-2">
                    {Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).map(([label, prob]) => (
                      <div key={label}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/60">{label}</span>
                          <span className="text-[#00ff88] font-semibold">{formatNumber(Math.round(prob * 100))}%</span>
                        </div>
                        <div className="progress-bar">
                          <motion.div className="progress-fill" initial={{ width: 0 }}
                            animate={{ width: `${prob * 100}%` }} transition={{ duration: 0.6, delay: 0.1 }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Macro chart + Radar */}
              <div className="grid md:grid-cols-2 gap-5">
                <div className="card">
                  <h4 className="font-bold font-display text-sm mb-3">{t('predictions:macroTargets')}</h4>
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={macroData}>
                      <XAxis dataKey="name" tick={{ fill: '#7a9cbf', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: '#7a9cbf', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="g" radius={[6, 6, 0, 0]} name={t('predictions:grams')}>
                        {macroData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="card">
                  <h4 className="font-bold font-display text-sm mb-3">{t('predictions:healthRadar')}</h4>
                  <ResponsiveContainer width="100%" height={160}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.06)" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#7a9cbf', fontSize: 10 }} />
                      <Radar name={t('predictions:score')} dataKey="A" stroke="#00ff88" fill="#00ff88" fillOpacity={0.15} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Weight projection */}
              <div className="card card-neon2">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  {[
                    { label: t('predictions:currentWeight'), val: `${formatNumber(form.weight_kg)} ${t('common:units.kg')}`, color: '#e0eeff' },
                    { label: t('predictions:projection30d'), val: `${formatNumber(result.new_weight_est)} ${t('common:units.kg')}`, color: '#00d4ff' },
                    { label: t('predictions:proteinPerDay'), val: `${formatNumber(result.protein_g)} ${t('common:units.g')}`, color: '#7b5cff' },
                    { label: t('predictions:waterPerDay'),   val: `${formatNumber(result.water_l)} ${t('common:units.l')}`, color: '#00ff88' },
                  ].map(({ label, val, color }) => (
                    <div key={label}>
                      <div className="text-xl font-bold font-display" style={{ color }}>{val}</div>
                      <div className="text-xs text-white/40 mt-1">{label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
