import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { User, Save, Shield, Activity, Target } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { userAPI } from '../lib/api'
import { formatNumber } from '../i18n/format'
import useStore from '../store/useStore'
import toast from 'react-hot-toast'

const ACTIVITY_LEVELS = [1, 2, 3, 4, 5]
const GOALS = ['lose', 'build', 'maintain']
const DIETARY = ['No Restrictions', 'Vegetarian', 'Vegan', 'Keto', 'Halal', 'Gluten-Free']
const TIMEFRAMES = ['1-3 months', '3-6 months', '6-12 months', '1+ years']

export default function Profile() {
  const { t } = useTranslation(['profile', 'common'])
  const { user, profile, setProfile } = useStore()
  const [form, setForm]     = useState({ name:'', age:25, weight_kg:70, height_cm:175, target_weight:65, activity_level:2, gender:1, goal:'lose', dietary_pref:'No Restrictions' })
  const [saving, setSaving] = useState(false)
  const [tab, setTab]       = useState('profile')
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  useEffect(() => {
    if (profile) setForm(f => ({ ...f, ...profile, gender: profile.gender === 'f' ? 0 : (profile.gender === 'm' ? 1 : profile.gender) }))
    else {
      userAPI.getProfile().then(({ data }) => {
        setForm(f => ({ ...f, ...data }))
        setProfile(data)
      }).catch(() => {})
    }
  }, [profile])

  async function handleSave() {
    setSaving(true)
    try {
      await userAPI.updateProfile(form)
      setProfile(form)
      toast.success(t('profile:saved'))
    } catch (e) { toast.error(t('profile:saveFailed')) }
    finally { setSaving(false) }
  }

  const bmi  = form.weight_kg && form.height_cm
    ? (form.weight_kg / ((form.height_cm / 100) ** 2)).toFixed(1) : '—'
  const bmr  = form.weight_kg && form.height_cm && form.age
    ? Math.round(10 * form.weight_kg + 6.25 * form.height_cm - 5 * form.age + (form.gender === 1 ? 5 : -161)) : '—'
  const tdee = bmr !== '—'
    ? Math.round(bmr * [1.2, 1.375, 1.55, 1.725, 1.9][form.activity_level - 1] || bmr * 1.375) : '—'

  const fmtOrDash = (v) => (typeof v === 'number' ? formatNumber(v) : v)

  const TABS = [
    { id: 'profile', icon: User,   label: t('profile:tabs.profile') },
    { id: 'goals',   icon: Target, label: t('profile:tabs.goals') },
  ]

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <div className="text-xs font-semibold text-[#7b5cff] tracking-widest uppercase mb-1">{t('profile:eyebrow')}</div>
        <h1 className="text-3xl font-bold font-display">{t('profile:title')} <span className="gradient-text">{t('profile:titleAccent')}</span></h1>
        <p className="text-white/40 text-sm mt-1">{t('profile:subtitle')}</p>
      </div>

      {/* User card */}
      <div className="card flex items-center gap-4">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#7b5cff] to-[#00d4ff] flex items-center justify-center text-2xl font-bold text-white flex-shrink-0">
          {(user?.name || form.name || 'U').charAt(0).toUpperCase()}
        </div>
        <div className="flex-1">
          <div className="font-bold text-lg font-display">{user?.name || form.name || t('common:athlete')}</div>
          <div className="text-white/40 text-sm">{user?.email || t('profile:noEmail')}</div>
          <div className="flex gap-2 mt-2">
            <span className="badge badge-green">{t('common:proMember')}</span>
            {form.goal && <span className="badge badge-purple">{t(`common:goals.${form.goal}`, { defaultValue: form.goal })}</span>}
          </div>
        </div>
        <div className="text-end hidden md:block">
          <div className="text-xs text-white/30 mb-1">{t('profile:computedBmi')}</div>
          <div className="text-2xl font-bold font-display text-[#00ff88]">{fmtOrDash(parseFloat(bmi) || bmi)}</div>
          <div className="text-xs text-white/30 mt-1">{t('profile:tdee', { value: fmtOrDash(tdee) })}</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        {TABS.map(tb => (
          <button key={tb.id} onClick={() => setTab(tb.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${tab === tb.id ? 'bg-[#00ff88]/10 border border-[#00ff88]/20 text-[#00ff88]' : 'text-white/40 hover:text-white/70'}`}>
            <tb.icon size={14} /> {tb.label}
          </button>
        ))}
      </div>

      {tab === 'profile' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card space-y-4">
          <h3 className="font-bold font-display mb-1">{t('profile:bodyStats')}</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('profile:fullName')}</label>
              <input className="input" value={form.name} onChange={e => set('name', e.target.value)} placeholder={t('profile:namePlaceholder')} />
            </div>
            <div>
              <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('profile:age')}</label>
              <input className="input" type="number" value={form.age} onChange={e => set('age', parseInt(e.target.value))} />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            {[[t('profile:weightKg'),'weight_kg'],[t('profile:heightCm'),'height_cm'],[t('profile:targetKg'),'target_weight']].map(([l,k]) => (
              <div key={k}>
                <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{l}</label>
                <input className="input" type="number" step="0.1" value={form[k]} onChange={e => set(k, parseFloat(e.target.value))} />
              </div>
            ))}
          </div>
          <div>
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('profile:gender')}</label>
            <div className="grid grid-cols-2 gap-3">
              {[[t('common:gender.male'), 1], [t('common:gender.female'), 0]].map(([l, v]) => (
                <button key={v} type="button" onClick={() => set('gender', v)}
                  className={`py-2 rounded-xl border text-sm font-semibold transition-all ${form.gender === v ? 'border-[#00ff88] bg-[#00ff88]/10 text-[#00ff88]' : 'border-white/10 text-white/40'}`}>
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('profile:activityLevel')}</label>
            <select className="select" value={form.activity_level} onChange={e => set('activity_level', parseInt(e.target.value))}>
              {ACTIVITY_LEVELS.map(lvl => <option key={lvl} value={lvl}>{t(`common:activities.${lvl}`)}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('profile:dietaryPreference')}</label>
            <select className="select" value={form.dietary_pref} onChange={e => set('dietary_pref', e.target.value)}>
              {DIETARY.map(d => <option key={d} value={d}>{t(`common:dietary.${d}`)}</option>)}
            </select>
          </div>

          {/* Live computed stats */}
          <div className="grid grid-cols-3 gap-3 pt-2">
            {[
              [t('profile:bmi'), fmtOrDash(parseFloat(bmi) || bmi), '#00ff88'],
              [t('profile:bmr'), t('profile:kcalValue', { value: fmtOrDash(bmr) }), '#00d4ff'],
              [t('profile:tdeeShort'), t('profile:kcalValue', { value: fmtOrDash(tdee) }), '#7b5cff'],
            ].map(([l,v,c]) => (
              <div key={l} className="rounded-xl p-3 text-center border border-white/05" style={{ background: c + '08' }}>
                <div className="text-lg font-bold font-display" style={{ color: c }}>{v}</div>
                <div className="text-xs text-white/30 mt-1">{l}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {tab === 'goals' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card space-y-4">
          <h3 className="font-bold font-display mb-1">{t('profile:trainingGoal')}</h3>
          {GOALS.map(g => (
            <button key={g} onClick={() => set('goal', g)} type="button"
              className={`w-full p-4 rounded-xl border text-start transition-all ${form.goal === g ? 'border-[#00ff88] bg-[#00ff88]/08' : 'border-white/08 hover:border-white/15'}`}>
              <div className="font-semibold text-sm" style={{ color: form.goal === g ? '#00ff88' : '#e0eeff' }}>{t(`common:goals.${g}`)}</div>
              <div className="text-xs text-white/40 mt-0.5">{t(`profile:goalDescs.${g}`)}</div>
            </button>
          ))}
          <div>
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-1.5 block">{t('profile:timeframe')}</label>
            <select className="select" value={form.timeframe || '3-6 months'} onChange={e => set('timeframe', e.target.value)}>
              {TIMEFRAMES.map(tf => <option key={tf} value={tf}>{t(`common:timeframes.${tf}`)}</option>)}
            </select>
          </div>
        </motion.div>
      )}

      <button onClick={handleSave} disabled={saving} className="btn-primary flex items-center gap-2 px-6 py-3">
        <Save size={16} /> {saving ? t('profile:saving') : t('profile:saveProfile')}
      </button>
    </div>
  )
}
