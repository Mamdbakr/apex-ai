import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, ArrowRight, ArrowLeft } from 'lucide-react'
import { authAPI } from '../lib/api'
import useStore from '../store/useStore'
import toast from 'react-hot-toast'

/**
 * Register.jsx — UI/UX-polished drop-in replacement.
 *
 * SIGNUP LOGIC UNCHANGED: same 3-step form, same `form` state keys, same
 * payload parsing, same authAPI.signup(payload), same setUser shape, same
 * navigation. Polish: cleaner step pills, segmented controls reuse the real
 * `.chip` token where possible, consistent spacing.
 */
const GOALS = ['lose', 'build', 'maintain']
const GOAL_LABELS = { lose: '🔥 Lose Fat', build: '💪 Build Muscle', maintain: '⚖️ Maintain' }
const ACTIVITIES = ['Sedentary 🛌', 'Lightly Active 🚶‍♂️', 'Moderately Active 💪', 'Very Active ⚡', 'Extremely Active 🔥🔥']

export default function Register() {
  const [step, setStep]   = useState(1)
  const [loading, setLoading] = useState(false)
  const [form, setForm]   = useState({
    full_name: '', email: '', password: '', gender: 'm',
    age: '', weight_kg: '', height_cm: '', target_weight: '',
    goal: 'lose', activity_level: 2, dietary_pref: 'No Restrictions', timeframe: '3-6 months',
  })
  const setUser = useStore((s) => s.setUser)
  const navigate    = useNavigate()
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  async function handleSubmit() {
    if (loading) return
    setLoading(true)
    try {
      const payload = {
        ...form,
        age: parseInt(form.age) || 25,
        weight_kg: parseFloat(form.weight_kg) || 70,
        height_cm: parseFloat(form.height_cm) || 170,
        target_weight: parseFloat(form.target_weight) || (parseFloat(form.weight_kg) || 70) - 5,
      }
      const { data } = await authAPI.signup(payload)

      // Cookie was set by the backend. Just hydrate the store cache.
      const userObj = {
        user_id: data.user_id,
        name:    data.name,
        email:   data.email,
        gender:  data.gender,
        role:    data.role,
      }
      setUser(userObj, data.profile || {})

      const firstName =
        (data.name && String(data.name).split(' ')[0]) ||
        (form.full_name && form.full_name.split(' ')[0]) ||
        'there'
      toast.success(`Welcome to APEX AI, ${firstName}!`)
      navigate('/dashboard', { replace: true })
    } catch (err) {
      toast.error(err.detail || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  // shared classes for the segmented "selected / not" buttons
  const segBase = 'py-2.5 rounded-xl border text-sm font-semibold transition-all'
  const segOn   = 'border-[#00ff88] bg-[#00ff88]/10 text-[#00ff88]'
  const segOff  = 'border-white/10 text-white/50 hover:border-white/25'

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12 relative"
         style={{ background: 'var(--bg-primary)' }}>
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="aurora w-[520px] h-[520px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
             style={{ background:'var(--accent3)', opacity:0.12 }} />
        <div className="aurora w-[280px] h-[280px] top-[10%] right-[8%]"
             style={{ background:'var(--accent2)', opacity:0.18 }} />
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md relative">
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2 font-bold text-2xl font-display mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00ff88] to-[#00d4ff] flex items-center justify-center"
                 style={{ boxShadow: '0 8px 24px var(--shadow-accent)' }}>
              <Zap size={18} className="text-[#060d1a]" />
            </div>
            APEX<span className="text-[#00ff88]">AI</span>
          </Link>
          <div className="flex items-center justify-center gap-2 mb-4">
            {[1,2,3].map(n => (
              <div key={n} className={`h-1.5 rounded-full transition-all duration-300 ${n <= step ? 'bg-[#00ff88] w-10' : 'bg-white/10 w-5'}`} />
            ))}
          </div>
          <h1 className="text-2xl font-bold font-display">
            {step === 1 ? 'Create account' : step === 2 ? 'Your body stats' : 'Your goals'}
          </h1>
          <p className="text-white/40 text-xs mt-1">Step {step} of 3</p>
        </div>

        <div className="card space-y-4">
          <AnimatePresence mode="wait">
            {step === 1 && (
              <motion.div key="s1" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="space-y-4">
                <div>
                  <label>Full Name</label>
                  <input placeholder="Ahmed Hassan" value={form.full_name} onChange={e => set('full_name', e.target.value)} />
                </div>
                <div>
                  <label>Email</label>
                  <input type="email" placeholder="you@example.com" value={form.email} onChange={e => set('email', e.target.value)} />
                </div>
                <div>
                  <label>Password</label>
                  <input type="password" placeholder="Min. 6 characters" value={form.password} onChange={e => set('password', e.target.value)} />
                </div>
                <div>
                  <label>Gender</label>
                  <div className="grid grid-cols-2 gap-3">
                    {['m', 'f'].map(g => (
                      <button key={g} type="button" onClick={() => set('gender', g)}
                        className={`${segBase} ${form.gender === g ? segOn : segOff}`}>
                        {g === 'm' ? '♂ Male' : '♀ Female'}
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
            {step === 2 && (
              <motion.div key="s2" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  {[['Age', 'age', '25'], ['Weight kg', 'weight_kg', '75'], ['Height cm', 'height_cm', '175']].map(([l, k, p]) => (
                    <div key={k}>
                      <label>{l}</label>
                      <input type="number" placeholder={p} value={form[k]} onChange={e => set(k, e.target.value)} />
                    </div>
                  ))}
                </div>
                <div>
                  <label>Target Weight (kg)</label>
                  <input type="number" placeholder="70" value={form.target_weight} onChange={e => set('target_weight', e.target.value)} />
                </div>
                <div>
                  <label>Activity Level</label>
                  <select value={form.activity_level} onChange={e => set('activity_level', parseInt(e.target.value))}>
                    {ACTIVITIES.map((a, i) => <option key={i} value={i + 1}>{a}</option>)}
                  </select>
                </div>
              </motion.div>
            )}
            {step === 3 && (
              <motion.div key="s3" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="space-y-4">
                <div>
                  <label>Primary Goal</label>
                  <div className="space-y-2">
                    {GOALS.map(g => (
                      <button key={g} type="button" onClick={() => set('goal', g)}
                        className={`w-full py-3 px-4 rounded-xl border text-sm font-semibold text-left transition-all ${form.goal === g ? segOn : 'border-white/10 text-white/60 hover:border-white/25'}`}>
                        {GOAL_LABELS[g]}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label>Dietary Preference</label>
                  <select value={form.dietary_pref} onChange={e => set('dietary_pref', e.target.value)}>
                    {['No Restrictions','Vegetarian','Vegan','Keto','Halal','Gluten-Free'].map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label>Timeframe</label>
                  <select value={form.timeframe} onChange={e => set('timeframe', e.target.value)}>
                    {['1-3 months','3-6 months','6-12 months','1+ years'].map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex gap-3 pt-2">
            {step > 1 && (
              <button onClick={() => setStep(s => s - 1)} className="btn-ghost flex items-center gap-1">
                <ArrowLeft size={15} /> Back
              </button>
            )}
            {step < 3 ? (
              <button onClick={() => setStep(s => s + 1)} className="btn-primary flex-1 flex items-center justify-center gap-1">
                Continue <ArrowRight size={15} />
              </button>
            ) : (
              <button onClick={handleSubmit} disabled={loading} className="btn-primary flex-1">
                {loading ? 'Creating account…' : '🚀 Launch APEX AI'}
              </button>
            )}
          </div>
          <p className="text-center text-sm text-white/50">
            Already have an account?{' '}
            <Link to="/login" className="text-[#00ff88] font-semibold link-underline">Sign In</Link>
          </p>
        </div>
      </motion.div>
    </div>
  )
}
