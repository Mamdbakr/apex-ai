import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Zap, User, RotateCcw } from 'lucide-react'
import { chatAPI } from '../lib/api'
import useStore from '../store/useStore'

/**
 * Chatbot.jsx — UI/UX-polished drop-in replacement.
 *
 * CHAT LOGIC UNCHANGED: same chatAPI.send payload (message, history, user_data
 * with identical keys + gender mapping), same store actions, same scroll-to-
 * bottom behaviour, same markdown-lite rendering.
 *
 * Fixes a silent bug: the original used `typing-dot` and `border-white/07`,
 * neither of which exist in index.css — so the typing animation and several
 * borders rendered as nothing. Replaced with real tokens + an inline-animated
 * typing indicator.
 */
const SUGGESTIONS = [
  'How many calories should I eat today?',
  'Why am I not losing weight?',
  'Give me a workout plan for my goal',
  'What should I eat post-workout?',
  'How do I break a plateau?',
]

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-sm font-bold
        ${isUser ? 'bg-gradient-to-br from-[#ff6b35] to-[#ffd93d] text-[#060d1a]'
                 : 'bg-gradient-to-br from-[#00ff88] to-[#00d4ff] text-[#060d1a]'}`}>
        {isUser ? <User size={14} /> : <Zap size={14} />}
      </div>
      <div className={`max-w-[75%] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
        <div
          className={`chat-bubble ${isUser ? 'user' : 'assistant'}`}
          style={{ maxWidth: '100%' }}
          dangerouslySetInnerHTML={{ __html: msg.content
            .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
            .replace(/\n/g, '<br />') }}
        />
        <span className="text-xs text-white/25 px-1">
          {isUser ? 'You' : `APEX Coach${msg.model ? ` · ${msg.model}` : ''}`}
        </span>
      </div>
    </motion.div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#00ff88] to-[#00d4ff] flex items-center justify-center">
        <Zap size={14} className="text-[#060d1a]" />
      </div>
      <div className="chat-bubble assistant flex items-center gap-1.5" style={{ padding: '0.85rem 1.1rem' }}>
        {[0,1,2].map(i => (
          <motion.span
            key={i}
            className="w-2 h-2 rounded-full"
            style={{ background: 'var(--accent)' }}
            animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
            transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.15 }}
          />
        ))}
      </div>
    </div>
  )
}

export default function Chatbot() {
  const { user, profile, intelligence, chatHistory, addMessage, clearChat } = useStore()
  const userId = user?.user_id
  const [input, setInput]   = useState('')
  const [typing, setTyping] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatHistory, typing])

  async function sendMessage(text = input) {
    if (!text.trim()) return
    const userMsg = { role: 'user', content: text.trim() }
    addMessage(userMsg)
    setInput('')
    setTyping(true)

    try {
      const uid = userId || 1
      const { data } = await chatAPI.send({
        message:   text.trim(),
        history:   chatHistory.slice(-10),
        user_data: {
          name:           profile?.name,
          age:            profile?.age,
          weight_kg:      profile?.weight_kg,
          height_cm:      profile?.height_cm,
          goal:           profile?.goal,
          activity_level: profile?.activity_level,
          gender:         profile?.gender === 'f' ? 0 : 1,
          target_weight:  profile?.target_weight,
        },
      })
      addMessage({ role: 'assistant', content: data.reply, model: data.model, source: data.source })
    } catch (e) {
      addMessage({ role: 'assistant', content: "I'm having a moment — try again! 💪", model: 'fallback' })
    } finally { setTyping(false) }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="eyebrow">AI Coach</div>
          <h1 className="text-2xl font-bold font-display">APEX Coach <span className="gradient-text">Chat</span></h1>
          <p className="text-white/40 text-xs mt-0.5">Knows your goals, your stats, and your history — every answer is made for you</p>
        </div>
        <button onClick={clearChat} className="btn-ghost flex items-center gap-2 text-xs py-2 px-3">
          <RotateCcw size={13} /> New Chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {chatHistory.length === 0 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-12">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#00ff88] to-[#00d4ff] flex items-center justify-center text-3xl mx-auto mb-4 glow-pulse">🧠</div>
            <h3 className="font-bold font-display text-lg mb-2">APEX Coach is ready</h3>
            <p className="text-white/40 text-sm mb-6 max-w-sm mx-auto">
              I know your weight, your goal, and your workout history. Ask me anything about training or nutrition.
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {SUGGESTIONS.map(s => (
                <button key={s} onClick={() => sendMessage(s)} className="chip text-xs">
                  {s}
                </button>
              ))}
            </div>
          </motion.div>
        )}
        <AnimatePresence>
          {chatHistory.map((msg, i) => <Message key={i} msg={msg} />)}
          {typing && <TypingIndicator key="typing" />}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="pt-4" style={{ borderTop: '1px solid var(--border)' }}>
        <form onSubmit={e => { e.preventDefault(); sendMessage() }} className="flex gap-3">
          <input
            className="flex-1"
            placeholder="Ask your AI coach anything…"
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={typing}
          />
          <button type="submit" disabled={!input.trim() || typing} className="btn-primary px-4 flex items-center gap-2">
            <Send size={16} />
          </button>
        </form>
        <p className="text-xs text-white/20 text-center mt-2">
          Powered by advanced AI — your data stays private and secure
        </p>
      </div>
    </div>
  )
}
