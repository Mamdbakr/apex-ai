/**
 * Vision.jsx — CV Trainer using WebSocket for real-time streaming (30+ FPS)
 * Fixes:
 *   1. HTTP polling (5 FPS) → WebSocket (30+ FPS)
 *   2. Skeleton drawing now correctly maps 17 COCO landmarks every frame
 *   3. Shows phase, stage, angles, top-3 classifier, form cues
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { Video, VideoOff, Activity, Target, BarChart2, RefreshCw, AlertCircle, Zap } from 'lucide-react'

const BASE_HTTP = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) || 'http://localhost:8000'
const BASE_WS   = BASE_HTTP.replace(/^http/, 'ws')

const EXERCISES = ['squat', 'push_up', 'plank', 'barbell_biceps_curl', 'deadlift', 'shoulder_press']
const EX_LABELS = {
  squat:               '🏋️ Squat',
  push_up:             '💪 Push-Up',
  plank:               '⏱️ Plank',
  barbell_biceps_curl: '💪 Biceps Curl',
  deadlift:            '🔩 Deadlift',
  shoulder_press:      '🏋️ Shoulder Press',
}

// COCO-17 skeleton connections
const CONNECTIONS = [
  [5,6],[5,7],[7,9],[6,8],[8,10],   // shoulders + arms
  [5,11],[6,12],[11,12],            // torso
  [11,13],[13,15],[12,14],[14,16],  // legs
  [0,1],[0,2],[1,3],[2,4],          // face
]

function drawSkeleton(canvas, video, landmarks) {
  if (!canvas || !video) return
  const W = video.videoWidth  || 640
  const H = video.videoHeight || 480
  canvas.width  = W
  canvas.height = H
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, W, H)
  if (!landmarks || landmarks.length === 0) return

  ctx.lineWidth   = 2.5
  ctx.strokeStyle = '#00ff88'
  ctx.shadowColor = '#00ff88'
  ctx.shadowBlur  = 6

  CONNECTIONS.forEach(([a, b]) => {
    const pa = landmarks[a]
    const pb = landmarks[b]
    if (!pa || !pb) return
    if ((pa.visibility ?? 1) < 0.25 || (pb.visibility ?? 1) < 0.25) return
    ctx.beginPath()
    ctx.moveTo(pa.x * W, pa.y * H)
    ctx.lineTo(pb.x * W, pb.y * H)
    ctx.stroke()
  })

  ctx.fillStyle  = '#00d4ff'
  ctx.shadowColor = '#00d4ff'
  ctx.shadowBlur  = 8
  landmarks.forEach(lm => {
    if (!lm || (lm.visibility ?? 1) < 0.25) return
    ctx.beginPath()
    ctx.arc(lm.x * W, lm.y * H, 5, 0, Math.PI * 2)
    ctx.fill()
  })
  ctx.shadowBlur = 0
}

export default function Vision() {
  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)
  const streamRef   = useRef(null)   // MediaStream
  const wsRef       = useRef(null)   // WebSocket
  const sendLoopRef = useRef(null)   // setInterval for frame capture
  const sessionRef  = useRef(`cv-${Date.now()}`)
  const fpsCountRef = useRef(0)
  const lastFpsTs   = useRef(Date.now())

  const [active,    setActive]    = useState(false)
  const [exercise,  setExercise]  = useState('squat')
  const [reps,      setReps]      = useState(0)
  const [formScore, setFormScore] = useState(null)
  const [feedback,  setFeedback]  = useState([])
  const [fps,       setFps]       = useState(0)
  const [detected,  setDetected]  = useState(false)
  const [phase,     setPhase]     = useState('—')
  const [error,     setError]     = useState(null)
  const [top3,      setTop3]      = useState([])
  const [angles,    setAngles]    = useState({})
  const [wsStatus,  setWsStatus]  = useState('disconnected') // 'connecting'|'connected'|'disconnected'

  // ── WebSocket connection ────────────────────────────────────────────────

  const connectWS = useCallback((hint) => {
    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.close()
    }
    setWsStatus('connecting')
    const sid = sessionRef.current
    const url = `${BASE_WS}/vision/stream?sid=${sid}&exercise_hint=${hint}`
    const ws  = new WebSocket(url)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      setWsStatus('connected')
      setError(null)
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        if (data.type === 'control') return

        // FPS counter — backend heartbeat sends ui_fps field
        fpsCountRef.current++
        const now = Date.now()
        if (now - lastFpsTs.current >= 1000) {
          setFps(fpsCountRef.current)
          fpsCountRef.current = 0
          lastFpsTs.current   = now
        }
        // Override with backend-reported UI fps if available
        if (data.ui_fps) setFps(data.ui_fps)

        const poseOk = data.pose_detected ?? data.person_detected ?? data.detected ?? false
        setDetected(poseOk)

        if (poseOk) {
          setReps(data.rep_count ?? data.reps ?? 0)
          setFormScore(data.form_score ?? null)
          setFeedback(data.feedback_cues ?? data.form_cues ?? [])
          setPhase(data.phase ?? data.stage ?? '—')
          setTop3(data.top_3 ?? [])
          setAngles({
            primary: data.primary_angle,
            left:    data.left_angle,
            right:   data.right_angle,
          })
          // Draw skeleton
          if (data.landmarks?.length > 0) {
            drawSkeleton(canvasRef.current, videoRef.current, data.landmarks)
          } else {
            const ctx = canvasRef.current?.getContext('2d')
            if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
          }
        } else {
          const ctx = canvasRef.current?.getContext('2d')
          if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
          setFeedback(['No pose detected — step back so your full body is visible'])
        }
      } catch (_) {}
    }

    ws.onerror = () => {
      setError('WebSocket error — make sure the backend is running on port 8000.')
      setWsStatus('disconnected')
    }

    ws.onclose = (evt) => {
      setWsStatus('disconnected')
      if (evt.code === 4401) {
        setError('Not logged in — please log out and back in.')
        stopCamera()
      }
    }

    wsRef.current = ws
  }, [])

  // ── Frame capture loop → WebSocket ─────────────────────────────────────

  const startFrameLoop = useCallback(() => {
    const INTERVAL_MS = 33  // send at 30fps; YOLO runs async, UI heartbeat fills gaps
    sendLoopRef.current = setInterval(() => {
      const video  = videoRef.current
      const ws     = wsRef.current
      if (!video || !video.videoWidth || !ws || ws.readyState !== WebSocket.OPEN) return
      // Throttle if socket is backlogged
      if (ws.bufferedAmount > 50_000) return

      const tmp = document.createElement('canvas')
      // Downscale aggressively — YOLO works fine at 256px, saves huge CPU
      const scale = Math.min(1, 256 / video.videoHeight)
      tmp.width  = Math.round(video.videoWidth  * scale)
      tmp.height = Math.round(video.videoHeight * scale)
      tmp.getContext('2d').drawImage(video, 0, 0, tmp.width, tmp.height)
      tmp.toBlob(blob => {
        if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return
        blob.arrayBuffer().then(buf => {
          if (ws.readyState === WebSocket.OPEN) ws.send(buf)
        })
      }, 'image/jpeg', 0.50)
    }, INTERVAL_MS)
  }, [])

  // ── Camera start / stop ─────────────────────────────────────────────────

  async function startCamera() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      sessionRef.current = `cv-${Date.now()}`
      connectWS(exercise)
      startFrameLoop()
      setActive(true)
      setFeedback(['Camera active — stand 2–3 m back so your full body is visible'])
    } catch (e) {
      setError('Camera access denied. Please allow camera permissions and try again.')
    }
  }

  async function stopCamera() {
    clearInterval(sendLoopRef.current)
    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.close()
      wsRef.current = null
    }
    streamRef.current?.getTracks().forEach(t => t.stop())
    setActive(false)
    setDetected(false)
    setFps(0)
    setWsStatus('disconnected')
    setFeedback(['Session ended. Great work! 💪'])
    // Tell backend to finalize
    try {
      const params = new URLSearchParams({ session_id: sessionRef.current, sets: 1, duration_min: 5 })
      await fetch(`${BASE_HTTP}/vision/session/finish?${params}`, {
        method: 'POST', credentials: 'include',
      })
    } catch (_) {}
  }

  async function resetSession() {
    setReps(0); setFormScore(null); setFeedback([]); setPhase('—')
    try {
      await fetch(`${BASE_HTTP}/vision/reset?session_id=${sessionRef.current}`, {
        method: 'POST', credentials: 'include',
      })
    } catch (_) {}
    // Tell WS to reset too
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send('reset')
    }
  }

  function switchExercise(ex) {
    setExercise(ex)
    resetSession()
    if (active && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(`hint:${ex}`)
    }
  }

  // Cleanup on unmount
  useEffect(() => () => {
    clearInterval(sendLoopRef.current)
    wsRef.current?.close()
    streamRef.current?.getTracks().forEach(t => t.stop())
  }, [])

  const scoreColor = formScore >= 85 ? '#00ff88' : formScore >= 65 ? '#ffd93d' : '#ff6b35'

  const wsIndicator = {
    connecting:   { color: '#ffd93d', label: 'Connecting…' },
    connected:    { color: '#00ff88', label: 'WS Live' },
    disconnected: { color: '#ff6b35', label: 'Offline' },
  }[wsStatus]

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <div className="text-xs font-semibold text-[#00d4ff] tracking-widest uppercase mb-1">Camera Trainer</div>
        <h1 className="text-3xl font-bold font-display">AI personal <span className="gradient-text">Trainer</span></h1>
        <p className="text-white/40 text-sm mt-1">Good lighting helps · Pick your exercise first, then press Start Camera</p>
      </div>

      <div className="rounded-xl border border-[#00d4ff]/20 bg-[#00d4ff]/05 px-4 py-3 text-xs text-white/60 space-y-1">
        <p className="font-semibold text-[#00d4ff]">For best results:</p>
        <p>• Stand 2–3 metres from the camera so your full body (head to feet) is visible</p>
        <p>• Good lighting · Select your exercise first, then press Start</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-5">
        {/* Video + Canvas */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative rounded-2xl overflow-hidden bg-[#0b1628] border border-white/07 aspect-video">
            <video ref={videoRef} className="w-full h-full object-cover"
              muted playsInline style={{ transform: 'scaleX(-1)' }} />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ transform: 'scaleX(-1)' }} />

            {!active && !error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                <Video size={48} className="text-white/10 mb-4" />
                <p className="text-white/30 text-sm">Camera feed will appear here</p>
                <p className="text-white/20 text-xs mt-1">Real-Time Video • not connected</p>
              </div>
            )}

            {active && (
              <>
                {/* Top-left HUD */}
                <div className="absolute top-3 left-3 glass rounded-xl px-3 py-2 text-xs font-mono space-x-2 flex items-center">
                  <span className="font-bold" style={{ color: wsIndicator.color }}>
                    {fps} FPS
                  </span>
                  <span className="text-white/30">·</span>
                  <span className={detected ? 'text-[#00ff88]' : 'text-[#ff6b35]'}>
                    {detected ? '✓ Pose' : '✗ No pose'}
                  </span>
                  <span className="text-white/30">·</span>
                  <span style={{ color: wsIndicator.color }}>{wsIndicator.label}</span>
                </div>

                {/* Phase badge */}
                {detected && phase && phase !== '—' && (
                  <div className="absolute top-3 left-1/2 -translate-x-1/2 glass rounded-xl px-4 py-1.5 text-xs font-bold uppercase tracking-widest text-[#ffd93d]">
                    {phase}
                  </div>
                )}

                {/* Form score */}
                {formScore !== null && (
                  <div className="absolute top-3 right-3 glass rounded-xl px-3 py-2 text-xs text-center">
                    <div className="text-white/40 mb-1">Form</div>
                    <div className="text-lg font-bold font-display" style={{ color: scoreColor }}>{formScore}/100</div>
                  </div>
                )}

                {/* Angles overlay */}
                {detected && (angles.left || angles.right || angles.primary) && (
                  <div className="absolute bottom-14 right-3 glass rounded-xl px-3 py-2 text-xs space-y-1">
                    {angles.primary != null && <div className="text-white/50">Angle: <b className="text-[#00d4ff]">{Math.round(angles.primary)}°</b></div>}
                    {angles.left    != null && <div className="text-white/50">L: <b className="text-[#7b5cff]">{Math.round(angles.left)}°</b></div>}
                    {angles.right   != null && <div className="text-white/50">R: <b className="text-[#7b5cff]">{Math.round(angles.right)}°</b></div>}
                  </div>
                )}

                {/* Feedback cues */}
                {feedback.length > 0 && (
                  <div className="absolute bottom-3 left-3 right-3 glass rounded-xl px-3 py-2 text-xs text-white/70 space-y-0.5">
                    {feedback.slice(0, 2).map((f, i) => <p key={i}>{f}</p>)}
                  </div>
                )}
              </>
            )}

            {error && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="glass rounded-xl px-5 py-4 text-center max-w-xs">
                  <AlertCircle size={28} className="text-[#ff6b35] mx-auto mb-2" />
                  <p className="text-sm text-white/70">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex gap-3">
            {!active ? (
              <button onClick={startCamera} className="btn-primary flex items-center gap-2 flex-1 justify-center py-3">
                <Video size={16} /> Start Camera
              </button>
            ) : (
              <button onClick={stopCamera} className="btn-ghost flex items-center gap-2 flex-1 justify-center py-3 border-[#ff6b35]/30 text-[#ff6b35]">
                <VideoOff size={16} /> Stop Camera
              </button>
            )}
            <button onClick={resetSession} className="btn-ghost flex items-center gap-2 px-4">
              <RefreshCw size={15} /> Reset
            </button>
          </div>

          {/* Classifier top-3 */}
          {top3.length > 0 && (
            <div className="card">
              <h4 className="font-bold font-display text-sm mb-3 flex items-center gap-2">
                <Zap size={14} className="text-[#ffd93d]" /> Auto-Classifier Confidence
              </h4>
              <div className="space-y-2">
                {top3.map((item, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-white/60 capitalize">{item.exercise_name || item.exercise_id}</span>
                      <span className="font-semibold" style={{ color: i === 0 ? '#00ff88' : '#7a9cbf' }}>
                        {Math.round((item.confidence || 0) * 100)}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/05 overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.round((item.confidence || 0) * 100)}%`,
                          background: i === 0 ? '#00ff88' : '#7b5cff',
                        }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          {/* Exercise picker */}
          <div className="card">
            <h3 className="font-bold font-display text-sm mb-3">Exercise</h3>
            <div className="space-y-2">
              {EXERCISES.map(ex => (
                <button key={ex} onClick={() => switchExercise(ex)}
                  className={`w-full py-2.5 px-4 rounded-xl border text-sm font-semibold text-left transition-all ${exercise === ex ? 'border-[#00d4ff] bg-[#00d4ff]/10 text-[#00d4ff]' : 'border-white/08 text-white/40 hover:border-white/15'}`}>
                  {EX_LABELS[ex]}
                </button>
              ))}
            </div>
          </div>

          {/* Session stats */}
          <div className="card">
            <h3 className="font-bold font-display text-sm mb-3">Session Stats</h3>
            {[
              { label: 'Reps',     val: reps,                                             color: '#00ff88', icon: Activity },
              { label: 'Form',     val: formScore != null ? `${formScore}/100` : '—',      color: scoreColor || '#7a9cbf', icon: Target },
              { label: 'Phase',    val: phase || '—',                                     color: '#ffd93d', icon: Zap },
              { label: 'Exercise', val: EX_LABELS[exercise]?.replace(/^\S+\s/, '') ?? exercise, color: '#00d4ff', icon: BarChart2 },
            ].map(({ label, val, color, icon: Icon }) => (
              <div key={label} className="flex items-center justify-between py-3 border-b border-white/05 last:border-0">
                <div className="flex items-center gap-2 text-sm text-white/50">
                  <Icon size={14} style={{ color }} />
                  {label}
                </div>
                <div className="text-lg font-bold font-display" style={{ color }}>{val}</div>
              </div>
            ))}
          </div>

          {/* How it works */}
          <div className="card card-neon2">
            <h4 className="text-xs font-bold text-[#00d4ff] tracking-wider uppercase mb-2">How It Works</h4>
            <ul className="text-xs text-white/50 space-y-1.5">
              <li>• Your camera sends live video to the AI</li>
              <li>• The AI detects your body position in real time</li>
              <li>• Your joint angles are measured every frame</li>
              <li>• The AI recognises which exercise you're doing</li>
              <li>•Reps are counted automatically as you move</li>
              <li>• Your form is scored on balance and range of motion</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
