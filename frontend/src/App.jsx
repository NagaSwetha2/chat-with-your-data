import { useState, useRef, useEffect } from 'react'
import * as XLSX from 'xlsx'

const FAQS = [
  {
    q: "How do I get started?",
    a: "Upload a CSV or .txt file using the sidebar. Once indexed, type your question in the chat box and hit Send.",
  },
  {
    q: "What kind of questions can I ask?",
    a: "Anything about your data — 'show me the top deals', 'which records are negative?', 'summarize Q3 performance', 'find happy customers', 'what is the average revenue?'",
  },
  {
    q: "Can I ask in plain English?",
    a: "Yes. Zarva understands natural language. You don't need to write SQL or code — just ask like you would ask a colleague.",
  },
  {
    q: "Can I ask Salesforce-style queries?",
    a: "Yes. You can type SOQL like 'SELECT Name, Amount FROM Opportunity WHERE StageName = Closed Won' and Zarva will interpret it.",
  },
  {
    q: "How does sentiment search work?",
    a: "When you upload your data, Zarva automatically tags each record as positive, negative, or neutral. Ask 'show me happy records' or 'find negative feedback' to filter by sentiment.",
  },
  {
    q: "Does Zarva remember my previous questions?",
    a: "Yes, within the same session. Zarva keeps track of your last 6 questions so you can ask follow-ups like 'show me more' or 'now filter those by date'.",
  },
  {
    q: "What file formats are supported?",
    a: "CSV and plain text (.txt) files. CSV works best for structured data like Salesforce exports, sales reports, or customer lists.",
  },
  {
    q: "How many questions can I ask?",
    a: "Thousands per day. There is no hard limit per session — ask as many questions as you need about your data.",
  },
]
import './App.css'

const API_BASE = ''

function MdText({ text, onOptionClick }) {
  if (!text) return null

  const renderInline = (t) => {
    const parts = t.split(/(\*\*[^*]+\*\*|`[^`]+`)/)
    if (parts.length === 1) return t
    return parts.map((p, i) => {
      if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>
      if (p.startsWith('`') && p.endsWith('`')) return <code key={i} className="md-code">{p.slice(1, -1)}</code>
      return p
    })
  }

  const lines = text.split('\n')
  const elements = []
  let bullets = []
  let tableLines = []

  const flushBullets = () => {
    if (!bullets.length) return
    elements.push(
      <ul key={`ul-${elements.length}`} className="md-list">
        {bullets.map((b, i) => onOptionClick ? (
          <li key={i} className="md-option" onClick={() => onOptionClick(b)} role="button">{renderInline(b)}</li>
        ) : (
          <li key={i}>{renderInline(b)}</li>
        ))}
      </ul>
    )
    bullets = []
  }

  const flushTable = () => {
    if (tableLines.length < 2) { tableLines = []; return }
    const headers = tableLines[0].split('|').filter(s => s.trim()).map(s => s.trim())
    const dataRows = tableLines.slice(2).filter(r => !/^[\|\s\-:]+$/.test(r))
    const rows = dataRows.map(row => row.split('|').filter(s => s.trim()).map(s => s.trim()))
    elements.push(
      <div key={`tbl-${elements.length}`} className="md-table-wrap">
        <table className="md-table">
          <thead><tr>{headers.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
          <tbody>{rows.map((row, ri) => (
            <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{renderInline(cell)}</td>)}</tr>
          ))}</tbody>
        </table>
      </div>
    )
    tableLines = []
  }

  lines.forEach((line, idx) => {
    const t = line.trim()
    if (!t) { flushBullets(); flushTable(); return }

    if (t.startsWith('|')) {
      flushBullets(); tableLines.push(t); return
    } else if (tableLines.length) { flushTable() }

    if (t.startsWith('- ') || t.startsWith('• ') || t.startsWith('* ')) {
      flushTable(); bullets.push(t.slice(2)); return
    }

    flushBullets(); flushTable()
    if (t.startsWith('### ')) elements.push(<h4 key={idx} className="md-h4">{renderInline(t.slice(4))}</h4>)
    else if (t.startsWith('## ')) elements.push(<h3 key={idx} className="md-h3">{renderInline(t.slice(3))}</h3>)
    else if (t.startsWith('# ')) elements.push(<h2 key={idx} className="md-h2">{renderInline(t.slice(2))}</h2>)
    else elements.push(<p key={idx} className="md-p">{renderInline(t)}</p>)
  })
  flushBullets(); flushTable()
  return <div className="md-body">{elements}</div>
}

function HypCard({ h, onSelect }) {
  const [evidenceOpen, setEvidenceOpen] = useState(false)
  return (
    <div className={`hyp-card impact-${h.impact}`} onClick={() => onSelect && onSelect(h)} style={{cursor: onSelect ? 'pointer' : 'default'}}>
      <div className="hyp-card-top">
        <span className={`hyp-impact-badge ${h.impact}`}>{h.impact === 'high' ? '⚠ High' : '● Med'}</span>
        <span className="hyp-category">{h.category}</span>
      </div>
      <h3 className="hyp-card-title">{h.title}</h3>
      <p className="hyp-narrative">{h.narrative}</p>
      <button className="hyp-evidence-toggle" onClick={() => setEvidenceOpen(o => !o)}>
        {evidenceOpen ? '▾' : '▸'} Evidence · {h.evidence.length} data points
      </button>
      {evidenceOpen && (
        <div className="hyp-evidence">
          {h.evidence.map((e, j) => (
            <div key={j} className="hyp-evidence-row">
              <span className="hyp-evidence-dot" /><span>{e}</span>
            </div>
          ))}
        </div>
      )}
      <div className="hyp-action">
        <span className="hyp-action-label">→ Action</span>
        <span className="hyp-action-text">{h.action}</span>
      </div>
    </div>
  )
}

const CAT_ICONS = {
  'Revenue':'💰','Sales':'📈','Risk':'⚠️','Performance':'🏆',
  'Customer':'👥','Customers':'👥','Forecast':'🔮','Pipeline':'🔄',
  'Analysis':'🔬','Operations':'⚙️','Finance':'💳','HR':'👤',
  'Data Quality':'✅','Trends':'📉','Compensation':'💵','Payroll':'💵',
  'Deductions':'📋','Tax':'🏛️','Benefits':'🏥',
}

const TYPE_META = {
  anomaly:        { icon: '⚠️', label: 'Anomaly',    color: '#ef4444' },
  trend:          { icon: '📈', label: 'Trend',       color: '#3b82f6' },
  comparison:     { icon: '⚖️', label: 'Comparison', color: '#a78bfa' },
  prediction:     { icon: '🔮', label: 'Prediction',  color: '#06b6d4' },
  risk:           { icon: '🛡️', label: 'Risk',        color: '#f59e0b' },
  recommendation: { icon: '💡', label: 'Action',      color: '#22c55e' },
}

const SEV_META = {
  critical: { badge: 'CRITICAL', bg: '#450a0a', fg: '#f87171' },
  warning:  { badge: 'WARNING',  bg: '#1c1208', fg: '#fbbf24' },
  positive: { badge: 'POSITIVE', bg: '#052e16', fg: '#4ade80' },
  info:     { badge: 'INSIGHT',  bg: '#0c1a2e', fg: '#60a5fa' },
}

function InsightCard({ f, onSelect }) {
  const meta = TYPE_META[f.type] || TYPE_META.recommendation
  const sev  = SEV_META[f.severity] || SEV_META.info
  const dirIcon = f.direction === 'up' ? '↑' : f.direction === 'down' ? '↓' : '→'

  return (
    <div className={`insight-card sev-${f.severity}`} style={{'--accent': meta.color, cursor: 'pointer'}} onClick={() => onSelect && onSelect(f)}>
      <div className="insight-card-top">
        <span className="insight-type-badge" style={{background: `${meta.color}18`, color: meta.color}}>
          {meta.icon} {meta.label}
        </span>
        <span className="insight-sev-badge" style={{background: sev.bg, color: sev.fg}}>
          {sev.badge}
        </span>
      </div>
      <h3 className="insight-title">{f.title}</h3>
      <div className="insight-metrics">
        {f.value && <span className="insight-value" style={{color: meta.color}}>{dirIcon} {f.value}</span>}
        {f.delta && <span className="insight-delta">{f.delta}</span>}
      </div>
      {f.evidence && (
        <p className="insight-evidence">{f.evidence}</p>
      )}
      {f.action && (
        <div className="insight-action">
          <span className="insight-action-label">→ Action</span>
          <span className="insight-action-text">{f.action}</span>
        </div>
      )}
    </div>
  )
}

function NarrativePanel({ data }) {
  if (!data?.narrative?.headline) return null
  const { narrative, domain } = data
  return (
    <div className="narrative-panel">
      <div className="narrative-domain-badge">{domain?.label || 'Analysis'}</div>
      <h2 className="narrative-headline">{narrative.headline}</h2>
      <div className="narrative-body">
        {narrative.body.split('\n\n').map((para, i) => (
          <p key={i}>{para}</p>
        ))}
      </div>
      {narrative.key_numbers?.length > 0 && (
        <div className="narrative-numbers">
          {narrative.key_numbers.map((n, i) => (
            <span key={i} className="narrative-number-chip">{n}</span>
          ))}
        </div>
      )}
      {narrative.next_action && (
        <div className="narrative-action">
          <span className="narrative-action-label">→ Next Action</span>
          <span className="narrative-action-text">{narrative.next_action}</span>
        </div>
      )}
    </div>
  )
}

function InsightsPanel({ data, onSelect }) {
  const [filter, setFilter] = useState('all')
  if (!data || !data.findings || data.findings.length === 0) return null

  const types = ['all', ...new Set(data.findings.map(f => f.type))]
  const filtered = filter === 'all' ? data.findings : data.findings.filter(f => f.type === filter)

  return (
    <section className="insights-panel">
      <div className="insights-header">
        <div className="insights-header-left">
          <h2 className="insights-title">🔍 AI Analysis</h2>
          <p className="insights-summary">{data.summary}</p>
        </div>
        <div className="insights-counts">
          {data.critical_count > 0 && (
            <span className="insights-count critical">{data.critical_count} critical</span>
          )}
          {data.warning_count > 0 && (
            <span className="insights-count warning">{data.warning_count} warnings</span>
          )}
          {data.opportunity_count > 0 && (
            <span className="insights-count positive">{data.opportunity_count} opportunities</span>
          )}
        </div>
      </div>

      <div className="insights-filters">
        {types.map(t => (
          <button
            key={t}
            className={`insights-filter-btn ${filter === t ? 'active' : ''}`}
            onClick={() => setFilter(t)}
            style={filter === t && t !== 'all'
              ? {borderColor: TYPE_META[t]?.color, color: TYPE_META[t]?.color, background: `${TYPE_META[t]?.color}12`}
              : {}}
          >
            {t === 'all' ? 'All' : `${TYPE_META[t]?.icon || ''} ${TYPE_META[t]?.label || t}`}
          </button>
        ))}
      </div>

      <div className="insights-grid">
        {filtered.map((f, i) => <InsightCard key={i} f={f} onSelect={onSelect} />)}
      </div>
    </section>
  )
}

function Sparkline({ values = [], color = '#6366f1', width = 110, height = 36 }) {
  if (values.length < 2) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 6) - 3
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

function IotPanel({ data, onAction }) {
  if (!data || !data.latest) return null
  const { status, readings, alerts, business_impact, device_id, latest } = data

  const temps = readings.map(r => r.temperature).filter(v => v != null)
  const vibs  = readings.map(r => r.vibration).filter(v => v != null)
  const pows  = readings.map(r => r.power).filter(v => v != null)

  const statusColor = status === 'critical' ? '#ef4444' : status === 'warning' ? '#f59e0b' : '#22c55e'
  const statusLabel = status === 'critical' ? '🔴 CRITICAL' : status === 'warning' ? '🟡 WARNING' : '🟢 NORMAL'

  const tempColor = (v) => v > 90 ? '#ef4444' : v > 75 ? '#f59e0b' : '#22c55e'
  const vibColor  = (v) => v > 2   ? '#f59e0b' : '#6366f1'
  const powColor  = (v) => v > 2000 ? '#f59e0b' : '#6366f1'

  return (
    <section className="iot-panel">
      <div className="iot-panel-header">
        <div>
          <h2 className="iot-panel-title">🔌 IoT Monitor — {device_id}</h2>
          <p className="iot-panel-sub">Raspberry Pi · Live sensor telemetry</p>
        </div>
        <span className="iot-status-badge" style={{ background: `${statusColor}22`, color: statusColor, borderColor: `${statusColor}55` }}>
          {statusLabel}
        </span>
      </div>

      <div className="iot-readings-grid">
        {latest.temperature != null && (
          <div className={`iot-reading-card ${latest.temperature > 90 ? 'crit' : latest.temperature > 75 ? 'warn' : ''}`}>
            <div className="iot-reading-top">
              <span className="iot-reading-icon">🌡️</span>
              <span className="iot-reading-val" style={{ color: tempColor(latest.temperature) }}>{latest.temperature}°C</span>
            </div>
            <span className="iot-reading-label">Temperature</span>
            <Sparkline values={temps} color={tempColor(latest.temperature)} />
          </div>
        )}
        {latest.vibration != null && (
          <div className={`iot-reading-card ${latest.vibration > 2 ? 'warn' : ''}`}>
            <div className="iot-reading-top">
              <span className="iot-reading-icon">📳</span>
              <span className="iot-reading-val" style={{ color: vibColor(latest.vibration) }}>{latest.vibration} g</span>
            </div>
            <span className="iot-reading-label">Vibration</span>
            <Sparkline values={vibs} color={vibColor(latest.vibration)} />
          </div>
        )}
        {latest.power != null && (
          <div className={`iot-reading-card ${latest.power > 2000 ? 'warn' : ''}`}>
            <div className="iot-reading-top">
              <span className="iot-reading-icon">⚡</span>
              <span className="iot-reading-val" style={{ color: powColor(latest.power) }}>{latest.power} W</span>
            </div>
            <span className="iot-reading-label">Power Draw</span>
            <Sparkline values={pows} color={powColor(latest.power)} />
          </div>
        )}
      </div>

      {alerts && alerts.length > 0 && (
        <div className="iot-alerts">
          {alerts.map((a, i) => (
            <div key={i} className={`iot-alert ${a.severity}`}>
              <span className="iot-alert-icon">{a.severity === 'critical' ? '🚨' : '⚠️'}</span>
              <span className="iot-alert-msg">{a.message}</span>
            </div>
          ))}
        </div>
      )}

      {business_impact && (
        <div className="iot-impact">
          <div className="iot-impact-header">
            <h3 className="iot-impact-title">Business Impact</h3>
            <span className={`iot-impact-sev ${business_impact.severity}`}>{business_impact.severity?.toUpperCase()}</span>
          </div>
          <p className="iot-impact-prod">{business_impact.production_impact}</p>
          {business_impact.estimated_downtime && (
            <p className="iot-impact-time">⏱ {business_impact.estimated_downtime}</p>
          )}
          {business_impact.at_risk_customers?.length > 0 && (
            <div className="iot-impact-customers">
              <span className="iot-impact-label">At-risk accounts:</span>
              {business_impact.at_risk_customers.map((c, i) => (
                <span key={i} className="iot-customer-chip">{c}</span>
              ))}
            </div>
          )}
          <div className="iot-impact-actions">
            <div className="iot-impact-steps">
              {business_impact.recommended_actions?.map((act, i) => (
                <div key={i} className="iot-step"><span className="iot-step-num">{i + 1}</span>{act}</div>
              ))}
            </div>
            <div className="iot-action-btns">
              <button className="iot-action-btn primary" onClick={() => onAction('maintenance', business_impact)}>
                🔧 Create Ticket
              </button>
              <button className="iot-action-btn" onClick={() => onAction('email', business_impact)}>
                ✉️ Draft Email
              </button>
              <button className="iot-action-btn" onClick={() => onAction('notify', business_impact)}>
                🔔 Notify Supervisor
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

function FAQ() {
  const [open, setOpen] = useState(null)
  return (
    <section className="faq">
      <h2>How to use Zarva</h2>
      <div className="faq-grid">
        {FAQS.map((item, i) => (
          <div key={i} className={`faq-card ${open === i ? 'open' : ''}`} onClick={() => setOpen(open === i ? null : i)}>
            <div className="faq-question">
              <span>{item.q}</span>
              <span className="faq-icon">{open === i ? '−' : '+'}</span>
            </div>
            {open === i && <p className="faq-answer">{item.a}</p>}
          </div>
        ))}
      </div>
    </section>
  )
}

function App() {
  const [indexId, setIndexId] = useState(null)
  const [recordCount, setRecordCount] = useState(null)
  const [indexing, setIndexing] = useState(false)
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [suggestions, setSuggestions] = useState({})
  const [openCategories, setOpenCategories] = useState({})
  const [kpis, setKpis] = useState(null)
  const [ml, setMl] = useState(null)
  const [mlTab, setMlTab] = useState('risk')
  const [hypotheses, setHypotheses] = useState(null)
  const [eda, setEda] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [kpisLoading, setKpisLoading] = useState(false)
  const [selectedInsight, setSelectedInsight] = useState(null)
  const [mode, setMode] = useState('data')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, asking])

  function toggleCategory(cat) {
    setOpenCategories(prev => ({ ...prev, [cat]: !prev[cat] }))
  }

  function exportToExcel() {
    const rows = []
    for (let i = 0; i < messages.length - 1; i++) {
      if (messages[i].role === 'user' && messages[i+1].role === 'assistant') {
        rows.push({ Question: messages[i].content, Answer: messages[i+1].content })
      }
    }
    const ws = XLSX.utils.json_to_sheet(rows)
    ws['!cols'] = [{ wch: 40 }, { wch: 80 }]
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Zarva Insights')
    XLSX.writeFile(wb, 'zarva-insights.xlsx')
  }

  async function handleFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return

    setIndexing(true)
    setMessages([])
    setSuggestions({})
    setOpenCategories({})
    setKpis(null)
    setMl(null)
    setHypotheses(null)
    setEda(null)
    setAnalysis(null)
    setAnalysisLoading(false)
    setKpisLoading(false)
    setIndexId(null)
    setRecordCount(null)
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch(`${API_BASE}/index`, { method: 'POST', body: formData })
    const data = await res.json()
    const followups = data.followups || []
    const id = data.index_id

    // Poll suggestions immediately in parallel — they're ready before indexing finishes
    ;(async () => {
      for (let attempt = 0; attempt < 30; attempt++) {
        await new Promise(r => setTimeout(r, 2000))
        try {
          const sd = await fetch(`${API_BASE}/suggestions/${id}`).then(r => r.json())
          const cats = sd.suggestions || {}
          if (Object.keys(cats).length > 0) {
            setSuggestions(cats)
            setOpenCategories(Object.fromEntries(Object.keys(cats).map((k, i) => [k, i === 0])))
            return
          }
        } catch {}
      }
    })()

    // Poll until ready
    while (true) {
      await new Promise(r => setTimeout(r, 2000))
      const s = await fetch(`${API_BASE}/status/${id}`).then(r => r.json())
      if (s.status === 'ready') {
        setIndexId(id)
        setRecordCount(s.record_count)
        setIndexing(false)
        // Auto-summarize the uploaded file
        fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index_id: id, question: 'What is this document or data about? Give me a 3-bullet summary of what you see and what I can ask you.' }),
        }).then(r => r.json()).then(d => {
          if (d.answer) setMessages([{ role: 'assistant', content: d.answer, followups: d.followups || [] }])
        }).catch(() => {})
        setKpisLoading(true)
        fetch(`${API_BASE}/kpis/${id}`).then(r => r.json()).then(d => { setKpis(d); setKpisLoading(false) }).catch(() => setKpisLoading(false))
        fetch(`${API_BASE}/ml/${id}`).then(r => r.json()).then(d => setMl(d)).catch(() => {})
        fetch(`${API_BASE}/hypotheses/${id}`).then(r => r.json()).then(d => setHypotheses(d)).catch(() => {})
        fetch(`${API_BASE}/eda/${id}`).then(r => r.json()).then(d => { if (!d.error) setEda(d) }).catch(() => {})
        setAnalysisLoading(true)
        fetch(`${API_BASE}/analysis/${id}`).then(r => r.json()).then(d => { if (!d.error && d.total > 0) setAnalysis(d); setAnalysisLoading(false) }).catch(() => setAnalysisLoading(false))
        break
      }
      if (s.status === 'error') {
        setIndexing(false)
        alert('Indexing failed. Please try again.')
        break
      }
    }
  }

  async function handleAsk(e) {
    e.preventDefault()
    if (!question.trim() || !indexId || asking) return

    const userMsg = { role: 'user', content: question }
    setMessages((prev) => [...prev, userMsg])
    setQuestion('')
    setAsking(true)

    const msgIndex = Date.now()

    // Salesforce mode: non-streaming
    if (mode === 'sf') {
      try {
        const res = await fetch(`${API_BASE}/sf-chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: userMsg.content }),
        })
        const data = await res.json()
        setMessages(prev => [...prev, { role: 'assistant', content: data.answer || 'No response.', inlineCharts: null, followups: data.followups || [], _id: msgIndex }])
      } catch {
        setMessages(prev => [...prev, { role: 'assistant', content: 'Request failed. Please try again.', _id: msgIndex }])
      }
      setAsking(false)
      return
    }

    // Data mode: streaming
    let fullAnswer = ''
    let followups = []
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), 45000)
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index_id: indexId, question: userMsg.content }),
        signal: ctrl.signal,
      })
      clearTimeout(timer)
      if (res.status === 404) {
        fullAnswer = 'Session expired — please re-upload your file to start a new session.'
      } else {
        const json = await res.json()
        fullAnswer = json.answer || 'I had trouble responding. Please try again.'
        followups = json.followups || []
      }
    } catch (err) {
      clearTimeout(timer)
      fullAnswer = err?.name === 'AbortError' ? 'Zarva took too long — please try again.' : 'Request failed. Please try again.'
    }

    setMessages(prev => [...prev, { role: 'assistant', content: fullAnswer, inlineCharts: null, followups, _id: msgIndex }])
    setAsking(false)

    const data = { answer: fullAnswer, followups }

    // Generate chart in background if requested
    const q = userMsg.content.toLowerCase()
    // "pi" alone = "pie" (autocorrect), also match plural/typo forms
    const normalizeQ = (s) => s.replace(/\bpi\b/g, 'pie').replace(/\bchar\b/g, 'chart').replace(/\bvisualise\b/g, 'visualize')
    const nq = normalizeQ(q)
    const chartWords = ['chart', 'pie', 'graph', 'bar', 'visual', 'draw', 'plot', 'visualize', 'visually', 'diagram', 'histogram', 'trend', 'scatter', 'bubble', 'candlestick', 'candle', 'ohlc', 'waterfall', 'radar', 'spider', 'violin', 'funnel', 'heatmap', 'heat map', 'area chart', 'breakdown', 'distribution of', 'correlation', 'versus', 'vs ', 'by month', 'by week', 'by year', 'by quarter', 'by day', 'over time', 'by stage', 'by owner', 'by category', 'by status']
    const answerMentionsChart = ['bar chart', 'pie chart', 'line chart', 'chart visualiz', 'here\'s the chart', 'here is the chart', 'visualiz'].some(w => data.answer.toLowerCase().includes(w))
    const wantsChart = chartWords.some(w => nq.includes(w)) || answerMentionsChart
    if (wantsChart && indexId) {
      const lastMessages = messages.slice(-4)
      const context = lastMessages.map(m => (m.role === 'user' ? 'Q: ' : 'A: ') + m.content).join('\n')
      // Always pass the answer so the chart engine can extract pre-computed data
      const answerContext = data.answer ? `\nThe assistant already computed this data:\n${data.answer}` : ''
      const chartQuestion = userMsg.content + answerContext + (context ? '\nContext from previous conversation:\n' + context : '')
      try {
        const ctrl = new AbortController()
        const timer = setTimeout(() => ctrl.abort(), 25000)
        const cr = await fetch(`${API_BASE}/dynamic-chart/${indexId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: chartQuestion }),
          signal: ctrl.signal,
        })
        clearTimeout(timer)
        if (cr.ok) {
          const result = await cr.json()
          if (result.chart) {
            // Attach chart to the assistant message we already showed
            setMessages(prev => prev.map(m =>
              m._id === msgIndex ? { ...m, inlineCharts: { chart: result.chart } } : m
            ))
          }
        }
      } catch {}
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Zarva</h1>
        <p>Upload any file — CSV, PDF, image — and ask questions about it.</p>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <h2>1. Upload your file</h2>
          <input type="file" accept=".csv,.txt,.pdf,.jpg,.jpeg,.jfif,.jpe,.png,.webp,.gif,.bmp,.dib,.tiff,.tif,.heic,.heif,.avif,.ico,.ppm,.pgm,.pbm,.pnm,.tga,.pcx,.svg" onChange={handleFileUpload} />
          {indexing && (
            <div className="indexing-progress">
              <div className="indexing-bar-wrap"><div className="indexing-bar-fill" /></div>
              <p className="status">Indexing &amp; generating prompts…</p>
            </div>
          )}
          {!indexing && recordCount !== null && (
            <>
              <div className="sidebar-stats">
                <span className="stat-num">{recordCount.toLocaleString()}</span>
                <span className="stat-label">{recordCount === 1 ? 'page indexed' : recordCount <= 200 ? 'pages indexed' : 'records indexed'}</span>
              </div>
              <div className="tree-prompts">
                <p className="tree-title">Ask Zarva</p>
                {Object.entries(suggestions).map(([cat, prompts]) => (
                  <div key={cat} className="tree-parent">
                    <button className={`tree-parent-row ${openCategories[cat] ? 'open' : ''}`} onClick={() => toggleCategory(cat)}>
                      <span className="tree-cat-icon">{CAT_ICONS[cat] || '📂'}</span>
                      <span className="tree-parent-label">{cat}</span>
                      <span className="tree-chevron">{openCategories[cat] ? '▾' : '▸'}</span>
                    </button>
                    {openCategories[cat] && Array.isArray(prompts) && (
                      <div className="tree-children">
                        {prompts.slice(0, 3).map((s, i) => (
                          <button key={i} className="tree-child-row" onClick={() => {
                            setQuestion(s)
                            setTimeout(() => document.getElementById('send-btn').click(), 50)
                          }}>
                            <span className="tree-child-label">{s}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </aside>

        <main className="chat">
          <h2>2. Ask questions about your data</h2>

          {mode === 'sf' && <p className="hint sf-hint">Ask any Salesforce or SOQL question — no upload needed.</p>}
          {mode === 'data' && !indexId && <p className="hint">Upload a file using the sidebar to get started.</p>}

          <div className="messages">
            {messages.map((msg, i) => (
              msg.role === 'user' ? (
                <div key={i} className="msg-user-wrap">
                  <div className="msg-user">{msg.content}</div>
                </div>
              ) : (
                <div key={i} className="msg-zarva-card">
                  <div className="msg-zarva-header">
                    <span className="msg-zarva-badge">⚡ Zarva</span>
                    <button className="msg-copy-btn" title="Copy response"
                      onClick={() => navigator.clipboard.writeText(msg.content)}>⎘ Copy</button>
                  </div>
                  <div className="msg-zarva-body">
                    <MdText text={msg.content} onOptionClick={(opt) => {
                      setQuestion(opt)
                      setTimeout(() => document.getElementById('send-btn').click(), 50)
                    }} />
                  </div>
                  {msg.inlineCharts && Object.keys(msg.inlineCharts).length > 0 && (
                    <div className="inline-charts">
                      {Object.entries(msg.inlineCharts).map(([key, b64]) => (
                        <div key={key} className="chart-wrap">
                          <img src={`data:image/png;base64,${b64}`} alt={key} className="chart-img" />
                          <div className="chart-footer">
                            <span className="chart-type-label">📊 Generated chart</span>
                            <a href={`data:image/png;base64,${b64}`} download="zarva-chart.png"
                              className="chart-dl-btn" onClick={e => e.stopPropagation()}>↓ Download</a>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {msg.followups && msg.followups.length > 0 && (
                    <div className="followups">
                      {msg.followups.map((f, fi) => (
                        <button key={fi} className="followup-chip" onClick={() => {
                          setQuestion(f)
                          setTimeout(() => document.getElementById('send-btn').click(), 50)
                        }}>↳ {f}</button>
                      ))}
                    </div>
                  )}
                </div>
              )
            ))}
            {asking && messages[messages.length - 1]?.role !== 'assistant' && (
              <div className="msg-zarva-card">
                <div className="msg-zarva-header">
                  <span className="msg-zarva-badge">⚡ Zarva</span>
                </div>
                <div className="msg-zarva-body">
                  <span className="dots"><span>.</span><span>.</span><span>.</span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {messages.length > 0 && (
            <button type="button" className="export-btn" onClick={exportToExcel}>
              ↓ Export to Excel
            </button>
          )}

          <form onSubmit={handleAsk}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask something about your data..."
              disabled={!indexId}
            />
            <button id="send-btn" type="submit" disabled={(mode === 'data' && !indexId) || asking}>
              Send
            </button>
          </form>
        </main>
      </div>

      {analysisLoading && !analysis && (
        <div className="narrative-panel skeleton-panel">
          <div className="skeleton-badge" />
          <div className="skeleton-line wide" />
          <div className="skeleton-line" />
          <div className="skeleton-line medium" />
          <div className="skeleton-chips"><div className="skeleton-chip"/><div className="skeleton-chip"/><div className="skeleton-chip"/></div>
        </div>
      )}
      <NarrativePanel data={analysis} />

      <div className="analysis-layout">
        <div className="analysis-main">
          <InsightsPanel data={analysis} onSelect={setSelectedInsight} />
        </div>
        {selectedInsight && (
          <aside className="insight-detail-panel">
            <div className="insight-detail-header">
              <span className="insight-detail-type">{TYPE_META[selectedInsight.type]?.icon} {TYPE_META[selectedInsight.type]?.label || selectedInsight.type}</span>
              <button className="insight-detail-close" onClick={() => setSelectedInsight(null)}>✕</button>
            </div>
            <h3 className="insight-detail-title">{selectedInsight.title}</h3>
            {selectedInsight.value && (
              <div className="insight-detail-value" style={{color: TYPE_META[selectedInsight.type]?.color || '#22c55e'}}>
                {selectedInsight.direction === 'up' ? '↑' : selectedInsight.direction === 'down' ? '↓' : '→'} {selectedInsight.value}
              </div>
            )}
            {selectedInsight.evidence && (
              <div className="insight-detail-section">
                <span className="insight-detail-label">Evidence</span>
                <p className="insight-detail-text">{selectedInsight.evidence}</p>
              </div>
            )}
            {selectedInsight.action && (
              <div className="insight-detail-section">
                <span className="insight-detail-label">→ Recommended Action</span>
                <p className="insight-detail-text">{selectedInsight.action}</p>
              </div>
            )}
            <div className="insight-detail-severity">
              <span className={`insight-sev-badge`} style={{background: SEV_META[selectedInsight.severity]?.bg, color: SEV_META[selectedInsight.severity]?.fg}}>
                {SEV_META[selectedInsight.severity]?.badge}
              </span>
            </div>
          </aside>
        )}
      </div>

      {kpisLoading && !kpis && (
        <section className="kpi-panel">
          <h2 className="kpi-panel-title">Data Insights</h2>
          <div className="kpi-panel-grid">
            {[1,2,3].map(i => (
              <div key={i} className="kpi-panel-card skeleton-card">
                <div className="skeleton-icon"/>
                <div className="skeleton-line wide"/>
                <div className="skeleton-line short"/>
              </div>
            ))}
          </div>
        </section>
      )}
      {kpis && (
        <section className="kpi-panel">
          <h2 className="kpi-panel-title">Data Insights</h2>
          <div className="kpi-panel-grid">
            {/* Extracted doc → generic semantic cards */}
            {kpis.cards && kpis.cards.map((card, i) => (
              <div key={i} className={`kpi-panel-card ${i === 0 ? 'accent-green' : ''}`}>
                <span className="kpi-panel-icon">{card.icon}</span>
                <span className="kpi-panel-val">{card.value}</span>
                <span className="kpi-panel-label">{card.label}</span>
                {card.category && <span className="kpi-panel-cat">{card.category}</span>}
              </div>
            ))}
            {/* CSV / tabular → CRM KPIs */}
            {!kpis.cards && kpis.total_records && (
              <div className="kpi-panel-card">
                <span className="kpi-panel-icon">📋</span>
                <span className="kpi-panel-val">{kpis.total_records.toLocaleString()}</span>
                <span className="kpi-panel-label">Total Records</span>
              </div>
            )}
            {!kpis.cards && kpis.total_revenue && (
              <div className="kpi-panel-card accent-green">
                <span className="kpi-panel-icon">💰</span>
                <span className="kpi-panel-val">{kpis.total_revenue}</span>
                <span className="kpi-panel-label">Total Revenue</span>
              </div>
            )}
            {!kpis.cards && kpis.avg_deal && (
              <div className="kpi-panel-card">
                <span className="kpi-panel-icon">📊</span>
                <span className="kpi-panel-val">{kpis.avg_deal}</span>
                <span className="kpi-panel-label">Avg Deal Size</span>
              </div>
            )}
            {!kpis.cards && kpis.win_rate && (
              <div className="kpi-panel-card accent-blue">
                <span className="kpi-panel-icon">🏆</span>
                <span className="kpi-panel-val">{kpis.win_rate}</span>
                <span className="kpi-panel-label">Win Rate</span>
              </div>
            )}
            {!kpis.cards && kpis.positive_pct && (
              <div className="kpi-panel-card">
                <span className="kpi-panel-icon">😊</span>
                <span className="kpi-panel-val">{kpis.positive_pct}</span>
                <span className="kpi-panel-label">Positive Sentiment</span>
              </div>
            )}
            {!kpis.cards && kpis.revenue_forecast && (
              <div className="kpi-panel-card accent-purple">
                <span className="kpi-panel-icon">🔮</span>
                <span className="kpi-panel-val">{kpis.revenue_forecast}</span>
                <span className="kpi-panel-label">Revenue Forecast</span>
              </div>
            )}
            {!kpis.cards && kpis.top_owner && (
              <div className="kpi-panel-card">
                <span className="kpi-panel-icon">⭐</span>
                <span className="kpi-panel-val" style={{fontSize:'1rem'}}>{kpis.top_owner}</span>
                <span className="kpi-panel-label">Top Owner</span>
              </div>
            )}
            {!kpis.cards && kpis.top_stage && (
              <div className="kpi-panel-card">
                <span className="kpi-panel-icon">📍</span>
                <span className="kpi-panel-val" style={{fontSize:'1rem'}}>{kpis.top_stage}</span>
                <span className="kpi-panel-label">Top Stage</span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ── EDA Panel ──────────────────────────────────────────────── */}
      {eda && (
        <section className="eda-panel">
          <div className="eda-header">
            <div>
              <h2 className="eda-title">Data Overview</h2>
              <p className="eda-subtitle">Shape · Column Types · Statistical Summary · Data Quality</p>
            </div>
            <div className="eda-shape-chips">
              <span className="eda-chip">{eda.shape?.rows?.toLocaleString()} rows</span>
              <span className="eda-chip">{eda.shape?.columns} cols</span>
              {eda.shape?.duplicates > 0 && <span className="eda-chip warn">{eda.shape.duplicates} dupes</span>}
              {eda.shape?.missing_pct > 0 && <span className="eda-chip warn">{eda.shape.missing_pct}% missing</span>}
            </div>
          </div>

          <div className="eda-tabs">
            <button className="eda-tab active">Overview</button>
          </div>

          <div className="eda-body">

            {/* ── Overview ── */}
            {(
              <div className="eda-overview">

                {/* Shape cards */}
                <div className="eda-cards-row">
                  {[
                    {icon:'📐', val: eda.shape?.rows?.toLocaleString(), label:'Rows'},
                    {icon:'📋', val: eda.shape?.columns, label:'Columns'},
                    {icon:'🔁', val: eda.shape?.duplicates, label:'Duplicate Rows', warn: eda.shape?.duplicates > 0},
                    {icon:'❓', val: `${eda.shape?.missing_pct}%`, label:'Missing Cells', warn: eda.shape?.missing_pct > 0},
                    {icon:'🔢', val: eda.shape?.column_types?.numeric?.length || 0, label:'Numeric Cols'},
                    {icon:'🔤', val: eda.shape?.column_types?.categorical?.length || 0, label:'Categorical Cols'},
                  ].map((c,i) => (
                    <div key={i} className={`eda-shape-card ${c.warn ? 'warn' : ''}`}>
                      <span className="eda-shape-icon">{c.icon}</span>
                      <span className="eda-shape-val">{c.val}</span>
                      <span className="eda-shape-label">{c.label}</span>
                    </div>
                  ))}
                </div>

                {/* Missing values chart */}
                {eda.missing_chart && (
                  <div className="eda-chart-wrap">
                    <h3 className="eda-section-label">Missing Values by Column</h3>
                    <img src={`data:image/png;base64,${eda.missing_chart}`} alt="missing values" className="eda-chart" />
                  </div>
                )}

                {/* Column detail table */}
                <div className="eda-table-wrap">
                  <h3 className="eda-section-label">Column Profile</h3>
                  <div className="eda-table-scroll">
                    <table className="eda-table">
                      <thead>
                        <tr>{['Column','Type','Non-Null','Unique','Missing %'].map(h => <th key={h}>{h}</th>)}</tr>
                      </thead>
                      <tbody>
                        {(eda.shape?.per_column || []).map((c, i) => (
                          <tr key={i}>
                            <td className="col-name">{c.column}</td>
                            <td><span className={`dtype-badge ${c.dtype.includes('int') || c.dtype.includes('float') ? 'num' : 'cat'}`}>{c.dtype}</span></td>
                            <td>{(c.count ?? (eda.shape?.rows - c.missing))?.toLocaleString()}</td>
                            <td>{c.unique?.toLocaleString()}</td>
                            <td>
                              <div className="missing-bar-cell">
                                <div className="missing-bar-fill" style={{width: `${Math.min(c.missing_pct, 100)}%`, background: c.missing_pct > 20 ? '#ef4444' : c.missing_pct > 5 ? '#f59e0b' : '#22c55e'}} />
                                <span>{c.missing_pct}%</span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Statistical summary */}
                {eda.stat_summary?.numeric?.length > 0 && (
                  <div className="eda-table-wrap">
                    <h3 className="eda-section-label">Statistical Summary — Numeric Columns</h3>
                    <div className="eda-table-scroll">
                      <table className="eda-table">
                        <thead>
                          <tr>{['Column','Count','Mean','Std','Min','Q25','Median','Q75','Max','Skewness','Kurtosis'].map(h => <th key={h}>{h}</th>)}</tr>
                        </thead>
                        <tbody>
                          {eda.stat_summary.numeric.map((r, i) => (
                            <tr key={i}>
                              <td className="col-name">{r.column}</td>
                              <td>{r.count?.toLocaleString()}</td>
                              <td>{r.mean}</td>
                              <td>{r.std}</td>
                              <td>{r.min}</td>
                              <td>{r.q25}</td>
                              <td className="highlight">{r.median}</td>
                              <td>{r.q75}</td>
                              <td>{r.max}</td>
                              <td className={Math.abs(r.skewness) > 1 ? 'warn-cell' : ''}>{r.skewness}</td>
                              <td className={r.kurtosis > 3 ? 'warn-cell' : ''}>{r.kurtosis}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {eda.stat_summary?.categorical?.length > 0 && (
                  <div className="eda-table-wrap">
                    <h3 className="eda-section-label">Statistical Summary — Categorical Columns</h3>
                    <div className="eda-table-scroll">
                      <table className="eda-table">
                        <thead>
                          <tr>{['Column','Count','Unique','Top Value','Frequency','Mode %','Missing %'].map(h => <th key={h}>{h}</th>)}</tr>
                        </thead>
                        <tbody>
                          {eda.stat_summary.categorical.map((r, i) => (
                            <tr key={i}>
                              <td className="col-name">{r.column}</td>
                              <td>{r.count?.toLocaleString()}</td>
                              <td>{r.unique}</td>
                              <td className="highlight">{r.top}</td>
                              <td>{r.freq}</td>
                              <td>{r.mode_pct}%</td>
                              <td className={r.missing_pct > 5 ? 'warn-cell' : ''}>{r.missing_pct}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}


          </div>
        </section>
      )}

      {hypotheses && hypotheses.hypotheses && hypotheses.hypotheses.length > 0 && (
        <section className="hyp-panel">
          <div className="hyp-header">
            <div>
              <h2 className="hyp-title">Business Hypotheses</h2>
              <p className="hyp-subtitle">Zarva found these patterns — tested against your actual data</p>
            </div>
            <span className="hyp-count">{hypotheses.hypotheses.length} findings</span>
          </div>
          <div className="hyp-grid">
            {hypotheses.hypotheses.map((h, i) => <HypCard key={i} h={h} onSelect={h => setSelectedInsight({title: h.title, evidence: h.narrative, action: h.action, type: 'recommendation', severity: h.impact === 'high' ? 'critical' : 'warning', value: null, direction: 'up'})} />)}
          </div>
        </section>
      )}

      {ml && !ml.error && (
        <section className="ml-panel">
          <div className="ml-header">
            <h2 className="ml-title">ML Predictions</h2>
            {ml.summary && (
              <div className="ml-summary-chips">
                {ml.summary.avg_win_probability != null && <span className="ml-chip">Avg Win {ml.summary.avg_win_probability}%</span>}
                {ml.summary.high_risk_count > 0 && <span className="ml-chip danger">{ml.summary.high_risk_count} High Risk</span>}
                {ml.summary.anomaly_count > 0 && <span className="ml-chip warn">{ml.summary.anomaly_count} Anomalies</span>}
                {ml.summary.forecast_trend && <span className="ml-chip">{ml.summary.forecast_trend === 'up' ? '↑' : '↓'} Revenue Trend</span>}
              </div>
            )}
          </div>

          <div className="ml-tabs">
            {[['risk','Deal Risk'],['win','Win Probability'],['anomaly','Anomalies']].map(([k,label]) => (
              <button key={k} className={`ml-tab ${mlTab === k ? 'active' : ''}`} onClick={() => setMlTab(k)}>{label}</button>
            ))}
          </div>

          <div className="ml-body">
            {mlTab === 'risk' && (
              <div className="ml-cards">
                {(ml.risk_scores || []).length === 0 && <p className="ml-empty">No at-risk records detected.</p>}
                {(ml.risk_scores || []).map((r, i) => (
                  <div key={i} className={`ml-card risk-card ${r.risk_score > 60 ? 'high' : 'med'}`}>
                    <div className="ml-card-top">
                      <span className="ml-record-name">{r.name}</span>
                      <span className={`ml-badge ${r.risk_score > 60 ? 'badge-danger' : 'badge-warn'}`}>{r.risk_label}</span>
                    </div>
                    <div className="ml-card-stage">{r.stage}{r.amount > 0 ? ` · $${r.amount.toLocaleString()}` : ''}</div>
                    <div className="ml-risk-bar-wrap"><div className="ml-risk-bar" style={{width: `${r.risk_score}%`, background: r.risk_score > 60 ? '#ef4444' : '#f59e0b'}} /></div>
                    <div className="ml-card-reasons">{(r.reasons || []).map((rs, j) => <span key={j} className="ml-reason">{rs}</span>)}</div>
                  </div>
                ))}
              </div>
            )}

            {mlTab === 'win' && (
              <div className="ml-cards">
                {(ml.win_probabilities || []).length === 0 && <p className="ml-empty">Not enough closed records to train model.</p>}
                {(ml.win_probabilities || []).map((w, i) => (
                  <div key={i} className="ml-card win-card">
                    <div className="ml-card-top">
                      <span className="ml-record-name">{w.name}</span>
                      <span className={`ml-badge ${w.win_probability >= 70 ? 'badge-green' : w.win_probability >= 40 ? 'badge-warn' : 'badge-danger'}`}>{w.win_probability}%</span>
                    </div>
                    <div className="ml-card-stage">{w.stage}{w.amount > 0 ? ` · $${w.amount.toLocaleString()}` : ''}</div>
                    <div className="ml-prob-bar-wrap">
                      <div className="ml-prob-bar" style={{width: `${w.win_probability}%`, background: w.win_probability >= 70 ? '#22c55e' : w.win_probability >= 40 ? '#f59e0b' : '#ef4444'}} />
                    </div>
                    <div className="ml-card-sub">Sentiment: {w.sentiment > 0.1 ? '😊 Positive' : w.sentiment < -0.1 ? '😟 Negative' : '😐 Neutral'}</div>
                  </div>
                ))}
              </div>
            )}

            {mlTab === 'anomaly' && (
              <div className="ml-cards">
                {(ml.anomalies || []).length === 0 && <p className="ml-empty">No anomalies detected.</p>}
                {(ml.anomalies || []).map((a, i) => (
                  <div key={i} className="ml-card anomaly-card">
                    <div className="ml-card-top">
                      <span className="ml-record-name">{a.name}</span>
                      <span className="ml-badge badge-purple">Anomaly</span>
                    </div>
                    <div className="ml-card-stage">{a.stage}{a.amount > 0 ? ` · $${a.amount.toLocaleString()}` : ''}</div>
                    <div className="ml-card-reasons">{(a.reasons || []).map((rs, j) => <span key={j} className="ml-reason">{rs}</span>)}</div>
                  </div>
                ))}
              </div>
            )}

          </div>
        </section>
      )}

      <FAQ />
    </div>
  )
}

export default App
