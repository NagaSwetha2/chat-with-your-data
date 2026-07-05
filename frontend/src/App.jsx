import { useState } from 'react'
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
  const [charts, setCharts] = useState(null)
  const [mode, setMode] = useState('data')

  async function loadCharts(id) {
    try {
      const res = await fetch(`${API_BASE}/charts/${id}`)
      if (res.ok) setCharts(await res.json())
    } catch {}
  }

  function exportToExcel() {
    const rows = messages.map((msg) => ({
      Role: msg.role === 'user' ? 'You' : 'Zarva',
      Message: msg.content,
    }))
    const ws = XLSX.utils.json_to_sheet(rows)
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Zarva Chat')
    XLSX.writeFile(wb, 'zarva-standup-export.xlsx')
  }

  async function handleFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return

    setIndexing(true)
    setMessages([])
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch(`${API_BASE}/index`, { method: 'POST', body: formData })
    const data = await res.json()
    const id = data.index_id

    // Poll until ready
    while (true) {
      await new Promise(r => setTimeout(r, 2000))
      const s = await fetch(`${API_BASE}/status/${id}`).then(r => r.json())
      if (s.status === 'ready') {
        setIndexId(id)
        setRecordCount(s.record_count)
        setIndexing(false)
        loadCharts(id)
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
    if (!question.trim() || !indexId) return

    const userMsg = { role: 'user', content: question }
    setMessages((prev) => [...prev, userMsg])
    setQuestion('')
    setAsking(true)

    const endpoint = mode === 'sf' ? '/sf-chat' : '/chat'
    const body = mode === 'sf'
      ? JSON.stringify({ question: userMsg.content })
      : JSON.stringify({ index_id: indexId, question: userMsg.content })

    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    })
    const data = await res.json()
    setMessages((prev) => [...prev, { role: 'assistant', content: data.answer }])
    setAsking(false)
  }

  return (
    <div className="app">
      <header>
        <h1>Zarva</h1>
        <p>Upload a CSV or text file, then ask questions about it.</p>
        <div className="mode-toggle">
          <button
            className={mode === 'data' ? 'mode-btn active' : 'mode-btn'}
            onClick={() => { setMode('data'); setMessages([]) }}>
            Your Data
          </button>
          <button
            className={mode === 'sf' ? 'mode-btn active' : 'mode-btn'}
            onClick={() => { setMode('sf'); setMessages([]) }}>
            Salesforce KB
          </button>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <h2>1. Upload your data</h2>
          <input type="file" accept=".csv,.txt" onChange={handleFileUpload} />
          {indexing && <p className="status">Indexing your data...</p>}
          {!indexing && recordCount !== null && (
            <p className="status success">Indexed {recordCount} records. Ask away →</p>
          )}
        </aside>

        <main className="chat">
          <h2>2. Ask questions about your data</h2>

          {mode === 'sf' && <p className="hint sf-hint">Ask any Salesforce or SOQL question — no upload needed.</p>}
          {mode === 'data' && !indexId && <p className="hint">Upload and index a file first using the sidebar.</p>}

          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <strong>{msg.role === 'user' ? 'You' : 'Zarva'}</strong> {msg.content}
              </div>
            ))}
            {asking && (
              <div className="message assistant typing">
                <strong>Zarva</strong>
                <span className="dots"><span>.</span><span>.</span><span>.</span></span>
              </div>
            )}
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
            <button type="submit" disabled={(mode === 'data' && !indexId) || asking}>
              Send
            </button>
          </form>
        </main>
      </div>

      {charts && (
        <section className="dashboard">
          <h2>Data Insights</h2>
          <div className="chart-grid">
            {Object.entries(charts).map(([key, b64]) => (
              <div key={key} className="chart-card">
                <img src={`data:image/png;base64,${b64}`} alt={key} />
              </div>
            ))}
          </div>
        </section>
      )}

      <FAQ />
    </div>
  )
}

export default App
