import { useState } from 'react'
import './App.css'

const API_BASE = ''

function App() {
  const [indexId, setIndexId] = useState(null)
  const [recordCount, setRecordCount] = useState(null)
  const [indexing, setIndexing] = useState(false)
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)

  async function handleFileUpload(e) {
    const file = e.target.files[0]
    if (!file) return

    setIndexing(true)
    setMessages([])
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch(`${API_BASE}/index`, {
      method: 'POST',
      body: formData,
    })
    const data = await res.json()
    setIndexId(data.index_id)
    setRecordCount(data.record_count)
    setIndexing(false)
  }

  async function handleAsk(e) {
    e.preventDefault()
    if (!question.trim() || !indexId) return

    const userMsg = { role: 'user', content: question }
    setMessages((prev) => [...prev, userMsg])
    setQuestion('')
    setAsking(true)

    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index_id: indexId, question: userMsg.content }),
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

          {!indexId && <p className="hint">Upload and index a file first using the sidebar.</p>}

          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <strong>{msg.role === 'user' ? 'You' : 'Zarva'}</strong> {msg.content}
              </div>
            ))}
            {asking && <div className="message assistant">Thinking...</div>}
          </div>

          <form onSubmit={handleAsk}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask something about your data..."
              disabled={!indexId}
            />
            <button type="submit" disabled={!indexId || asking}>
              Send
            </button>
          </form>
        </main>
      </div>
    </div>
  )
}

export default App
