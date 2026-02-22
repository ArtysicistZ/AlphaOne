import React, { useState } from 'react';
import { runInference } from '../api/sentimentApi';

const EXAMPLES = [
  { text: 'AAPL is great but TSLA is doomed', targets: 'AAPL, TSLA' },
  { text: 'NVDA earnings were incredible, this stock is going to the moon', targets: 'NVDA' },
  { text: 'I sold all my AMD shares, the semiconductor sector looks weak', targets: 'AMD' },
  { text: 'Microsoft keeps delivering while Google is falling behind', targets: 'Microsoft, Google' },
];

const sentimentClass = (label) => {
  const l = (label ?? '').toLowerCase();
  if (l === 'bullish' || l === 'positive') return 'is-positive';
  if (l === 'bearish' || l === 'negative') return 'is-negative';
  return 'is-neutral';
};

function PlaygroundPage() {
  const [text, setText] = useState('');
  const [targets, setTargets] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim() || !targets.trim()) return;

    const targetList = targets.split(',').map((t) => t.trim()).filter(Boolean);
    if (targetList.length === 0) return;

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const data = await runInference(text.trim(), targetList);
      setResults(data.results);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(detail || 'Inference request failed. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const fillExample = (ex) => {
    setText(ex.text);
    setTargets(ex.targets);
    setResults(null);
    setError('');
  };

  return (
    <div className="playground-page">
      <div className="page-wrap">
        <section className="playground-hero panel">
          <p className="hero-tag">Interactive Demo</p>
          <h1 className="playground-title">ABSA Model Playground</h1>
          <p className="playground-subtitle">
            Test our fine-tuned DeBERTa-v3 model in real time. Enter any sentence and the
            target words you want to analyze. The model uses entity replacement to classify
            sentiment toward each target independently.
          </p>
        </section>

        <section className="playground-input panel">
          <form onSubmit={handleSubmit} className="playground-form">
            <div className="playground-field">
              <label htmlFor="pg-text" className="playground-label">Sentence</label>
              <textarea
                id="pg-text"
                className="playground-textarea"
                rows={3}
                placeholder='e.g. "AAPL is great but TSLA is doomed"'
                value={text}
                onChange={(e) => setText(e.target.value)}
                maxLength={1000}
              />
            </div>

            <div className="playground-row">
              <div className="playground-field playground-field--target">
                <label htmlFor="pg-targets" className="playground-label">
                  Targets <span className="playground-optional">(comma-separated)</span>
                </label>
                <input
                  id="pg-targets"
                  type="text"
                  className="text-input"
                  placeholder="e.g. AAPL, TSLA"
                  value={targets}
                  onChange={(e) => setTargets(e.target.value)}
                />
              </div>

              <button
                type="submit"
                className="btn-primary playground-submit"
                disabled={loading || !text.trim() || !targets.trim()}
              >
                {loading ? 'Analyzing...' : 'Analyze Sentiment'}
              </button>
            </div>
          </form>

          <div className="playground-examples">
            <span className="playground-examples-label">Try an example:</span>
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                type="button"
                className="playground-example-btn"
                onClick={() => fillExample(ex)}
              >
                &ldquo;{ex.text.length > 40 ? ex.text.slice(0, 40) + '...' : ex.text}&rdquo; &rarr; {ex.targets}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="inline-alert">{error}</div>
        )}

        {results && results.length > 0 && (
          <section className="playground-results">
            <h2 className="playground-results-title">Results</h2>
            <div className="playground-results-grid">
              {results.map((r, i) => (
                <article key={i} className="playground-result-card panel">
                  <div className="result-header">
                    <span className="result-ticker">{r.target}</span>
                    <span className={`evidence-badge ${sentimentClass(r.sentiment)}`}>
                      {r.sentiment.toUpperCase()}
                    </span>
                    <span className="evidence-score">
                      {r.score > 0 ? `+${r.score.toFixed(4)}` : r.score.toFixed(4)}
                    </span>
                  </div>
                  <p className="result-sentence">{r.sentence}</p>
                  <div className="result-model-input">
                    <span className="result-model-label">Model sees:</span>
                    <code className="result-model-text">{r.normalizedInput}</code>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

export default PlaygroundPage;
