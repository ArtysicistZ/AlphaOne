import React from 'react';
import { Link } from 'react-router-dom';

function HomePage() {
  const capabilities = [
    {
      title: 'Signal Ingestion',
      description: 'Collect high-volume social posts continuously and normalize them into a clean analytics stream.',
    },
    {
      title: 'Subject-Aware NLP',
      description:
        'Score sentiment per stock using a fine-tuned DeBERTa-v3 ABSA model with entity replacement — one sentence, multiple verdicts.',
    },
    {
      title: 'Actionable Visuals',
      description: 'Turn noisy sentiment flow into trend charts, keyword clouds, and evidence-level explanations.',
    },
  ];

  return (
    <div className="home-page">
      <div className="page-wrap">
        <section className="welcome-hero panel">
          <div>
            <p className="hero-tag">Social Sentiment Intelligence</p>
            <h1 className="hero-title">Welcome to AlphaOne</h1>
            <p className="hero-copy">
              AlphaOne turns large-scale social chatter into an interpretable signal layer for markets. Track
              sentiment by ticker and theme, inspect trend changes, and review direct textual evidence in one
              workflow.
            </p>

            <div className="cta-row">
              <Link to="/dashboard" className="btn-primary">
                Open Live Dashboard
              </Link>
              <Link to="/playground" className="btn-secondary">
                Try the Model
              </Link>
            </div>
          </div>

          <div>
            <h2 className="hero-side-title">What this platform can do</h2>
            <ul className="capability-list">
              <li className="capability-item">
                <span className="capability-dot" />
                <span>Surface daily sentiment momentum for each tracked asset.</span>
              </li>
              <li className="capability-item">
                <span className="capability-dot" />
                <span>Classify sentiment <strong>per stock</strong> in multi-ticker sentences.</span>
              </li>
              <li className="capability-item">
                <span className="capability-dot" />
                <span>Show sentence-level evidence so model outputs stay explainable.</span>
              </li>
              <li className="capability-item">
                <span className="capability-dot" />
                <span>Reveal dominant keyword themes driving market narratives.</span>
              </li>
            </ul>
          </div>
        </section>

        <section className="absa-showcase panel">
          <div className="absa-showcase-text">
            <p className="hero-tag">Core Innovation</p>
            <h2 className="absa-showcase-title">Subject-Aware Sentiment Analysis</h2>
            <p className="absa-showcase-copy">
              Standard models produce one label per sentence. But when someone writes
              &ldquo;AAPL is great but TSLA is doomed&rdquo;, the sentiment is different for each
              stock. Our fine-tuned{' '}
              <a
                href="https://huggingface.co/ArtysicistZ/absa-deberta"
                target="_blank"
                rel="noopener noreferrer"
                className="absa-link"
              >
                DeBERTa-v3 ABSA model
              </a>{' '}
              solves this using <strong>entity replacement</strong> &mdash; classifying sentiment
              toward each stock independently.
            </p>
            <Link to="/playground" className="btn-primary" style={{ marginTop: '1rem', display: 'inline-block' }}>
              Try It Yourself
            </Link>
          </div>

          <div className="absa-demo">
            <div className="absa-demo-card">
              <div className="absa-demo-label">Same sentence, two analyses:</div>
              <div className="absa-demo-example">
                <code className="absa-demo-input">&ldquo;AAPL is great but TSLA is doomed&rdquo;</code>
              </div>
              <div className="absa-demo-row">
                <span className="absa-demo-target">target: AAPL</span>
                <span className="absa-demo-arrow">&rarr;</span>
                <code className="absa-demo-replaced">target is great but other is doomed</code>
                <span className="evidence-badge is-positive">BULLISH</span>
              </div>
              <div className="absa-demo-row">
                <span className="absa-demo-target">target: TSLA</span>
                <span className="absa-demo-arrow">&rarr;</span>
                <code className="absa-demo-replaced">other is great but target is doomed</code>
                <span className="evidence-badge is-negative">BEARISH</span>
              </div>
            </div>
          </div>
        </section>

        <section id="capabilities" className="capability-grid">
          {capabilities.map((item) => (
            <article key={item.title} className="capability-card">
              <h3>{item.title}</h3>
              <p>{item.description}</p>
            </article>
          ))}
        </section>

        <section className="metric-strip panel">
          <div className="metric-block">
            <span className="metric-value">DeBERTa-v3</span>
            <span className="metric-label">Fine-tuned ABSA model</span>
          </div>
          <div className="metric-block">
            <span className="metric-value">Per-subject</span>
            <span className="metric-label">Sentiment granularity</span>
          </div>
          <div className="metric-block">
            <span className="metric-value">79.4%</span>
            <span className="metric-label">Accuracy (3-class)</span>
          </div>
        </section>
      </div>
    </div>
  );
}

export default HomePage;
