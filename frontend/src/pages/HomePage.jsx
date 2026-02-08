import React from 'react';
import { Link } from 'react-router-dom';

function HomePage() {
  const capabilities = [
    {
      title: 'Signal Ingestion',
      description: 'Collect high-volume social posts continuously and normalize them into a clean analytics stream.',
    },
    {
      title: 'NLP Sentiment Scoring',
      description: 'Score each relevant sentence and classify mood shifts for tracked assets and macro topics.',
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
              <Link to="/architecture" className="btn-secondary">
                Explore Architecture
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
                <span>Show sentence-level evidence so model outputs stay explainable.</span>
              </li>
              <li className="capability-item">
                <span className="capability-dot" />
                <span>Reveal dominant keyword themes driving market narratives.</span>
              </li>
            </ul>
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
            <span className="metric-value">4</span>
            <span className="metric-label">Live analytics endpoints</span>
          </div>
          <div className="metric-block">
            <span className="metric-value">Sentence-level</span>
            <span className="metric-label">Evidence granularity</span>
          </div>
          <div className="metric-block">
            <span className="metric-value">Near real-time</span>
            <span className="metric-label">Refreshable signal feed</span>
          </div>
        </section>
      </div>
    </div>
  );
}

export default HomePage;
