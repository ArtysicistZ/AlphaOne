import React from 'react';
import { Link } from 'react-router-dom';

const STACKS = [
  {
    title: 'Frontend',
    items: ['React 18', 'Vite', 'Chart.js', 'react-d3-cloud'],
  },
  {
    title: 'Backend API',
    items: ['Java 21', 'Spring Boot 3.5', 'REST API'],
  },
  {
    title: 'Worker',
    items: ['Python 3.11', 'Celery', 'spaCy', 'Transformers'],
  },
  {
    title: 'Data Layer',
    items: ['PostgreSQL', 'Redis', 'Daily aggregations'],
  },
];

const ArchitecturePage = () => {
  return (
    <div className="architecture-page">
      <div className="page-wrap">
        <section className="architecture-hero panel">
          <div>
            <p className="hero-tag">System Design</p>
            <h1 className="hero-title">Platform Architecture</h1>
            <p className="hero-copy">
              End-to-end pipeline from source ingestion to visual analytics. This page maps the runtime path behind
              the sentiment dashboard.
            </p>
            <div className="cta-row">
              <Link to="/sentiment-summary" className="btn-primary">
                Open Sentiment View
              </Link>
              <Link to="/dashboard" className="btn-secondary">
                Back to Dashboard
              </Link>
            </div>
          </div>
        </section>

        <section className="architecture-flow panel">
          <div className="panel-head">
            <div>
              <h2>Data Flow Pipeline</h2>
              <p>How raw social text becomes daily sentiment and explainable evidence.</p>
            </div>
          </div>
          <pre className="pipeline-diagram">{`[REDDIT STREAM]
      |
      v
[INGESTION SERVICE] --> [REDIS QUEUE] --> [CELERY WORKERS]
                                              |
                                              v
                                        [POSTGRESQL DB]
                                              |
                                              v
                                       [SPRING BOOT API]
                                              |
                                              v
                                        [REACT FRONTEND]`}</pre>
        </section>

        <section className="architecture-stack-grid">
          {STACKS.map((group) => (
            <article key={group.title} className="stack-card panel">
              <h3>{group.title}</h3>
              <ul>
                {group.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          ))}
        </section>
      </div>
    </div>
  );
};

export default ArchitecturePage;
