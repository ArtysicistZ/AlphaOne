import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

const SERVICES = [
  {
    title: 'React Frontend',
    tech: 'React 18, Vite, Chart.js',
    description:
      'Single-page dashboard with sentiment trend charts, per-ticker evidence feeds, and an interactive model playground.',
  },
  {
    title: 'Spring Boot API',
    tech: 'Java 21, Spring Boot 3.5, Spring Data JPA',
    description:
      'Read-only REST API serving processed sentiment data, daily aggregations, and asset tracking to the frontend.',
  },
  {
    title: 'Celery Worker',
    tech: 'Python 3.11, Celery 5, PRAW, spaCy',
    description:
      'Scheduled every 2 hours via Celery Beat. Fetches Reddit posts, splits sentences, normalizes tickers, applies entity replacement, and writes results to PostgreSQL.',
  },
  {
    title: 'Inference API',
    tech: 'Python, FastAPI, Transformers',
    description:
      'Dedicated service for real-time model inference. Powers the interactive playground and proxied inference from Spring Boot.',
  },
  {
    title: 'PostgreSQL',
    tech: 'External managed database (e.g. Neon)',
    description:
      'Stores raw Reddit posts, per-subject sentiment results, topic tags, and daily aggregations. Not containerized.',
  },
  {
    title: 'Redis',
    tech: 'In-memory store',
    description: 'Message broker and result backend for the Celery task queue.',
  },
];

const ML_PIPELINE = [
  {
    step: '1',
    title: 'Sentence Splitting',
    detail: 'Raw post text is split into individual sentences using spaCy.',
  },
  {
    step: '2',
    title: 'Ticker Normalization',
    detail:
      'Stock mentions ($AAPL, AAPL, Apple) are detected and mapped to canonical ticker symbols.',
  },
  {
    step: '3',
    title: 'Entity Replacement',
    detail:
      'For each (sentence, target) pair: the target ticker becomes "target", all others become "other".',
  },
  {
    step: '4',
    title: 'Batch Inference',
    detail:
      'Replaced sentences are tokenized and passed through DeBERTa-ABSA-v2 in a single batched forward pass.',
  },
  {
    step: '5',
    title: 'Classification',
    detail:
      'Softmax over 3 logits produces per-subject labels (bullish / bearish / neutral) and confidence scores.',
  },
];

const ArchitecturePage = () => {
  const wrapRef = useRef(null);

  useEffect(() => {
    function drawConnector() {
      const wrap = wrapRef.current;
      if (!wrap) return;
      const svg = wrap.querySelector('.hflow-connector');
      const nlp = wrap.querySelector('#nlp-node');
      const hf = wrap.querySelector('#hf-node');
      if (!svg || !nlp || !hf) return;

      const wrapRect = wrap.getBoundingClientRect();
      const nlpRect = nlp.getBoundingClientRect();
      const hfRect = hf.getBoundingClientRect();

      // NLP Pipeline bottom-center
      const x1 = nlpRect.left + nlpRect.width / 2 - wrapRect.left;
      const y1 = nlpRect.bottom - wrapRect.top;
      // HuggingFace top-center
      const x2 = hfRect.left + hfRect.width / 2 - wrapRect.left;
      const y2 = hfRect.top - wrapRect.top;
      // Midpoint Y for horizontal segment
      const yMid = (y1 + y2) / 2;

      svg.setAttribute('width', wrapRect.width);
      svg.setAttribute('height', wrapRect.height);
      svg.innerHTML = `
        <path d="M${x2},${y2} L${x2},${yMid} L${x1},${yMid} L${x1},${y1}"
              fill="none" stroke="#667b8d" stroke-width="1.5" stroke-dasharray="6,4"/>
        <text x="${(x1 + x2) / 2}" y="${yMid - 6}" text-anchor="middle"
              fill="#667b8d" font-size="11" font-weight="600">model feeds runtime</text>
      `;
    }

    drawConnector();
    window.addEventListener('resize', drawConnector);
    return () => window.removeEventListener('resize', drawConnector);
  }, []);

  return (
    <div className="architecture-page">
      <div className="page-wrap">
        <section className="architecture-hero panel">
          <div>
            <p className="hero-tag">System Design</p>
            <h1 className="hero-title">Platform Architecture</h1>
            <p className="hero-copy">
              End-to-end pipeline from Reddit ingestion to per-stock sentiment analytics. Five
              Docker Compose services coordinate data collection, NLP inference, storage, and
              visualization.
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

        {/* ---- Full-Stack Data Flow ---- */}
        <section className="arch-section">
          <h2 className="arch-section-title">System Architecture</h2>
          <p className="arch-section-subtitle">
            Two pipelines: a runtime pipeline that continuously ingests and serves sentiment, and a
            training pipeline that produces the model powering it.
          </p>

          <div className="hflow-wrap panel" ref={wrapRef}>
            <div className="hflow-label">Runtime Pipeline</div>
            <div className="hflow-row" id="runtime-row">
              <div className="hflow-node hflow-node--source">
                <strong>Reddit API</strong>
                <span>PRAW</span>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-node hflow-node--worker">
                <strong>Celery Worker</strong>
                <span>Every 2 hrs</span>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-node hflow-node--ml" id="nlp-node">
                <strong>NLP Pipeline</strong>
                <span>Entity replacement + DeBERTa-ABSA-v2</span>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-node hflow-node--db">
                <strong>PostgreSQL</strong>
                <span>Sentiment data</span>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-stack">
                <div className="hflow-node hflow-node--api">
                  <strong>Spring Boot API</strong>
                  <span>REST endpoints</span>
                </div>
                <div className="hflow-node hflow-node--api">
                  <strong>Inference API</strong>
                  <span>FastAPI playground</span>
                </div>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-node hflow-node--frontend">
                <strong>React Dashboard</strong>
                <span>Charts + evidence</span>
              </div>
            </div>

            <div className="hflow-label">Training Pipeline</div>
            <div className="hflow-row">
              <div className="hflow-stack">
                <div className="hflow-node hflow-node--data">
                  <strong>Reddit Labels</strong>
                  <span>4,501 audited</span>
                </div>
                <div className="hflow-node hflow-node--data">
                  <strong>Synthetic Data</strong>
                  <span>1,786 generated</span>
                </div>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-node hflow-node--ml">
                <strong>Fine-Tuning</strong>
                <span>DeBERTa-v3-base, full FT, 8 epochs</span>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-node hflow-node--ml">
                <strong>Eval</strong>
                <span>82.5% acc, 0.823 F1</span>
              </div>
              <div className="hflow-arrow" />
              <div className="hflow-node hflow-node--publish" id="hf-node">
                <strong>HuggingFace</strong>
                <span>DeBERTa-ABSA-v2</span>
              </div>
            </div>

            <svg className="hflow-connector" />
          </div>
        </section>

        {/* ---- Service Cards ---- */}
        <section className="arch-section">
          <h2 className="arch-section-title">Services</h2>
          <p className="arch-section-subtitle">
            Six components orchestrated by Docker Compose.
          </p>
          <div className="architecture-stack-grid">
            {SERVICES.map((svc) => (
              <article key={svc.title} className="stack-card panel">
                <h3>{svc.title}</h3>
                <p className="stack-card-tech">{svc.tech}</p>
                <p>{svc.description}</p>
              </article>
            ))}
          </div>
        </section>

        {/* ---- ML Pipeline ---- */}
        <section className="arch-section">
          <h2 className="arch-section-title">ML Pipeline: DeBERTa-ABSA-v2</h2>
          <p className="arch-section-subtitle">
            184M-parameter DeBERTa-v3-base fine-tuned for 3-class entity-level sentiment (bullish /
            bearish / neutral) on 6,287 hand-audited Reddit training pairs. 82.5% accuracy, 0.823
            macro F1.
          </p>
          <div className="ml-pipeline-steps">
            {ML_PIPELINE.map((s) => (
              <div key={s.step} className="ml-step">
                <span className="ml-step-num">{s.step}</span>
                <div>
                  <strong>{s.title}</strong>
                  <p>{s.detail}</p>
                </div>
              </div>
            ))}
          </div>

          <h3 className="arch-subsection-title">Training Ablation</h3>
          <p className="arch-section-subtitle">
            Key milestones across 9 iterations, 4 base models, and 4 dataset versions.
          </p>
          <div className="api-table-wrap panel">
            <table className="api-table api-table--compact">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Base Model</th>
                  <th>Dataset</th>
                  <th>Key Change</th>
                  <th>Acc</th>
                  <th>Macro F1</th>
                  <th>Bull F1</th>
                  <th>Bear F1</th>
                  <th>Neut F1</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>baseline</td>
                  <td>FinBERT</td>
                  <td>3,500</td>
                  <td>No fine-tuning (0-shot)</td>
                  <td>0.543</td>
                  <td>0.433</td>
                  <td>0.162</td>
                  <td>0.475</td>
                  <td>0.662</td>
                </tr>
                <tr>
                  <td>v1</td>
                  <td>FinBERT</td>
                  <td>3,500</td>
                  <td>Initial fine-tune</td>
                  <td>0.744</td>
                  <td>0.684</td>
                  <td>0.496</td>
                  <td>0.749</td>
                  <td>0.808</td>
                </tr>
                <tr>
                  <td>v4</td>
                  <td>DeBERTa-v3</td>
                  <td>3,500</td>
                  <td>Switched to DeBERTa-v3</td>
                  <td>0.805</td>
                  <td>0.757</td>
                  <td>0.640</td>
                  <td>0.777</td>
                  <td>0.853</td>
                </tr>
                <tr>
                  <td>v7</td>
                  <td>DeBERTa-v3</td>
                  <td>6,233</td>
                  <td>Plain-word entity replacement + synthetic multi-target</td>
                  <td>0.740</td>
                  <td>0.733</td>
                  <td>0.694</td>
                  <td>0.744</td>
                  <td>0.761</td>
                </tr>
                <tr>
                  <td>v8</td>
                  <td>DeBERTa-v3</td>
                  <td>5,985</td>
                  <td>Hand-audited labels (~615 corrections)</td>
                  <td>0.769</td>
                  <td>0.767</td>
                  <td>0.771</td>
                  <td>0.749</td>
                  <td>0.780</td>
                </tr>
                <tr className="ablation-best">
                  <td><strong>v9</strong></td>
                  <td><strong>DeBERTa-v3</strong></td>
                  <td><strong>6,287</strong></td>
                  <td><strong>+ 302 error-targeted synthetic pairs</strong></td>
                  <td><strong>0.825</strong></td>
                  <td><strong>0.823</strong></td>
                  <td><strong>0.836</strong></td>
                  <td><strong>0.800</strong></td>
                  <td><strong>0.833</strong></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="arch-subsection-title">Published Benchmark Comparison</h3>
          <div className="api-table-wrap panel">
            <table className="api-table">
              <thead>
                <tr>
                  <th>System</th>
                  <th>Domain</th>
                  <th>Macro F1</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>SEntFiN (RoBERTa)</td>
                  <td>Formal news headlines</td>
                  <td>0.933</td>
                </tr>
                <tr>
                  <td>FinEntity (FinBERT-CRF)</td>
                  <td>Formal financial news</td>
                  <td>0.850</td>
                </tr>
                <tr className="ablation-best">
                  <td><strong>AlphaOne (DeBERTa-v3)</strong></td>
                  <td><strong>Reddit (informal)</strong></td>
                  <td><strong>0.823</strong></td>
                </tr>
                <tr>
                  <td>FinEntity (ChatGPT 0-shot)</td>
                  <td>Formal financial news</td>
                  <td>0.560</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="arch-section-footnote">
            Our 0.823 on Reddit text (sarcasm, slang, implicit sentiment) is within 3pp of
            formal-news SOTA, using a 184M-parameter model on 6K training pairs.
          </p>
        </section>

        {/* ---- API Endpoints ---- */}
        <section className="arch-section">
          <h2 className="arch-section-title">API Endpoints</h2>
          <p className="arch-section-subtitle">Spring Boot REST API on port 8080.</p>
          <div className="api-table-wrap panel">
            <table className="api-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Endpoint</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>GET</code></td>
                  <td><code>/api/v1/assets/tracked</code></td>
                  <td>List all tracked stock tickers</td>
                </tr>
                <tr>
                  <td><code>GET</code></td>
                  <td><code>/api/v1/signals/social-sentiment/&#123;ticker&#125;/evidence</code></td>
                  <td>Sentence-level sentiment evidence for a ticker</td>
                </tr>
                <tr>
                  <td><code>GET</code></td>
                  <td><code>/api/v1/signals/social-sentiment/&#123;ticker&#125;/daily</code></td>
                  <td>Daily sentiment aggregation</td>
                </tr>
                <tr>
                  <td><code>GET</code></td>
                  <td><code>/api/v1/signals/social-sentiment/summary/&#123;topicSlug&#125;</code></td>
                  <td>Summary stats for a topic</td>
                </tr>
                <tr>
                  <td><code>POST</code></td>
                  <td><code>/api/v1/inference</code></td>
                  <td>Real-time inference (proxied to FastAPI)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
};

export default ArchitecturePage;
