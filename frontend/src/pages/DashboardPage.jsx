import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import SentimentChart from '../components/SentimentChart';
import { getTrackedAssets, getMacroDailyChart, getMacroSummary } from '../api/sentimentApi';

const formatScore = (score) => {
  const parsed = Number(score);
  if (!Number.isFinite(parsed)) return '0.000';
  return parsed > 0 ? `+${parsed.toFixed(3)}` : parsed.toFixed(3);
};

function DashboardPage() {
  const [assets, setAssets] = useState([]);
  const [macroChart, setMacroChart] = useState([]);
  const [macroScore, setMacroScore] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      const [assetRes, chartRes, summaryRes] = await Promise.allSettled([
        getTrackedAssets(),
        getMacroDailyChart(),
        getMacroSummary(),
      ]);

      if (assetRes.status === 'fulfilled') {
        setAssets(Array.isArray(assetRes.value) ? assetRes.value : []);
      }
      if (chartRes.status === 'fulfilled' && Array.isArray(chartRes.value)) {
        setMacroChart(chartRes.value);
      }
      if (summaryRes.status === 'fulfilled') {
        const avg = Number(summaryRes.value?.averageScore);
        setMacroScore(Number.isFinite(avg) ? avg : 0);
      }

      if (assetRes.status === 'rejected' || chartRes.status === 'rejected') {
        setError('Some dashboard data could not be loaded.');
      }

      setLoading(false);
    };

    fetchData();
  }, []);

  return (
    <div className="dashboard-page">
      <div className="page-wrap">
        <section className="dashboard-hero panel">
          <div>
            <p className="hero-tag">Operations Snapshot</p>
            <h1 className="hero-title">Market Signal Dashboard</h1>
            <p className="hero-copy">
              This view gives a quick operational summary of what is currently tracked and where to drill deeper for
              trend and evidence analysis.
            </p>
            <div className="cta-row">
              <Link to="/sentiment-summary" className="btn-primary">
                Open Sentiment Module
              </Link>
              <Link to="/architecture" className="btn-secondary">
                View Data Flow
              </Link>
            </div>
          </div>
          <div className="dashboard-metrics">
            <article className="panel dashboard-metric-card">
              <p className="summary-label">Tracked Assets</p>
              <p className="summary-score">{assets.length}</p>
              <p className="summary-subtitle">Symbols available for sentiment analysis</p>
            </article>
            <article className="panel dashboard-metric-card">
              <p className="summary-label">Pipeline Status</p>
              <p className="summary-score">Online</p>
              <p className="summary-subtitle">Ingestion and NLP worker pipeline active</p>
            </article>
          </div>
        </section>

        <section className="dashboard-assets panel">
          <div className="panel-head">
            <div>
              <h2>Macro Sentiment Trend</h2>
              <p>Daily average sentiment aggregated across all {assets.length} tracked assets.</p>
            </div>
            <span className={`tone-pill ${macroScore > 0.05 ? 'is-positive' : macroScore < -0.05 ? 'is-negative' : 'is-neutral'}`}>
              {formatScore(macroScore)}
            </span>
          </div>

          {loading ? <p className="loading-note">Loading macro sentiment...</p> : null}
          {error ? <p className="inline-alert">{error}</p> : null}

          {!loading && !error ? (
            <div className="chart-shell">
              <SentimentChart data={macroChart} ticker="ALL" />
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}

export default DashboardPage;
