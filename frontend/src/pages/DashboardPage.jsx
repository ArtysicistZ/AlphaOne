import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { getTrackedAssets } from '../api/sentimentApi';

function DashboardPage() {
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const data = await getTrackedAssets();
        setAssets(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error('Error fetching assets:', error);
        setError('Failed to load tracked assets.');
      } finally {
        setLoading(false);
      }
    };

    fetchAssets();
  }, []);

  const topAssets = useMemo(() => assets.slice(0, 8), [assets]);

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
              <h2>Tracked Universe</h2>
              <p>Assets currently tagged and available for search and detail view.</p>
            </div>
          </div>

          {loading ? <p className="loading-note">Loading tracked assets...</p> : null}
          {error ? <p className="inline-alert">{error}</p> : null}

          {!loading && !error ? (
            <div className="asset-grid">
              {topAssets.map((asset) => (
                <article key={asset.id} className="asset-card">
                  <h3>{asset.slug}</h3>
                  <p>Monitored for social sentiment and topic scoring.</p>
                </article>
              ))}
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}

export default DashboardPage;
