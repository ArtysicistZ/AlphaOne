import React, { useEffect, useMemo, useState } from 'react';
import WordCloud from 'react-d3-cloud';
import Sidebar from '../components/Sidebar';
import SentimentChart from '../components/SentimentChart';
import EvidenceList from '../components/EvidenceList';
import {
  getWordCloudData,
  getTopicSummary,
  getSentimentEvidence,
  getTrackedAssets,
  getSentimentForTickerChart,
} from '../api/sentimentApi';

const getScore = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const formatScore = (score) => {
  const parsed = getScore(score);
  return parsed > 0 ? `+${parsed.toFixed(3)}` : parsed.toFixed(3);
};

const scoreToneClass = (score) => {
  const parsed = getScore(score);
  if (parsed > 0.05) return 'is-positive';
  if (parsed < -0.05) return 'is-negative';
  return 'is-neutral';
};

function SummaryCard({ title, score, subtitle }) {
  return (
    <article className={`summary-card panel ${scoreToneClass(score)}`}>
      <p className="summary-label">{title}</p>
      <p className="summary-score">{formatScore(score)}</p>
      <p className="summary-subtitle">{subtitle}</p>
    </article>
  );
}

function SentimentSummaryPage() {
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState('');
  const [searchTicker, setSearchTicker] = useState('');

  const [macroScore, setMacroScore] = useState(0);
  const [techScore, setTechScore] = useState(0);
  const [assetScore, setAssetScore] = useState(0);

  const [wordCloudData, setWordCloudData] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [evidence, setEvidence] = useState([]);

  const [cloudWidth, setCloudWidth] = useState(540);
  const [isBootLoading, setIsBootLoading] = useState(true);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [bootError, setBootError] = useState('');
  const [detailError, setDetailError] = useState('');

  useEffect(() => {
    const updateCloudWidth = () => {
      if (window.innerWidth < 640) {
        setCloudWidth(300);
      } else if (window.innerWidth < 900) {
        setCloudWidth(420);
      } else {
        setCloudWidth(540);
      }
    };

    updateCloudWidth();
    window.addEventListener('resize', updateCloudWidth);
    return () => window.removeEventListener('resize', updateCloudWidth);
  }, []);

  useEffect(() => {
    let isActive = true;

    const loadBootstrap = async () => {
      setIsBootLoading(true);
      setBootError('');

      const [assetRes, macroRes, techRes, cloudRes] = await Promise.allSettled([
        getTrackedAssets(),
        getTopicSummary('MACRO'),
        getTopicSummary('TECHNOLOGY'),
        getWordCloudData(),
      ]);

      if (!isActive) return;

      const nextAssets = assetRes.status === 'fulfilled' && Array.isArray(assetRes.value) ? assetRes.value : [];
      setAssets(nextAssets);

      if (macroRes.status === 'fulfilled') {
        setMacroScore(getScore(macroRes.value?.averageScore));
      }

      if (techRes.status === 'fulfilled') {
        setTechScore(getScore(techRes.value?.averageScore));
      }

      if (cloudRes.status === 'fulfilled' && Array.isArray(cloudRes.value)) {
        setWordCloudData(cloudRes.value);
      }

      if (nextAssets.length > 0) {
        const firstAsset = nextAssets[0].slug;
        setSelectedAsset(firstAsset);
        setSearchTicker(firstAsset);
      }

      if (
        assetRes.status === 'rejected' ||
        macroRes.status === 'rejected' ||
        techRes.status === 'rejected' ||
        cloudRes.status === 'rejected'
      ) {
        setBootError('Some global metrics are temporarily unavailable.');
      }

      setIsBootLoading(false);
    };

    loadBootstrap().catch((error) => {
      console.error('Bootstrap error:', error);
      if (isActive) {
        setBootError('Unable to load dashboard overview.');
        setIsBootLoading(false);
      }
    });

    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedAsset) return;

    let isActive = true;

    const loadAssetDetails = async () => {
      setIsDetailLoading(true);
      setDetailError('');

      const [chartRes, evidenceRes, summaryRes] = await Promise.allSettled([
        getSentimentForTickerChart(selectedAsset),
        getSentimentEvidence(selectedAsset),
        getTopicSummary(selectedAsset),
      ]);

      if (!isActive) return;

      if (chartRes.status === 'fulfilled' && Array.isArray(chartRes.value)) {
        setChartData(chartRes.value);
      } else {
        setChartData([]);
      }

      if (evidenceRes.status === 'fulfilled' && Array.isArray(evidenceRes.value)) {
        setEvidence(evidenceRes.value);
      } else {
        setEvidence([]);
      }

      if (summaryRes.status === 'fulfilled') {
        setAssetScore(getScore(summaryRes.value?.averageScore));
      } else {
        setAssetScore(0);
      }

      if (chartRes.status === 'rejected' || evidenceRes.status === 'rejected' || summaryRes.status === 'rejected') {
        setDetailError(`Some details for ${selectedAsset} could not be loaded.`);
      }

      setIsDetailLoading(false);
    };

    loadAssetDetails().catch((error) => {
      console.error('Asset details error:', error);
      if (isActive) {
        setChartData([]);
        setEvidence([]);
        setDetailError(`Unable to load sentiment details for ${selectedAsset}.`);
        setIsDetailLoading(false);
      }
    });

    return () => {
      isActive = false;
    };
  }, [selectedAsset]);

  const handleTickerSearch = (event) => {
    event.preventDefault();
    const normalized = searchTicker.trim().toUpperCase();
    if (!normalized) return;
    setSelectedAsset(normalized);
    setSearchTicker(normalized);
  };

  const handleAssetSelect = (assetSlug) => {
    setSelectedAsset(assetSlug);
    setSearchTicker(assetSlug);
  };

  const [minFreq, maxFreq] = useMemo(() => {
    if (wordCloudData.length === 0) {
      return [0, 0];
    }

    const values = wordCloudData.map((word) => Number(word.value));
    return [Math.min(...values), Math.max(...values)];
  }, [wordCloudData]);

  const wordCloudFontSize = (word) => {
    const value = Number(word.value);
    if (maxFreq === minFreq) {
      return 36;
    }

    const normalized = (value - minFreq) / (maxFreq - minFreq);
    return normalized * 56 + 16;
  };

  return (
    <div className="sentiment-page">
      <div className="page-wrap">
        <section className="sentiment-header panel">
          <div>
            <p className="sentiment-kicker">Live Analytics</p>
            <h1 className="sentiment-title">Sentiment Command Center</h1>
            <p className="sentiment-subtitle">
              Track market mood shifts with trend lines, keyword pressure, and sentence-level evidence.
            </p>
          </div>

          <form onSubmit={handleTickerSearch} className="ticker-form">
            <input
              className="text-input"
              type="text"
              placeholder="Enter ticker (e.g. NVDA)"
              value={searchTicker}
              onChange={(event) => setSearchTicker(event.target.value.toUpperCase())}
              list="tracked-assets"
            />
            <datalist id="tracked-assets">
              {assets.map((asset) => (
                <option key={asset.id} value={asset.slug} />
              ))}
            </datalist>
            <button type="submit" className="btn-action">
              Load Asset
            </button>
          </form>
        </section>

        {bootError ? <p className="inline-alert">{bootError}</p> : null}

        <section className="summary-grid">
          <SummaryCard title="Selected Asset" score={assetScore} subtitle={selectedAsset || 'No asset selected'} />
          <SummaryCard title="Macro Sentiment" score={macroScore} subtitle="Cross-market narrative pressure" />
          <SummaryCard title="Technology Sentiment" score={techScore} subtitle="Sector-wide signal direction" />
        </section>

        {isBootLoading ? (
          <div className="loading-card panel">Loading dashboard data...</div>
        ) : (
          <section className="sentiment-main">
            <Sidebar assets={assets} onSelectAsset={handleAssetSelect} selectedAsset={selectedAsset} />

            <div className="sentiment-content">
              <article className="chart-panel panel">
                <div className="panel-head">
                  <div>
                    <h2>Trend Overview</h2>
                    <p>Recent daily sentiment averages for {selectedAsset || 'your selected asset'}.</p>
                  </div>
                  <span className={`tone-pill ${scoreToneClass(assetScore)}`}>{formatScore(assetScore)}</span>
                </div>

                <div className="chart-shell">
                  {isDetailLoading ? (
                    <p className="loading-note">Refreshing sentiment trend...</p>
                  ) : (
                    <SentimentChart data={chartData} ticker={selectedAsset} />
                  )}
                </div>
              </article>

              {detailError ? <p className="inline-alert">{detailError}</p> : null}

              <div className="sentiment-lower">
                <article className="word-cloud-panel panel">
                  <div className="panel-head">
                    <div>
                      <h2>Keyword Pressure</h2>
                      <p>Most repeated words from today's sentiment stream.</p>
                    </div>
                  </div>

                  <div className="word-cloud-shell">
                    {wordCloudData.length > 0 ? (
                      <WordCloud
                        data={wordCloudData}
                        width={cloudWidth}
                        height={280}
                        font="Space Grotesk"
                        fontSize={wordCloudFontSize}
                        rotate={0}
                        spiral="archimedean"
                        padding={2}
                      />
                    ) : (
                      <p className="loading-note">No keyword cloud data available.</p>
                    )}
                  </div>
                </article>

                <article className="evidence-panel panel">
                  <div className="panel-head">
                    <div>
                      <h2>Evidence Feed</h2>
                      <p>Sentence-level mentions used for model scoring.</p>
                    </div>
                  </div>

                  {isDetailLoading ? (
                    <p className="loading-note">Loading evidence feed...</p>
                  ) : (
                    <EvidenceList evidence={evidence} />
                  )}
                </article>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

export default SentimentSummaryPage;
