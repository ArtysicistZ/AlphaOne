import React from 'react';

const formatTimestamp = (timestamp) => {
  if (!timestamp) return 'Unknown time';

  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return 'Unknown time';
  }

  return parsed.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const formatScore = (score) => {
  if (typeof score !== 'number' || Number.isNaN(score)) {
    return 'N/A';
  }

  return score > 0 ? `+${score.toFixed(3)}` : score.toFixed(3);
};

const toneClass = (sentiment) => {
  if (sentiment === 'POSITIVE' || sentiment === 'BULLISH') return 'is-positive';
  if (sentiment === 'NEGATIVE' || sentiment === 'BEARISH') return 'is-negative';
  return 'is-neutral';
};

const EvidenceList = ({ evidence }) => {
  const rows = Array.isArray(evidence) ? evidence : [];

  if (rows.length === 0) {
    return <p className="empty-state">No evidence available for this asset yet.</p>;
  }

  return (
    <ul className="evidence-list">
      {rows.map((item, index) => {
        const sentiment = (item.sentimentLabel ?? item.sentiment ?? 'NEUTRAL').toUpperCase();
        const text = item.relevantText ?? item.text ?? 'No text available.';
        const score = Number(item.sentimentScore);
        const key = item.id ?? `${sentiment}-${index}`;

        return (
          <li key={key} className="evidence-item">
            <div className="evidence-meta">
              <span className={`evidence-badge ${toneClass(sentiment)}`}>{sentiment}</span>
              <span className="evidence-score">{formatScore(score)}</span>
            </div>
            <p className="evidence-text">{text}</p>
            <div className="evidence-time">{formatTimestamp(item.createdAt ?? item.timestamp)}</div>
          </li>
        );
      })}
    </ul>
  );
};

export default EvidenceList;
