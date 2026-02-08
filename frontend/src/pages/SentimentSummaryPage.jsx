import React, { useState, useEffect, useMemo } from 'react';
import WordCloud from 'react-d3-cloud';
import { 
  getWordCloudData, 
  getTopicSummary, 
  getSentimentEvidence,
  getTrackedAssets
} from '../api/sentimentApi';

// --- (Your SummaryWidget and TickerSearchResult components are unchanged) ---
function SummaryWidget({ title, score }) {
  const scoreColor = score > 0 ? 'green' : score < 0 ? 'red' : 'gray';
  const scoreText = score.toFixed(4);
  return (
    <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px', minWidth: '200px', textAlign: 'center' }}>
      <h3 style={{ margin: 0, color: '#555' }}>{title}</h3>
      <h2 style={{ color: scoreColor, margin: '10px 0 0 0' }}>{scoreText}</h2>
    </div>
  );
}
function TickerSearchResult({ ticker, evidence, summary }) {
  if (!ticker || !summary) return null;
  const scoreColor = summary.averageScore > 0 ? 'green' : summary.averageScore < 0 ? 'red' : 'gray';
  return (
    <div style={{ 
        marginTop: '20px', 
        padding: '15px', 
        border: '1px solid #eee',
        maxWidth: '650px',  // Sets a max width for this box
        margin: '0 auto'    // Re-centers the box within the page
    }}>
      <h3>Results for: {ticker}</h3>
      <p style={{fontSize: '1.2rem', fontWeight: 'bold'}}>
        Today's Average Score: 
        <span style={{ color: scoreColor, marginLeft: '10px' }}>
          {(summary.averageScore ?? 0).toFixed(4)}
        </span>
      </p>
      <h4>Sample Sentences (Evidence):</h4>
      <ul style={{ listStyle: 'none', padding: 0, textAlign: 'left' }}>
        {evidence.map((item) => (
          <li key={item.id} style={{ borderBottom: '1px solid #ddd', padding: '10px 0' }}>
            <strong style={{ color: (item.sentimentLabel ?? '') === 'positive' ? 'green' : item.sentimentLabel === 'negative' ? 'red' : 'gray', marginRight: '10px' }}>
              [{(item.sentimentLabel ?? 'neutral').toUpperCase()}]
            </strong> 
            {item.relevantText}
          </li>
        ))}
      </ul>
    </div>
  );
}
// --- (End of unchanged components) ---


// This is your main page component
function SentimentSummaryPage() {
  // State for your page sections
  const [macroScore, setMacroScore] = useState(0);
  const [techScore, setTechScore] = useState(0);
  const [wordCloudData, setWordCloudData] = useState([]);
  const [allTopics, setAllTopics] = useState([]); // <-- MODIFIED: New state for dropdown

  // State for your ticker search
  const [searchTicker, setSearchTicker] = useState('');
  const [searchedTicker, setSearchedTicker] = useState(null);
  const [searchSummary, setSearchSummary] = useState(null);
  const [searchEvidence, setSearchEvidence] = useState([]);

  // Fetch summary data when the page loads
  useEffect(() => {
    // 1. Fetch general topics
    getTopicSummary('MACRO').then(data => setMacroScore(data.averageScore)).catch(err => console.error("Could not fetch MACRO", err));
    getTopicSummary('TECHNOLOGY').then(data => setTechScore(data.averageScore)).catch(err => console.error("Could not fetch TECHNOLOGY", err));

    // 2. Fetch word cloud data
    getWordCloudData().then(data => {setWordCloudData(data);}).catch(err => console.error("Could not fetch Word Cloud", err));
    
    // 3. <-- MODIFIED: Fetch all topics for the datalist
    getTrackedAssets().then(data => {setAllTopics(data);}).catch(err => console.error("Could not fetch assets", err));
    
  }, []); // The empty [] means this runs once on page load

  // ... (handleTickerSearch function is unchanged) ...
  const handleTickerSearch = async (e) => {
    e.preventDefault(); 
    if (!searchTicker) return; 
    try {
      setSearchedTicker(searchTicker);
      setSearchSummary(null);
      setSearchEvidence([]);
      const summaryData = await getTopicSummary(searchTicker);
      const evidenceData = await getSentimentEvidence(searchTicker);
      setSearchSummary(summaryData);
      setSearchEvidence(evidenceData);
    } catch (error) {
      console.error("Error searching for ticker:", error);
      alert("Could not find data for that ticker. (Check your topic_tagger.py)");
    }
  };
  
  // ... (wordCloudFontSize logic is unchanged) ...
  const minFontSize = 14;
  const maxFontSize = 100;
  const [minFreq, maxFreq] = useMemo(() => {
    if (!wordCloudData || wordCloudData.length === 0) {
      return [0, 0];
    }
    const values = wordCloudData.map(w => w.value);
    return [Math.min(...values), Math.max(...values)];
  }, [wordCloudData]);
  const wordCloudFontSize = (word) => {
    if (maxFreq === minFreq) {
      return (minFontSize + maxFontSize) / 2;
    }
    const normalizedValue = (word.value - minFreq) / (maxFreq - minFreq);
    return normalizedValue * (maxFontSize - minFontSize) + minFontSize;
  };
  // ... (End of unchanged logic) ...

  // --- NEW RENDER STRUCTURE ---
  return (
    <div>
      {/* --- 1. FULL-WIDTH "COVER" SECTION --- */}
      <div style={{
        width: '100%',
        backgroundColor: '#1a202c', // Dark background
        color: 'white',
        padding: '1.5rem 0', // Padding top/bottom, 0 left/right
        textAlign: 'center'
      }}>
        <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Today's Keywords</h3>
        
        {/* This div centers the word cloud and handles scrolling */}
        <div style={{ maxWidth: '650px', margin: '1rem auto 0 auto', overflowX: 'auto' }}>
          {wordCloudData.length > 0 ? (
            <WordCloud
              data={wordCloudData}
              width={650}
              height={300}
              font="Arial"
              fontSize={wordCloudFontSize}
              spiral="archimedean" // <-- This makes it dense in the center
              rotate={0}
              padding={1}
            />
          ) : (
            <p>Loading word cloud...</p>
          )}
        </div>
      </div>

      {/* --- 2. CENTERED CONTENT SECTION --- */}
      {/* This div wraps all your other content and centers it */}
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '1.5rem',
        textAlign: 'center'
      }}>

        {/* --- Summary Widgets --- */}
        <h2>Sentiment Summary</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'center' }}>
          <SummaryWidget title="Macro Sentiment" score={macroScore} />
          <SummaryWidget title="Technology Sentiment" score={techScore} />
        </div>

        {/* --- Ticker Search Section --- */}
        <div style={{ marginTop: '30px' }}>
          <h3>Get Detailed Ticker Sentiment</h3>
          <form onSubmit={handleTickerSearch}>
            
            {/* --- MODIFIED: Added the 'list' prop to the input --- */}
            <input 
              type="text" 
              placeholder="Enter ticker (e.g., NVDA)" 
              value={searchTicker}
              onChange={(e) => setSearchTicker(e.target.value.toUpperCase())}
              style={{ padding: '10px', fontSize: '1rem', minWidth: '250px' }}
              list="topics-list" // <-- Links to the datalist below
            />
            
            {/* --- MODIFIED: Added the datalist element --- */}
            <datalist id="topics-list">
              {allTopics.map((topic) => (
                <option key={topic.id} value={topic.slug} />
              ))}
            </datalist>

            <button type="submit" style={{ padding: '10px', fontSize: '1rem', marginLeft: '10px' }}>
              Search
            </button>
          </form>
          
          <TickerSearchResult 
            ticker={searchedTicker} 
            summary={searchSummary}
            evidence={searchEvidence} 
          />
        </div>
        
      </div>
    </div>
  );
}

export default SentimentSummaryPage;