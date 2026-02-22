import apiClient from './index.js';

// --- This function is for your dropdown menu ---
export const getTrackedAssets = async () => {
  const response = await apiClient.get('/assets/tracked');
  return response.data;
};

// --- These two are for your "Ticker Search" ---
export const getSentimentForTickerChart = async (ticker) => {
  const response = await apiClient.get(`/signals/social-sentiment/${ticker}/daily`);
  return response.data;
};

export const getSentimentEvidence = async (ticker) => {
  const response = await apiClient.get(`/signals/social-sentiment/${ticker}/evidence`);
  return response.data;
};

export const getTopicSummary = async (topicSlug) => {
  const response = await apiClient.get(`/signals/social-sentiment/summary/${topicSlug}`);
  return response.data; // Returns { day: "...", average_score: 0.123 }
};

export const getWordCloudData = async () => {
  const response = await apiClient.get('/signals/social-sentiment/wordcloud');
  return response.data; // Returns [{ text: "word", value: 10 }, ...]
};

export const runInference = async (text, targets) => {
  const response = await apiClient.post('/inference', { text, targets });
  return response.data;
};