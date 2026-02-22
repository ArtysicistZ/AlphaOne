import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import SentimentSummaryPage from './pages/SentimentSummaryPage';
import DashboardPage from './pages/DashboardPage';
import ArchitecturePage from './pages/ArchitecturePage';
import PlaygroundPage from './pages/PlaygroundPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="sentiment-summary" element={<SentimentSummaryPage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="architecture" element={<ArchitecturePage />} />
        <Route path="playground" element={<PlaygroundPage />} />
      </Route>
    </Routes>
  );
}

export default App;
