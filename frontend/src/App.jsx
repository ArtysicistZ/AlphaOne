import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import SentimentSummaryPage from './pages/SentimentSummaryPage';
import DashboardPage from './pages/DashboardPage';
import ArchitecturePage from './pages/ArchitecturePage';

// (You can also delete App.css, as we aren't using its styles anymore)

function App() {
  return (
    <Routes>
      {/* This is a parent route. It tells React to use the <Layout /> component
        for all child routes (all pages on your site).
      */}
      <Route path="/" element={<Layout />}>
        {/* This is the default child route (path="/").
          It renders <HomePage /> inside the <Layout />'s <Outlet />
        */}
        <Route index element={<HomePage />} />
        
        {/* This is your first module's page.
          It renders <SentimentSummaryPage /> at the "/sentiment-summary" URL.
        */}
        <Route path="sentiment-summary" element={<SentimentSummaryPage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="architecture" element={<ArchitecturePage />} />
      </Route>
    </Routes>
  );
}

export default App;
