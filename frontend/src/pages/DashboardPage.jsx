import React, { useState, useEffect } from 'react';
import { getTrackedAssets } from '../api/sentimentApi';

function DashboardPage() {
  // 1. Create a "state" variable to hold our list of assets
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);

  // 2. Use "useEffect" to fetch data when the page loads
  useEffect(() => {
    // Define an async function inside
    const fetchAssets = async () => {
      try {
        const data = await getTrackedAssets();
        setAssets(data); // Save the data from the API into our state
      } catch (error) {
        console.error("Error fetching assets:", error);
      } finally {
        setLoading(false); // Stop loading, even if there was an error
      }
    };

    fetchAssets(); // Call the function
  }, []); // The empty array [] means "run this only once"

  // 3. Render the component's HTML
  if (loading) {
    return <div>Loading data...</div>;
  }

  return (
    <div>
      <h1>Tracked Assets</h1>
      <p>Data fetched live from your FastAPI backend:</p>
      <ul>
        {/* 4. Loop over the assets and display them in a list */}
        {assets.map((asset) => (
          <li key={asset.id}>{asset.slug}</li>
        ))}
      </ul>
    </div>
  );
}

export default DashboardPage;