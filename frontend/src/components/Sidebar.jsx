import React from 'react';

const Sidebar = ({ assets, onSelectAsset, selectedAsset }) => {
  if (!assets) return null;

  return (
    <aside className="watchlist panel">
      <div className="watchlist-head">
        <h3>Tracked Assets</h3>
        <p>{assets.length} symbols available</p>
      </div>

      {assets.length === 0 ? (
        <div className="watchlist-empty">No assets found.</div>
      ) : (
        <ul className="watchlist-list">
          {assets.map((asset) => {
            const isSelected = asset.slug === selectedAsset;
            return (
              <li key={asset.id}>
                <button
                  type="button"
                  onClick={() => onSelectAsset(asset.slug)}
                  className={`watchlist-btn${isSelected ? ' is-active' : ''}`}
                >
                  {asset.slug}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
};

export default Sidebar;
