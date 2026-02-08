import React from 'react';
import { Link } from 'react-router-dom';

function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer-inner page-wrap">
        <span>AlphaOne (c) 2026. Live market sentiment intelligence.</span>
        <div className="footer-links">
          <Link to="/">Home</Link>
          <Link to="/sentiment-summary">Sentiment Dashboard</Link>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
