import React from 'react';
import { Link } from 'react-router-dom'; // Import Link for SPA navigation

function Navbar() {
  // Basic styles to make it look like a nav bar
  const navStyle = {
    backgroundColor: '#1a202c',
    padding: '1rem',
    display: 'flex',
    gap: '20px',
    justifyContent: 'center' 
  };

  const linkStyle = {
    color: 'white',
    textDecoration: 'none',
    fontSize: '1rem',
    fontWeight: '500',
  };

  return (
    <nav style={navStyle}>
      <Link to="/" style={linkStyle}>
        Home
      </Link>
      <Link to="/sentiment-summary" style={linkStyle}>
        Sentiment Summary
      </Link>
      {/* These are placeholders for your future modules */}
      <Link to="/factor-analysis" style={{...linkStyle, opacity: 0.5}}>
        Factor Analysis (Future)
      </Link>
      <Link to="/portfolio-optimization" style={{...linkStyle, opacity: 0.5}}>
        Portfolio Optimization (Future)
      </Link>
    </nav>
  );
}

export default Navbar;