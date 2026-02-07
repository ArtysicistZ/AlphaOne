import React from 'react';

function Footer() {
  // This footer will be at the bottom of all your pages
  return (
    <footer style={{ padding: '20px', textAlign: 'center', opacity: 0.7 }}>

      <p>alphaone © 2025</p>

      {/* This is the HTML code from your DNS provider, adapted for React.
        - "target" opens the link in a new tab.
        - "rel" is for security.
      */}
      <a href="http://dnsexit.com" target="_blank" rel="noopener noreferrer">
        <img 
          src="http://dnsexit.com/images/dns.gif" 
          style={{ border: 0 }} 
          alt="DNS Powered by DNSEXIT.COM" 
        />
      </a>

    </footer>
  );
}

export default Footer;