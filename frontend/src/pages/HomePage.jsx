import React from 'react';

function HomePage() {
  return (
    // This div applies centering *only* to the home page content
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '1.5rem',
      textAlign: 'center'
    }}>
      <h1>Welcome to alphaone</h1>
      <p>
        This is your all-in-one platform for quantitative analysis.
        Select a module from the navigation bar to get started.
      </p>
    </div>
  );
}

export default HomePage;