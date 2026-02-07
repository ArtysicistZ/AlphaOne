import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';

function Layout() {
  return (
    // This div wrapper is gone. Navbar and Footer are full-width.
    <> 
      <Navbar />

      {/* This <main> tag is now "dumb." It has no max-width or margin.
          This gives your pages full control over the layout. */}
      <main style={{
        flex: 1, // Makes the main content grow to fill the space
        width: '100%'
      }}>
        <Outlet />
      </main>

      <Footer />
    </>
  );
}

export default Layout;