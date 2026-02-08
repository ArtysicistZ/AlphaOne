import React from 'react';
import { NavLink } from 'react-router-dom';

function Navbar() {
  return (
    <header className="top-nav">
      <nav className="top-nav-inner page-wrap" aria-label="Main navigation">
        <NavLink to="/" className="brand">
          Alpha<span className="brand-mark">One</span>
        </NavLink>

        <div className="nav-links">
          <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' is-active' : ''}`}>
            Welcome
          </NavLink>
          <NavLink to="/dashboard" className={({ isActive }) => `nav-link${isActive ? ' is-active' : ''}`}>
            Dashboard
          </NavLink>
          <NavLink to="/sentiment-summary" className={({ isActive }) => `nav-link${isActive ? ' is-active' : ''}`}>
            Sentiment
          </NavLink>
          <NavLink to="/architecture" className={({ isActive }) => `nav-link${isActive ? ' is-active' : ''}`}>
            Architecture
          </NavLink>
        </div>
      </nav>
    </header>
  );
}

export default Navbar;
