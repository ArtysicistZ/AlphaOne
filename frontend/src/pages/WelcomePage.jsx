import React from 'react';
import { Link } from 'react-router-dom';

const WelcomePage = () => {
    return (
        <div className="min-h-full flex flex-col bg-white">

            {/* Simple Navbar */}
            <nav className="flex justify-between items-center px-8 py-6 max-w-7xl mx-auto w-full">
                <div className="text-2xl font-bold text-gray-900 tracking-tight">
                    Alpha<span className="text-indigo-600">One</span>
                </div>
                <div className="flex gap-4">
                    <Link to="/about" className="text-sm font-medium text-gray-500 hover:text-gray-900">About</Link>
                    <Link to="/pricing" className="text-sm font-medium text-gray-500 hover:text-gray-900">Pricing</Link>
                    <Link to="/login" className="text-sm font-medium text-gray-500 hover:text-gray-900">Log in</Link>
                </div>
            </nav>

            {/* Hero Section */}
            <main className="flex-1 flex flex-col items-center justify-center text-center px-4 py-20">
                <div className="max-w-3xl mx-auto">
                    <span className="inline-block px-4 py-1.5 rounded-full bg-indigo-50 text-indigo-700 text-sm font-semibold mb-6">
                        v1.0 Now Available
                    </span>
                    <h1 className="text-5xl md:text-6xl font-extrabold text-gray-900 tracking-tight mb-8 leading-tight">
                        Market sentiment analysis <br />
                        <span className="text-indigo-600">reimagined for clarity.</span>
                    </h1>
                    <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto leading-relaxed">
                        Stop guessing. Start knowing. AlphaOne aggregates millions of social signals to give you a clear interface for market direction.
                    </p>

                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Link to="/terminal" className="px-8 py-4 rounded-lg bg-indigo-600 text-white font-bold text-lg hover:bg-indigo-700 transition-colors shadow-lg shadow-indigo-200">
                            Launch Dashboard
                        </Link>
                        <a href="#" className="px-8 py-4 rounded-lg bg-white text-gray-700 border border-gray-200 font-bold text-lg hover:bg-gray-50 transition-colors">
                            View Documentation
                        </a>
                    </div>
                </div>

                {/* Feature preview */}
                <div className="mt-20 max-w-5xl mx-auto w-full grid grid-cols-1 md:grid-cols-3 gap-8 text-left px-4">
                    <div className="p-6 rounded-2xl bg-gray-50 border border-gray-100">
                        <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center mb-4 text-2xl">⚡</div>
                        <h3 className="text-lg font-bold text-gray-900 mb-2">Real-time Stream</h3>
                        <p className="text-gray-600">Direct integration with high-volume discussion boards.</p>
                    </div>
                    <div className="p-6 rounded-2xl bg-gray-50 border border-gray-100">
                        <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center mb-4 text-2xl">🧠</div>
                        <h3 className="text-lg font-bold text-gray-900 mb-2">Neural Analysis</h3>
                        <p className="text-gray-600">Advanced NLP models scoring sentiment instantly.</p>
                    </div>
                    <div className="p-6 rounded-2xl bg-gray-50 border border-gray-100">
                        <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center mb-4 text-2xl">📈</div>
                        <h3 className="text-lg font-bold text-gray-900 mb-2">Clear Metrics</h3>
                        <p className="text-gray-600">Simple, actionable velocity scores and trend lines.</p>
                    </div>
                </div>
            </main>

            <footer className="py-8 text-center text-sm text-gray-400 border-t border-gray-100">
                © 2026 AlphaOne Systems. All rights reserved.
            </footer>
        </div>
    );
};

export default WelcomePage;
