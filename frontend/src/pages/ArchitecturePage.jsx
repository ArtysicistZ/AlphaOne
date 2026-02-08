import React from 'react';

const TechBadge = ({ label }) => (
    <span className="inline-block px-3 py-1 bg-gray-100 text-gray-700 rounded-md text-xs font-semibold mr-2 mb-2 border border-gray-200">
        {label}
    </span>
);

const ArchitecturePage = () => {
    return (
        <div className="h-full w-full overflow-y-auto bg-gray-50 p-8">
            <div className="max-w-5xl mx-auto">
                <header className="mb-10 text-center">
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">System Architecture</h1>
                    <p className="text-gray-500">Technical overview of the AlphaOne platform</p>
                </header>

                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 mb-10">
                    <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                        <span className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center text-sm">1</span>
                        Data Flow Pipeline
                    </h2>

                    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 font-mono text-sm text-gray-700 text-center overflow-x-auto">
                        <pre>{`
[REDDIT STREAM] 
     |
     v
(Ingestion Service) --> [REDIS QUEUE] --> (Celery Workers)
                                            |
                                            v
                                       [POSTGRES DB] 
                                            |
                                            v
                                    (Spring Boot API)
                                            |
                                            v 
                                     [React Dashboard]
`}</pre>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
                        <h2 className="text-lg font-bold text-gray-800 mb-4">Frontend Stack</h2>
                        <TechBadge label="React 18" />
                        <TechBadge label="Vite" />
                        <TechBadge label="Tailwind CSS" />
                        <TechBadge label="Chart.js" />
                    </div>

                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
                        <h2 className="text-lg font-bold text-gray-800 mb-4">Backend Services</h2>
                        <TechBadge label="Java 21" />
                        <TechBadge label="Spring Boot 3.5" />
                        <TechBadge label="Python 3.11" />
                        <TechBadge label="Celery" />
                    </div>

                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
                        <h2 className="text-lg font-bold text-gray-800 mb-4">Data Persistence</h2>
                        <TechBadge label="PostgreSQL 16" />
                        <TechBadge label="Redis" />
                    </div>

                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
                        <h2 className="text-lg font-bold text-gray-800 mb-4">Infrastructure</h2>
                        <TechBadge label="Docker Compose" />
                        <TechBadge label="Nginx" />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ArchitecturePage;
