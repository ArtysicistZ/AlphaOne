import React from 'react';

const EvidenceList = ({ evidence }) => {
    if (!evidence || evidence.length === 0) {
        return (
            <div className="p-8 text-center text-sm text-gray-400">
                Waiting for incoming signals...
            </div>
        );
    }

    return (
        <div className="divide-y divide-gray-100">
            {evidence.map((item, index) => {
                const isPositive = item.sentiment === 'POSITIVE';

                return (
                    <div key={index} className="p-4 hover:bg-gray-50 transition-colors">
                        <div className="flex justify-between items-start mb-2">
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${isPositive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                                }`}>
                                {item.sentiment}
                            </span>
                            <span className="text-xs text-gray-400">
                                {item.source || 'Reddit'}
                            </span>
                        </div>

                        <p className="text-sm text-gray-600 leading-relaxed mb-2">
                            {item.text}
                        </p>

                        <div className="text-right">
                            <span className="text-[10px] text-gray-400 font-medium">
                                {item.timestamp || 'Just now'}
                            </span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default EvidenceList;
