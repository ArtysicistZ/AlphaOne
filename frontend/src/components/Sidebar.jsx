import React from 'react';

const Sidebar = ({ assets, onSelectAsset, selectedAsset }) => {
    if (!assets) return null;

    return (
        <aside
            className="h-full flex flex-col bg-white border-r border-border-default z-20 shadow-sm"
            style={{ width: '260px', flexShrink: 0 }}
        >
            {/* Header */}
            <div className="p-6 border-b border-border-default bg-gray-50/50">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">My Watchlist</h3>
                <p className="text-xs text-muted flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                    {assets.length} assets tracked
                </p>
            </div>

            {/* Scrollable List */}
            <div className="flex-1 overflow-y-auto w-full py-4 space-y-1">
                {assets.map((asset) => {
                    const isSelected = asset.slug === selectedAsset;
                    return (
                        <div
                            key={asset.id}
                            onClick={() => onSelectAsset(asset.slug)}
                            className={`relative px-6 py-3 cursor-pointer transition-colors flex items-center justify-between group
                    ${isSelected ? 'bg-indigo-50 border-r-4 border-indigo-600' : 'hover:bg-gray-50 border-r-4 border-transparent'}
                `}
                        >
                            <div className="flex flex-col">
                                <span className={`text-sm font-medium ${isSelected ? 'text-indigo-900' : 'text-gray-700'}`}>
                                    {asset.slug}
                                </span>
                                <span className="text-xs text-gray-400 group-hover:text-gray-500">NASDAQ / US</span>
                            </div>

                            {/* Arrow or subtle indicator */}
                            {isSelected && (
                                <span className="text-indigo-600 text-sm font-bold">→</span>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-border-default bg-gray-50 text-xs text-gray-400">
                <div className="flex justify-between items-center">
                    <span>Server Status</span>
                    <span className="text-green-600 font-medium bg-green-100 px-2 py-0.5 rounded-full text-[10px]">Operational</span>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
