import React from 'react';

const Sidebar = ({ assets, onSelectAsset, selectedAsset }) => {
    if (!assets) return null;

    return (
        <aside
            style={{
                width: '240px',
                backgroundColor: 'var(--bg-card)',
                borderRight: '1px solid var(--border-default)',
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                flexShrink: 0,
                zIndex: 20
            }}
        >
            <div className="p-3 border-b border-border-default flex justify-between items-center bg-panel">
                <h3 className="text-secondary text-xs uppercase font-bold tracking-widest pl-1">Watchlist</h3>
                <span className="text-xs text-muted font-mono">{assets.length}</span>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar bg-black">
                {assets.map((asset) => {
                    const isSelected = asset.slug === selectedAsset;
                    return (
                        <div
                            key={asset.id}
                            onClick={() => onSelectAsset(asset.slug)}
                            className="group relative cursor-pointer transition-colors"
                            style={{
                                padding: '10px 12px',
                                borderLeft: isSelected ? '3px solid var(--color-primary)' : '3px solid transparent',
                                backgroundColor: isSelected ? 'var(--bg-hover)' : 'transparent',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                borderBottom: '1px solid var(--border-muted)'
                            }}
                        >
                            {/* Hover Effect Overlay */}
                            <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-5 pointer-events-none transition-opacity"></div>

                            <div>
                                <span className="font-mono font-bold text-sm block" style={{
                                    color: isSelected ? 'var(--text-white)' : 'var(--text-primary)'
                                }}>
                                    {asset.slug}
                                </span>
                                <span className="text-xs text-muted block mt-0.5">NASDAQ</span>
                            </div>

                            <div className="text-right">
                                {/* Fake spark text or percent change */}
                                <div className="font-mono text-xs" style={{
                                    color: Math.random() > 0.5 ? 'var(--color-positive)' : 'var(--color-negative)'
                                }}>
                                    {Math.random() > 0.5 ? '+' : '-'}{(Math.random() * 2).toFixed(2)}%
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Footer / Status */}
            <div className="p-3 border-t border-border-default bg-panel text-xs text-muted font-mono">
                <div className="flex items-center gap-sm">
                    <span className="w-1.5 h-1.5 rounded-full bg-positive animate-pulse"></span>
                    <span>DATA_STREAM: ON</span>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
