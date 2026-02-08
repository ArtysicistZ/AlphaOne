import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const SentimentChart = ({ data, ticker }) => {
    if (!data || data.length === 0) {
        return (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
                No chart data available for {ticker}
            </div>
        );
    }

    const chartData = {
        labels: data.map(d => d.date),
        datasets: [
            {
                label: 'Sentiment Score',
                data: data.map(d => d.score),
                fill: true,
                backgroundColor: 'rgba(79, 70, 229, 0.1)', // Indigo fill
                borderColor: '#4f46e5', // Indigo line
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#4f46e5',
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: '#fff',
                titleColor: '#111827',
                bodyColor: '#4b5563',
                borderColor: '#e5e7eb',
                borderWidth: 1,
                padding: 10,
                displayColors: false,
            },
        },
        scales: {
            x: {
                grid: {
                    display: false,
                },
                ticks: {
                    color: '#9ca3af',
                    font: { size: 10 },
                    maxTicksLimit: 6,
                },
            },
            y: {
                grid: {
                    color: '#f3f4f6',
                    borderDash: [5, 5],
                },
                ticks: {
                    color: '#9ca3af',
                    font: { size: 10 },
                },
                suggestedMin: -1,
                suggestedMax: 1,
            },
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false,
        },
    };

    return <Line data={chartData} options={options} />;
};

export default SentimentChart;
