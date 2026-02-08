import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const SentimentChart = ({ data, ticker }) => {
  const points = Array.isArray(data) ? data : [];

  if (points.length === 0) {
    return <p className="empty-state">No trend data available for {ticker}.</p>;
  }

  const scoreSeries = points.map((item) => Number(item.averageScore ?? item.score ?? 0));
  const averageScore = scoreSeries.reduce((sum, value) => sum + value, 0) / scoreSeries.length;
  const lineColor = averageScore >= 0 ? '#0f9d76' : '#d54f45';
  const fillColor = averageScore >= 0 ? 'rgba(15, 157, 118, 0.18)' : 'rgba(213, 79, 69, 0.15)';

  const chartData = {
    labels: points.map((item) => item.date ?? item.day ?? ''),
    datasets: [
      {
        label: 'Average Sentiment',
        data: scoreSeries,
        fill: true,
        backgroundColor: fillColor,
        borderColor: lineColor,
        borderWidth: 2.5,
        tension: 0.35,
        pointRadius: 2,
        pointHoverRadius: 5,
        pointHoverBorderWidth: 2,
        pointHoverBackgroundColor: lineColor,
        pointHoverBorderColor: '#ffffff',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: '#ffffff',
        titleColor: '#102032',
        bodyColor: '#2f4458',
        borderColor: '#d4e1eb',
        borderWidth: 1,
        padding: 10,
        callbacks: {
          label: (context) => `Score: ${Number(context.raw).toFixed(3)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#667b8d',
          maxTicksLimit: 7,
          font: {
            size: 11,
          },
        },
        grid: {
          display: false,
        },
      },
      y: {
        min: -1,
        max: 1,
        ticks: {
          color: '#667b8d',
          font: {
            size: 11,
          },
          callback: (value) => Number(value).toFixed(1),
        },
        grid: {
          color: 'rgba(212, 225, 235, 0.8)',
        },
      },
    },
  };

  return <Line data={chartData} options={options} />;
};

export default SentimentChart;
