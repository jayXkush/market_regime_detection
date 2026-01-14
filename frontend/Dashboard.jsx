import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar,
  LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell
} from 'recharts';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Dashboard() {
  const [regimes, setRegimes] = useState([]);
  const [modelEval, setModelEval] = useState([]);
  const [transitions, setTransitions] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentRegime, setCurrentRegime] = useState(null);
  const [fetchingCurrent, setFetchingCurrent] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch regime characteristics
      const regimeRes = await fetch(`${API_URL}/regime-characteristics`);
      if (regimeRes.ok) {
        const data = await regimeRes.json();
        setRegimes(data.regimes || []);
      }
      
      // Fetch model evaluation
      const modelRes = await fetch(`${API_URL}/model-evaluation`);
      if (modelRes.ok) {
        const data = await modelRes.json();
        setModelEval(data.all_models || []);
      }
      
      // Fetch transition matrix
      const transRes = await fetch(`${API_URL}/transition-matrix`);
      if (transRes.ok) {
        const data = await transRes.json();
        setTransitions(data.data || {});
      }
      
      setError(null);
    } catch (err) {
      setError(`Failed to fetch data: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchCurrentRegime = async () => {
    try {
      setFetchingCurrent(true);
      const res = await fetch(`${API_URL}/current-regime`);
      
      if (res.ok) {
        const data = await res.json();
        setCurrentRegime(data);
      } else {
        alert('Failed to fetch current market regime. Make sure backend is running.');
      }
    } catch (err) {
      alert(`Error: ${err.message}`);
      console.error(err);
    } finally {
      setFetchingCurrent(false);
    }
  };

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-900">
        <div className="text-white text-2xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      {/* Header */}
      <div className="mb-12">
        <h1 className="text-5xl font-bold text-white mb-2">
          🎯 Market Regime Detection
        </h1>
        <p className="text-gray-400 text-lg">
          Unsupervised clustering of BNB/FDUSD high-frequency trading data
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-900/20 border border-red-700 text-red-200 p-4 rounded-lg mb-8">
          ⚠️ {error}
        </div>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-12">
        <div className="bg-slate-700/50 p-6 rounded-lg border border-slate-600">
          <div className="text-gray-300 text-sm mb-2">Total Samples</div>
          <div className="text-3xl font-bold text-blue-400">1,152</div>
        </div>
        <div className="bg-slate-700/50 p-6 rounded-lg border border-slate-600">
          <div className="text-gray-300 text-sm mb-2">Regimes Found</div>
          <div className="text-3xl font-bold text-green-400">4</div>
        </div>
        <div className="bg-slate-700/50 p-6 rounded-lg border border-slate-600">
          <div className="text-gray-300 text-sm mb-2">Best Silhouette</div>
          <div className="text-3xl font-bold text-purple-400">0.512</div>
        </div>
        <div className="bg-slate-700/50 p-6 rounded-lg border border-slate-600">
          <div className="text-gray-300 text-sm mb-2">Algorithm</div>
          <div className="text-2xl font-bold text-yellow-400">HDBSCAN</div>
        </div>
      </div>

      {/* Regime Characteristics */}
      <div className="mb-12">
        <h2 className="text-3xl font-bold text-white mb-6">📊 Market Regimes</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {regimes.map((regime, idx) => (
            <div key={idx} className="bg-slate-700/60 border border-slate-600 p-6 rounded-lg hover:border-slate-500 transition">
              <div className="flex items-center mb-4">
                <div className="w-4 h-4 rounded-full mr-3" style={{ backgroundColor: COLORS[idx] }}></div>
                <h3 className="text-xl font-bold text-white">Regime {regime.regime}</h3>
              </div>
              <p className="text-gray-300 mb-4">{regime.name}</p>
              <div className="space-y-2 text-sm text-gray-400">
                <div>📊 Samples: <span className="text-white font-bold">{regime.size?.toFixed(2)}%</span></div>
                <div>📈 Volatility: <span className="text-white font-bold">{regime.volatility?.toFixed(2)}%</span></div>
                <div>💹 Returns: <span className="text-white font-bold">{regime.returns?.toFixed(6)}</span></div>
                <div>📦 Volume: <span className="text-white font-bold">{regime.volume?.toFixed(2)}</span></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Evaluation */}
      <div className="mb-12">
        <h2 className="text-3xl font-bold text-white mb-6">📈 Model Comparison</h2>
        <div className="bg-slate-700/60 border border-slate-600 p-8 rounded-lg">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={modelEval}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="model" stroke="#cbd5e1" />
              <YAxis stroke="#cbd5e1" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Legend />
              <Bar dataKey="silhouette_score" fill="#3b82f6" name="Silhouette Score" />
              <Bar dataKey="davies_bouldin_score" fill="#ef4444" name="Davies-Bouldin (×0.1)" />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-6 p-4 bg-slate-600/30 rounded border border-slate-600">
            <p className="text-gray-300 text-sm">
              <strong>Winner: HDBSCAN (mcs=10)</strong> - Achieved best separation with automatic cluster detection and noise labeling.
            </p>
          </div>
        </div>
      </div>

      {/* Transition Matrix */}
      <div className="mb-12">
        <h2 className="text-3xl font-bold text-white mb-6">🔄 Regime Transitions</h2>
        <div className="bg-slate-700/60 border border-slate-600 p-8 rounded-lg overflow-x-auto">
          <p className="text-gray-300 mb-4">Probability of transitioning from one regime to another</p>
          <table className="w-full text-sm text-gray-300">
            <thead>
              <tr className="border-b border-slate-600">
                <th className="text-left p-3 text-white">From → To</th>
                {Object.keys(transitions).length > 0 && Object.keys(Object.values(transitions)[0] || {}).map(col => (
                  <th key={col} className="text-center p-3 text-white">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(transitions).map(([from, probs]) => (
                <tr key={from} className="border-b border-slate-600/50 hover:bg-slate-600/30 transition">
                  <td className="p-3 font-bold text-white">{from}</td>
                  {Object.values(probs).map((prob, idx) => (
                    <td key={idx} className="text-center p-3 text-white">
                      {typeof prob === 'number' ? (prob * 100).toFixed(1) : prob}%
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Live Market Regime */}
      <div className="mb-12">
        <h2 className="text-3xl font-bold text-white mb-6">🔴 Live Market Regime</h2>
        <div className="bg-slate-700/60 border border-slate-600 p-8 rounded-lg">
          <p className="text-gray-300 mb-6">
            Analyze current BNB/FDUSD market conditions using live Binance data
          </p>
          
          <button
            onClick={fetchCurrentRegime}
            disabled={fetchingCurrent}
            className={`px-8 py-4 rounded-lg font-bold text-white text-lg transition ${
              fetchingCurrent
                ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700'
            }`}
          >
            {fetchingCurrent ? '⏳ Analyzing...' : '🎯 Check Current Market Regime'}
          </button>

          {currentRegime && (
            <div className="mt-8 p-6 bg-slate-800/80 border border-slate-600 rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div>
                  <p className="text-gray-400 text-sm mb-1">Current Regime</p>
                  <div className="flex items-center">
                    <div 
                      className="w-6 h-6 rounded-full mr-3"
                      style={{ backgroundColor: COLORS[currentRegime.regime] || '#gray' }}
                    ></div>
                    <p className="text-3xl font-bold text-white">
                      Regime {currentRegime.regime}
                    </p>
                  </div>
                </div>
                
                <div>
                  <p className="text-gray-400 text-sm mb-1">Current Price</p>
                  <p className="text-3xl font-bold text-green-400">
                    ${currentRegime.current_price?.toFixed(4)}
                  </p>
                </div>
                
                <div>
                  <p className="text-gray-400 text-sm mb-1">Confidence</p>
                  <p className="text-3xl font-bold text-yellow-400">
                    {(currentRegime.confidence * 100).toFixed(0)}%
                  </p>
                </div>
              </div>

              <div className="space-y-4 border-t border-slate-600 pt-6">
                <div>
                  <p className="text-gray-400 text-sm mb-2">Market Interpretation</p>
                  <p className="text-white text-lg">{currentRegime.regime_name}</p>
                </div>
                
                <div>
                  <p className="text-gray-400 text-sm mb-2">💡 Trading Strategy</p>
                  <p className="text-green-400 text-lg font-medium">{currentRegime.strategy}</p>
                </div>
                
                <div className="flex justify-between items-center text-sm text-gray-400 pt-4 border-t border-slate-700">
                  <span>📊 {currentRegime.timeframe}</span>
                  <span>🕐 {new Date(currentRegime.timestamp).toLocaleString()}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="text-center py-8 border-t border-slate-700">
        <p className="text-gray-400 text-sm">
          Built by Jay Kushwaha
        </p>
        <p className="text-gray-500 text-xs mt-2">
          <a href="https://github.com/jayXkush/market_regime_detection" className="text-blue-400 hover:text-blue-300">
            View on GitHub
          </a>
        </p>
      </div>
    </div>
  );
}
