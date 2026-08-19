import { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import Playground from './pages/Playground';
import LatentExplorer from './pages/LatentExplorer';
import AttentionViz from './pages/AttentionViz';
import TrainingDashboard from './pages/TrainingDashboard';
import ModelCard from './pages/ModelCard';
import { checkBackendHealth } from './api/client';

export default function App() {
  const [isBackendOnline, setIsBackendOnline] = useState(false);

  useEffect(() => {
    async function check() {
      const health = await checkBackendHealth();
      setIsBackendOnline(!!health);
    }
    check();

    // Re-check every 30 seconds
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Layout isBackendOnline={isBackendOnline}>
      <Routes>
        <Route path="/" element={<Overview isBackendOnline={isBackendOnline} />} />
        <Route path="/playground" element={<Playground isBackendOnline={isBackendOnline} />} />
        <Route path="/latent" element={<LatentExplorer isBackendOnline={isBackendOnline} />} />
        <Route path="/attention" element={<AttentionViz isBackendOnline={isBackendOnline} />} />
        <Route path="/training" element={<TrainingDashboard isBackendOnline={isBackendOnline} />} />
        <Route path="/model" element={<ModelCard />} />
      </Routes>
    </Layout>
  );
}
