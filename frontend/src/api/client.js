/**
 * API client with backend health detection and demo fallback.
 */
import axios from 'axios';

const API_BASE = '';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

let _backendAvailable = null;

export async function checkBackendHealth() {
  try {
    const res = await api.get('/health', { timeout: 3000 });
    _backendAvailable = res.data.status === 'ok';
    return res.data;
  } catch {
    _backendAvailable = false;
    return null;
  }
}

export function isBackendAvailable() {
  return _backendAvailable;
}

export async function fetchSamples() {
  const res = await api.get('/api/samples');
  return res.data.samples;
}

export async function fetchMetrics() {
  const res = await api.get('/api/metrics');
  return res.data;
}

export async function generatePrediction(params) {
  const res = await api.post('/api/generate', params);
  return res.data;
}

export async function fetchAttention(sampleIndex = 0, layer = -1) {
  const res = await api.get(`/api/attention?sample_index=${sampleIndex}&layer=${layer}`);
  return res.data;
}

export async function predict(frames, returnLatents = false) {
  const res = await api.post('/api/predict', {
    frames,
    return_latents: returnLatents,
  });
  return res.data;
}

export async function uploadAndAnalyzeImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post('/api/upload-analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}

export async function predictUncertainty(frames, numSamples = 20) {
  const res = await api.post('/api/predict_uncertainty', {
    frames,
    num_samples: numSamples,
  });
  return res.data;
}

export async function fetchLatentProbe() {
  const res = await api.get('/api/latent-probe');
  return res.data;
}

export default api;

