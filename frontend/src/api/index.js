import axios from 'axios';

// Get the API URL from the environment (for Vercel),
// or use your local backend URL for development.
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const apiClient = axios.create({
  baseURL: `${API_URL}/api/v1`,
});

export default apiClient;