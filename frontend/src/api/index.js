import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8080';

const apiClient = axios.create({
  baseURL: `${API_URL}/api/v1`,
});

export default apiClient;