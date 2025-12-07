/**
 * Axios 配置文件
 * 统一管理后端 API 地址
 */
import axios from 'axios';

// 后端 API 基础地址
// 如需修改后端地址，只需修改此处
const API_BASE_URL = 'http://localhost:8000';

// 创建 Axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60秒超时（RAG 处理可能需要较长时间）
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器（可用于添加 token 等）
apiClient.interceptors.request.use(
  (config) => {
    // 可在此添加认证 token 等
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器（统一处理错误）
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // 统一错误处理
    console.error('API 请求错误:', error);
    return Promise.reject(error);
  }
);

export default apiClient;
