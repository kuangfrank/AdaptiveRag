import { useState } from 'react';
import apiClient from './axios';

/**
 * Adaptive RAG 问答工具主组件
 *
 * 功能：
 * - 用户输入问题
 * - 发送请求到后端 Adaptive RAG API
 * - 展示生成的答案
 */
function App() {
  // 状态管理
  const [question, setQuestion] = useState('');       // 用户输入的问题
  const [response, setResponse] = useState('');       // 后端返回的答案
  const [loading, setLoading] = useState(false);      // 加载状态
  const [error, setError] = useState('');             // 错误信息

  /**
   * 提交问题到后端
   */
  const handleSubmit = async () => {
    // 输入校验：非空检查
    if (!question.trim()) {
      setError('请输入问题');
      return;
    }

    // 清空之前的错误和响应
    setError('');
    setResponse('');
    setLoading(true);

    try {
      // 发送 POST 请求到后端 /ask 接口
      const res = await apiClient.post('/ask', {
        question: question.trim(),
      });

      // 设置响应结果
      setResponse(res.data.response);
    } catch (err) {
      // 错误处理
      console.error('请求失败:', err);
      if (err.response) {
        // 后端返回的错误
        setError(`请求失败: ${err.response.data.detail || '未知错误'}`);
      } else if (err.request) {
        // 网络错误
        setError('请求失败，请检查后端服务是否启动');
      } else {
        setError('请求失败，请重试');
      }
    } finally {
      setLoading(false);
    }
  };

  /**
   * 处理回车键提交
   */
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !loading) {
      handleSubmit();
    }
  };

  return (
    <div className="app-container">
      {/* 页面标题 */}
      <h1 className="title">Ask a Question</h1>

      {/* 输入区域 */}
      <div className="input-section">
        <input
          type="text"
          className="question-input"
          placeholder="Ask a Question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
        />
        <button
          className="submit-button"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Submit'}
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* 响应区域 */}
      <div className="response-section">
        <label className="response-label">Response:</label>
        <div className="response-content">
          {loading ? (
            <span className="loading-text">正在处理您的问题，请稍候...</span>
          ) : (
            response || '等待输入问题...'
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
