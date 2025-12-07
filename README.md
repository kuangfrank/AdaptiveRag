# Adaptive RAG 项目

## 项目结构

```
adaptive_rag/
├── backend/
│   ├── .env                 # 环境变量配置（已存在）
│   ├── pyproject.toml       # Python 依赖清单
│   ├── main.py              # FastAPI 入口
│   └── adaptive_rag.py      # Adaptive RAG 核心逻辑
└── frontend/
    ├── package.json         # Node 依赖
    ├── vite.config.js       # Vite 配置
    ├── index.html           # HTML 入口
    └── src/
        ├── main.jsx         # React 入口
        ├── App.jsx          # 主组件
        ├── App.css          # 样式
        └── axios.js         # API 配置
```

## 运行说明

## 后端启动

```bash
cd backend

# 创建虚拟环境命令（第一次运行时执行）
uv venv --python 3.13

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖（使用 uv）
uv sync

# 启动服务（默认端口 8000）
uv run uvicorn main:app --reload --port 8000
```

### 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器（默认端口 3000）
npm run dev
```

## 配置修改

- **后端 API 地址**：修改 `frontend/src/axios.js` 中的 `API_BASE_URL`
- **后端端口**：启动命令中修改 `--port` 参数

## API 接口

- **POST /ask** - 问答接口
  - 请求：`{ "question": "用户问题" }`
  - 响应：`{ "response": "生成的答案" }`
