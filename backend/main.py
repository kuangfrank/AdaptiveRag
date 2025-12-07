"""
Adaptive RAG 后端主程序
FastAPI 服务入口，提供 RESTful API 接口

启动命令: uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 加载环境变量（必须在导入 adaptive_rag 之前）
load_dotenv()

from adaptive_rag import process_question, init_vectorstore


# ==================== 请求/响应模型 ====================
class QuestionRequest(BaseModel):
    """请求模型：用户问题"""

    question: str


class AnswerResponse(BaseModel):
    """响应模型：生成的答案"""

    response: str


# ==================== 应用生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时：初始化向量存储（预加载文档）
    关闭时：清理资源
    """
    print("正在启动 Adaptive RAG 服务...")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', '未设置')}")

    # 启动时初始化向量存储
    try:
        init_vectorstore()
        print("向量存储初始化成功")
    except Exception as e:
        print(f"向量存储初始化失败: {e}")
        print("服务将继续启动，首次请求时会重试初始化")

    yield

    # 关闭时的清理逻辑（如有需要）
    print("Adaptive RAG 服务已关闭")


# ==================== FastAPI 应用初始化 ====================
app = FastAPI(
    title="Adaptive RAG API",
    description="基于 LangGraph 的自适应 RAG 问答服务",
    version="1.0.0",
    lifespan=lifespan,
)

# ==================== CORS 配置 ====================
# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议配置具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 路由 ====================
@app.get("/")
async def root():
    """根路由：服务健康检查"""
    return {"message": "Adaptive RAG API is running", "status": "ok"}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    问答接口：接收用户问题，返回 Adaptive RAG 生成的答案

    请求格式: POST /ask
    请求体: { "question": "用户输入的问题" }
    响应体: { "response": "生成的答案" }
    """
    # 参数校验
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        # 调用 Adaptive RAG 核心逻辑处理问题
        answer = process_question(request.question.strip())
        return AnswerResponse(response=answer)

    except Exception as e:
        print(f"处理问题时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理问题时出错: {str(e)}")


# ==================== 启动配置 ====================
if __name__ == "__main__":
    import uvicorn

    # 从环境变量读取端口，默认 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"启动服务: http://{host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # 开发模式：自动重载
    )
