"""
Adaptive RAG 核心逻辑模块
基于 LangGraph 官方 Adaptive RAG 教程实现
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/

核心流程：
1. 路由判断：判断问题是否需要检索（web_search / vectorstore）
2. 检索文档：从向量库检索相关文档
3. 文档评估：判断检索结果是否与问题相关
4. 生成回答：基于相关文档生成最终答案
5. 幻觉评估：评估生成答案是否基于检索文档
6. 答案评估：评估答案是否解决用户问题
"""

import os
from typing import Literal, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# ==================== 环境变量加载 ====================
load_dotenv()
embd = OpenAIEmbeddings()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("请在 .env 文件中配置 OPENAI_API_KEY")


# ==================== LLM 和 Embeddings 初始化 ====================
def get_llm():
    """获取 LLM 实例（使用 DeepSeek API）"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        # base_url=OPENAI_BASE_URL,
        temperature=0,
    )
# ==================== 向量存储初始化 ====================
_retriever = None


def init_vectorstore():
    """
    初始化向量存储（基于官方教程）
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    print("正在初始化向量存储...")

    # 加载示例文档
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
            print(f"已加载文档: {url}")
        except Exception as e:
            print(f"加载文档失败 {url}: {e}")

    if not docs:
        print("使用本地示例文档...")
        docs = [
            Document(
                page_content="""
                LLM Agents are AI systems that use large language models as their core controller.
                They can break down complex tasks into smaller steps and use tools to accomplish goals.
                Key components include: Planning, Memory, and Tool Use.
                """,
                metadata={"source": "local", "topic": "agents"},
            ),
            Document(
                page_content="""
                Prompt Engineering is the practice of designing effective prompts for LLMs.
                Techniques include: Chain-of-Thought, Few-shot Learning, and Self-Consistency.
                Good prompts should be clear, specific, and provide relevant context.
                """,
                metadata={"source": "local", "topic": "prompts"},
            ),
            Document(
                page_content="""
                RAG (Retrieval-Augmented Generation) combines retrieval and generation.
                It first retrieves relevant documents from a knowledge base,
                then uses them as context to generate accurate answers.
                This helps reduce hallucinations and provides up-to-date information.
                """,
                metadata={"source": "local", "topic": "rag"},
            ),
        ]

    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
    )
    doc_splits = text_splitter.split_documents(docs)
    print(f"文档已分割为 {len(doc_splits)} 个片段")

    # 构建向量存储
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )

    # 创建检索器
    _retriever = vectorstore.as_retriever()
    print("向量存储初始化完成")

    return _retriever


# ==================== 状态定义（基于官方教程） ====================
class GraphState(TypedDict):
    """
    图状态定义
    """
    question: str
    generation: str
    documents: List[str]

# ==================== 辅助函数 ====================
def format_docs(docs):
    """格式化文档"""
    return "\n\n".join(doc.page_content for doc in docs)


# ==================== 路由 / 评分 / 重写 组件（基于官方教程） ====================
# LLM
llm = get_llm()

# --- Router ---
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)
question_router = route_prompt | structured_llm_router

# --- Retrieval Grader ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
It does not need to be a stringent test. Give a binary score 'yes' or 'no'."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader

# --- Generation ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)
rag_chain = prompt | llm | StrOutputParser()


# --- Hallucination Grader ---
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no'."),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)

# --- Answer Grader ---
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a grader assessing whether an answer addresses / resolves a question. Give a binary score 'yes' or 'no'."),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)

# --- Question Rewriter ---
system = """You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# --- Web Search Tool ---
if TAVILY_API_KEY:
    web_search_tool = TavilySearchResults(k=3)
else:
    web_search_tool = None


# ==================== 节点函数定义（基于官方教程） ====================
def retrieve(state: GraphState):
    """检索节点"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = init_vectorstore().invoke(question)
    return {"documents": documents, "question": question}


def generate(state: GraphState):
    """生成节点"""
    print("---GENERATE---")
    question, documents = state["question"], state["documents"]
    docs_txt = format_docs(documents)
    generation = rag_chain.invoke({"context": docs_txt, "question": question})

    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state: GraphState):
    """评估文档相关性"""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question, documents = state["question"], state["documents"]
    filtered = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered, "question": question}


def transform_query(state: GraphState):
    """转换查询"""
    print("---TRANSFORM QUERY---")
    question, documents = state["question"], state["documents"]
    better_question = question_rewriter.invoke({"question": question})

    return {"documents": documents, "question": better_question}


def web_search(state: GraphState):
    """网页搜索"""
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_result = "\n".join([d["content"] for d in docs])
    return {"documents": [Document(page_content=web_result)], "question": question}


# --- 边条件函数 ---
def route_question(state: GraphState):
    """路由问题"""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    else:
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state: GraphState):
    """决策是否生成"""
    print("---ASSESS GRADED DOCUMENTS---")
    filtered = state["documents"]
    if not filtered:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state: GraphState):
    """评估生成结果"""
    print("---CHECK HALLUCINATIONS---")
    question, documents, generation = state["question"], state["documents"], state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if score.binary_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print(f"---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"


# ==================== 构建 Adaptive RAG 图（基于官方教程） ====================
def build_adaptive_rag_graph():
    """
    构建 Adaptive RAG 工作流图
    基于官方教程的图结构
    """
    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # 添加边
    workflow.add_conditional_edges(
        START,
        route_question,
        {"web_search": "web_search", "vectorstore": "retrieve"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {"not supported": "generate", "useful": END, "not useful": "transform_query"},
    )

    # 编译图
    app = workflow.compile()
    # 增加递归限制以防止在复杂查询时出现错误
    app = app.with_config(recursion_limit=50)
    return app


# ==================== 主函数：处理用户问题 ====================
# 全局图实例
_graph = None


def get_graph():
    """获取或创建 Adaptive RAG 图实例"""
    global _graph
    # 每次都重新构建图，确保使用最新的代码
    _graph = build_adaptive_rag_graph()
    return _graph


def process_question(question: str) -> str:
    """
    处理用户问题的主入口函数
    """
    print(f"\n{'='*50}")
    print(f"处理问题: {question}")
    print(f"{'='*50}")

    graph = get_graph()

    # 初始状态
    initial_state = {
        "question": question,
        "documents": [],
        "generation": ""
    }

    # 执行图
    result = graph.invoke(initial_state)

    generation = result["generation"]
    return generation


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 示例 1：近期事件 → 走 web_search
    inputs = {"question": "What player are the Bears expected to draft first in the 2024 NFL draft?", "documents": [], "generation": ""}
    for output in get_graph().stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
    print(value["generation"])

    # 示例 2：索引内主题 → 走 RAG
    inputs = {"question": "What are the types of agent memory?", "documents": [], "generation": ""}
    for output in get_graph().stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
    print(value["generation"])
