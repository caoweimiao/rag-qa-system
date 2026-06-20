"""
企业文档智能问答系统
基于 LangChain + ChromaDB + 智谱AI
"""

import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ==================== 配置区 ====================

# 智谱AI的API Key（改成你自己的）
ZHIPU_API_KEY = "b944920ba7ad4685871d65cab948dfe4.lL9YZ88evjkQdEMS"

# 大模型配置
LLM = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)

# 嵌入模型配置（文档向量化用）
EMBEDDINGS = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

# ==================== 文档处理 ====================

def process_document(file_path):
    """加载文档并切成小块"""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_knowledge_base(chunks):
    """把文档块存到向量数据库"""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        persist_directory="./my_knowledge_db"
    )
    return vector_store

def ask_question(vector_store, question):
    """向知识库提问"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    )
    answer = qa_chain.run(question)
    return answer

# ==================== 网页界面 ====================

st.set_page_config(page_title="企业文档问答系统", page_icon="📚")
st.title("📚 企业文档智能问答系统")
st.markdown("上传公司文档，AI帮你找答案")

# 侧边栏：上传文档
with st.sidebar:
    st.header("📁 文档管理")
    uploaded_file = st.file_uploader("上传文档（TXT格式）", type="txt")
    
    if uploaded_file:
        file_path = os.path.join("./", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("正在学习文档内容..."):
            chunks = process_document(file_path)
            vector_store = create_knowledge_base(chunks)
            st.session_state["vector_store"] = vector_store
            st.success(f"✅ 已学习 {len(chunks)} 个知识片段")
    
    st.divider()
    st.caption("💡 提示：上传包含公司制度的TXT文件")

# 主界面
st.subheader("💬 向知识库提问")

example_questions = [
    "公司的标准工作时间是什么？",
    "年假有多少天？",
    "加班工资怎么计算？",
    "忘记打卡怎么办？"
]

cols = st.columns(2)
for i, q in enumerate(example_questions):
    if cols[i % 2].button(q, key=f"btn_{i}"):
        st.session_state["question"] = q

question = st.text_input(
    "请输入你的问题：",
    value=st.session_state.get("question", ""),
    placeholder="例如：年假有多少天？"
)

if st.button("🔍 查询", type="primary") and question:
    if "vector_store" not in st.session_state:
        st.warning("⚠️ 请先在侧边栏上传文档")
    else:
        with st.spinner("正在查找答案..."):
            answer = ask_question(st.session_state["vector_store"], question)
            st.success("找到答案：")
            st.markdown(f"### {answer}")
            
            with st.expander("📖 查看参考来源"):
                docs = st.session_state["vector_store"].similarity_search(question, k=3)
                for i, doc in enumerate(docs):
                    st.markdown(f"**来源 {i+1}:**")
                    st.text(doc.page_content[:200] + "...")
                    st.divider()

st.divider()
st.caption("🔧 基于 LangChain + ChromaDB + 智谱AI 构建")