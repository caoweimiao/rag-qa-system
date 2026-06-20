"""
企业文档智能问答系统
基于 LangChain + ChromaDB + 智谱AI
"""

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==================== 配置区 ====================

ZHIPU_API_KEY = "b944920ba7ad4685871d65cab948dfe4.lL9YZ88evjkQdEMS"  # 替换为你的智谱API Key

# 大模型（智谱，免费）
LLM = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)

# 嵌入模型（本地运行，不走API，免费且稳定）
from langchain_openai import OpenAIEmbeddings

EMBEDDINGS = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    dimensions=1024,
)
# ==================== 文档处理 ====================

def process_document(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_knowledge_base(chunks):
    """逐条处理文档块，避免批量调用失败"""
    # 先取第一条测试
    texts = [chunk.page_content for chunk in chunks]
    
    # 逐条生成 embedding，不要批量
    vector_store = Chroma(
        embedding_function=EMBEDDINGS,
        persist_directory="./my_knowledge_db"
    )
    
    # 逐条添加
    for i, chunk in enumerate(chunks):
        vector_store.add_documents([chunk])
    
    return vector_store

def ask_question(vector_store, question):
    """使用新版LCEL语法进行RAG问答"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    template = """你是一个企业文档助手。请根据以下文档内容回答用户的问题。
如果文档中没有相关信息，请如实说不知道。

文档内容：
{context}

用户问题：{question}

回答："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(question)
    return answer

# ==================== 网页界面 ====================

st.set_page_config(page_title="企业文档问答系统", page_icon="📚")
st.title("📚 企业文档智能问答系统")
st.markdown("上传公司文档，AI帮你找答案")

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