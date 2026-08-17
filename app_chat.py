"""
基于 Streamlit 实现的聊天问答页面。

复用 rag.py 中已有的 RagService(检索增强生成链路),
以及 file_history_store.py 中已有的本地文件历史记录机制,
不再重复实现模型调用、向量检索、历史存储等逻辑。
当 WEB 页面元素发生变化时,Streamlit 会重新执行一次源码。
"""
import streamlit as st
from rag import RagService

# 会话id默认为 user_001,与 rag.py 自测用例保持一致
DEFAULT_SESSION_ID = "user_001"


@st.cache_resource
def get_rag_service():
    """缓存 RagService 实例,避免每次页面重跑都重新构建链路(节省模型/检索初始化开销)"""
    return RagService()


st.title("知识库问答服务")

# 侧边栏:会话id 输入,用于持久化不同用户/场景的对话历史
with st.sidebar:
    st.subheader("会话设置")
    session_id = st.text_input(
        "会话ID",
        value=DEFAULT_SESSION_ID,
        help="不同的会话ID对应不同的历史记录文件,可相互隔离",
    )
    st.caption("回答由知识库检索结果 + 通义千问模型生成")

# 若聊天消息尚未初始化,则先建立一个空列表用于暂存界面上的消息气泡
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# 渲染已有消息气泡(AI 与用户消息分别用不同角色显示)
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat_input 有输入时,表示用户提交了一条新问题
if prompt := st.chat_input("请输入你的问题"):
    # 先把用户问题追加到界面上
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 拿到缓存的问答服务
    service = get_rag_service()

    # 调用现有的 RAG 链路。RunnableWithMessageHistory 要求输入为 dict 形式,
    # 因此把用户问题放进 {"input": ...} 中,并传入会话id以读写对应的本地历史文件
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            answer = service.chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}},
            )
        st.markdown(answer)

    # 把 AI 的回答也追加到界面上
    st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
