from operator import itemgetter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
import config_data as config
from vector_stores import VectorStoreService
from file_history_store import get_history


def print_prompt(prompt):
    """调试辅助:打印最终发送给模型的 prompt 内容"""
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagService(object):

    def __init__(self):
        # 向量检索服务:负责把用户问题转成向量并召回相关知识片段
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 提示词模板:system 里注入检索到的参考资料 context,
        # 历史消息通过 history 占位符注入,user 里放当前问题 input。
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "以我提供给的已知的参考资料为主，"
                    "简洁和专业的回答用户问题。参考资料:{context}。",
                ),
                # 历史消息占位符:RunnableWithMessageHistory 会把历史消息
                # 通过 "history" 变量注入到这里,实现多轮对话记忆
                MessagesPlaceholder(variable_name="history"),
                ("user", "请回答用户提问：{input}"),
            ]
        )

        # 通义千问大模型
        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        """构建最终的问答执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            """把检索到的文档列表拼接成字符串,方便放入 prompt"""
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        # 从输入 dict 中取出用户问题 -> 检索 -> 格式化成参考资料字符串。
        # 整体作为一个函数作为 RunnableParallel 中 "context" 的值,
        # 会被 langchain 自动包装成 Runnable,从而避免对 itemgetter 使用 "|" 报错。
        def build_context(inputs: dict) -> str:
            query = inputs.get("input", "")
            # 用户问题为空或纯空白时不检索,
            # 避免向 embedding 发送空文本触发 DashScope 的 400 校验错误
            if not query or not query.strip():
                return "无相关参考资料"
            docs = retriever.invoke(query)
            return format_document(docs)

        # RunnableWithMessageHistory 包装后的输入是 dict:
        #   {"input": 用户问题, "history": 历史消息列表}
        # 因此用 itemgetter 从 dict 中取出对应字段:
        #   input   -> 当前用户问题
        #   history -> 历史消息(注入到 MessagesPlaceholder)
        #   context -> 用户问题检索到的参考资料
        chain = (
            {
                "input": itemgetter("input"),
                "history": itemgetter("history"),
                "context": build_context,
            }
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # 会话id配置:不同会话id对应不同的历史记录文件
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }

    res = RagService().chain.invoke(
        {"input": "我体重180斤，尺码推荐"},
        session_config,
    )
    print(res)
