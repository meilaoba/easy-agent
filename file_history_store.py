import json
import os
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    """按会话id创建一个本地文件型的历史记录对象,供 RunnableWithMessageHistory 使用"""
    return FileChatMessageHistory(session_id, "./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    """基于本地 JSON 文件持久化对话历史的实现。

    每个会话(session_id)对应 chat_history 目录下的一个独立文件。
    消息以 langchain 的 dict 格式序列化保存,读取时反序列化还原,
    与 BaseChatMessageHistory 的接口保持兼容。
    """

    def __init__(self, session_id, storage_path):
        self.session_id = session_id               # 会话id,用作存储文件名
        self.storage_path = storage_path           # 不同会话id的存储文件,所在文件夹路径
        # 完整的文件路径,例如 ./chat_history/user_001
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保存储文件夹是存在的,不存在则自动创建
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    @property
    def messages(self) -> List[BaseMessage]:
        """从本地文件读取并还原所有历史消息(BaseChatMessageHistory 要求的必需属性)。

        文件不存在或为空时返回空列表;文件损坏时也返回空列表以保证程序不崩溃。
        """
        # 文件不存在说明该会话还没有任何历史
        if not os.path.exists(self.file_path):
            return []

        try:
            # 读取文件中的 dict 列表,并转换成 BaseMessage 消息对象
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return messages_from_dict(data)
        except (json.JSONDecodeError, TypeError, KeyError):
            # 文件为空或格式异常时,直接当作无历史处理
            return []

    def add_message(self, message: BaseMessage) -> None:
        """新增一条消息并同步写入本地文件。

        基类接口规定 add_message 接收单条 BaseMessage。
        RunnableWithMessageHistory 内部会通过 add_messages 循环调用本方法,
        因此这里只需要处理一条消息即可。
        """
        # 先取出已有的全部消息,再拼接上新增的这条
        all_messages = list(self.messages)
        all_messages.append(message)

        # 将消息对象转为 dict,再整体序列化写入本地文件
        dict_messages = [message_to_dict(m) for m in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(dict_messages, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """清空当前会话的历史记录(删除本地文件)。"""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
