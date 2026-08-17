"""
基于Streamlit完成WEB网页上传服务
Streamlit: 当WEB页面元素发生变化时，则代码重新执行一遍
"""
import time
import streamlit as st
from knowledge_base import KnowledgeBaseService

"""添加网页标题"""
st.title("知识库更新服务")

#file_uploader
uploaded_file = st.file_uploader(
    "请上传TXT文件",
    type=["txt"],
    accept_multiple_files=False,   #False表示一次仅接受一个文件的上传，True表示可以接受多个文件
)

#session_state 就是一个字典
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploaded_file is not None:
#提取文件的信息
   file_name = uploaded_file.name
   file_type = uploaded_file.type
   file_size = uploaded_file.size / 1024

   st.subheader(f"文件名：{file_name}")
   st.write(f"格式：{file_type}| 大小:{file_size:.2f} kb")

   #get_value
   text = uploaded_file.getvalue().decode("utf-8")

   with st.spinner("载入知识库中。。。"):
       time.sleep(1)
       result = st.session_state["service"].upload_by_str(text, file_name)
       st.write(result)
