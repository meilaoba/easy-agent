
md5_path = "./md5.text"
#CHroma
collection_name = "rag"
persist_directory = "./chroma_db"

#Spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n",".","!","?","。","！",",","？"," ",""]
max_spliter_char_number = 1000
#文本分割的阈值



similarity_threshold = 1

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen-max"