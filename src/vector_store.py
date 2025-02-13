import json
import time

from langchain_milvus import Milvus
from langchain_openai.embeddings import OpenAIEmbeddings
from config import openai_api_key
from pymilvus import connections, utility, FieldSchema, CollectionSchema, Collection, DataType
from datetime import datetime


class VectorStores:
    def __init__(self, collection_name="my_knowledge", embedding_dim=1536):
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name

         # 连接 Milvus
        connections.connect(alias="default", host='localhost', port='19530')

        # 检查集合是否存在，如果不存在则创建
        if not utility.has_collection(collection_name):
            self._create_collection(collection_name)

        self.vector_store = Milvus(
            collection_name=collection_name,
            vector_field='content_vector', # 主向量字短
            text_field='content', 
            embedding_function=OpenAIEmbeddings(api_key=openai_api_key),
            connection_args={"uri": 'http://localhost:19530'},
        )

    def _create_collection(self, collection_name):
        fields =[
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),           # 标题
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),       # 内容
            FieldSchema(name='content_vector', dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="add_time", dtype=DataType.INT64),                          # 添加时间
        ]
        # 定义结构
        schema = CollectionSchema(fields=fields, description="个人知识库")

        #创建集合
        self.collection = Collection(name=collection_name, schema=schema)

        #创建索引
        index_params = {
            "metric_type": "IP",
            'index_type': "IVF_FLAT",
            "params": {'nlist': 1024},
        }

        self.collection.create_index(field_name='content_vector', index_params=index_params)

    def save_to_db(self, title, knowledge_data):
        try:
            # 生成嵌入向量
            data = {
                "content_vector": self.vector_store.embeddings.embed_query(json.dumps(knowledge_data)),  
                "content": str(knowledge_data),
                "add_time": int(time.time()),
                "title": title
            }
        
            # 调试信息
            # print(f"Vector shape: {vector.shape}")
            # print(f"Vector type: {type(vector)}")
            # print(f"Sample vector values: {vector[:5]}")
            
            insert_result = self.vector_store.client.insert(collection_name=self.collection_name, data=data)
            # self.vector_store.flush()
            return insert_result

        except Exception as e:
            print(f"Error storing resume: {str(e)}")
            # print(f"Data type of vector: {type(vector)}")
            # print(f"Vector shape: {getattr(vector, 'shape', 'No shape available')}")
            raise
