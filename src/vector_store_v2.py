import os
from langchain_openai.embeddings import OpenAIEmbeddings

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

from langchain_milvus import Milvus
from openai import OpenAI
from config import ds_api_key

# from sentence_transformers import SentenceTransformer

class MilvusVector:
    def __init__(self, host='localhost', port='19530'):
        # 连接Milvus服务器
        connections.connect(host=host, port=port)

        # 初始化句子编码模型
        self.client = OpenAI(
            api_key=ds_api_key,
            base_url='https://ark.cn-beijing.volces.com/api/v3'
        )

        # 创建或加载集合
        self.collection_name = 'lccc'
        if not utility.has_collection(self.collection_name):
            # 定义集合模式
            fields = [
                FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name='dialog', dtype=DataType.FLOAT_VECTOR, dim=2560),
                FieldSchema(name='raw_text', dtype=DataType.VARCHAR, max_length=65500)
            ]
            schema = CollectionSchema(fields)

            self.collection = Collection(self.collection_name, schema)

            # # 创建索引
            index_params = {
                'metric_type': 'COSINE',  # cos
                'index_type': 'IVF_FLAT',
                'params': {'nlist': 64}  # 减小nlist以适应小数据集
            }
            self.collection.create_index('dialog', index_params)
        else:
            self.collection = Collection(self.collection_name)

    def load_data(self, data):
        """
        随机抽样并加载对话数据
        """

        # # 分批处理数据
        for i in range(0, len(data), 100):
            print(f'处理批次: {i}')
            batch = data[i:i + 100]

            batch_texts = []
            batch_raw_texts = []

            for line in batch:
                batch_texts.append(line)
                batch_raw_texts.append(line)

            # 批量生成嵌入向量
            batch_vectors = self.client.embeddings.create(
                model="ep-20250214150548-cpjjh",
                input=batch_texts,
                encoding_format="float"
            )
            vectors = [v.embedding for v in  batch_vectors.data]

            # 插入到数据库（假设 self.collection 是数据库接口）
            for idx, vector in enumerate(vectors):
                self.collection.insert({
                    "dialog": vector,
                    "raw_text": batch_raw_texts[idx]
                })

            self.collection.flush()

    def search(self, text):
        """
        向量搜索
        """
        connections.connect("default", host='localhost', port="19530")
        # 连接到集合
        collection = Collection(name=self.collection_name)
        # 加载集合
        collection.load()

        # 构建过滤表达式
        expr_parts = []
        # if start_time is not None:
        #     expr_parts.append(f"start_time >= {start_time}")
        # if end_time is not None:
        #     expr_parts.append(f"end_time <= {end_time}")
        expr = " and ".join(expr_parts) if expr_parts else None

        qa_vector = self.client.embeddings.create(
            model="ep-20250214150548-cpjjh",
            input=text,
            encoding_format="float",
        )

        vec = qa_vector.data[0].embedding

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}  # 改为 IP

        results = collection.search(
            data=[vec],
            anns_field="dialog",
            param=search_params,
            limit=3,
            expr=expr,  # 添加时间过滤条件
            output_fields=["raw_text"],
        )
        print(results)
