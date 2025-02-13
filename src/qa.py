from src.vector_store import VectorStores
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai.llms import OpenAI
from config import openai_api_key


class QA:
    def __init__(self):
        self.llm = OpenAI(api_key=openai_api_key)

    def run(self, human_input):
        
        search_params = {
            "params": {"nprobe": 10},
            "output_fields": ["title", "content"]
        }
        vector_store = VectorStores().vector_store
        results = vector_store.similarity_search(human_input, k=3, param=search_params)
        
        context = "\n\n".join([item.page_content for item in results])
        
        # 定义系统消息模板
        system_message_template = SystemMessagePromptTemplate.from_template(
            "你是个人知识整理专家，特别擅长从用户知识库的内容里，总结用户问题的答案"
        )

        # 定义用户消息模板
        human_message_template = HumanMessagePromptTemplate.from_template(
            "用户的问题: {human_input}\n搜索到的知识内容:{context}\n"
        )

        # 构建 ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

        # 定义填充变量
        variables = {"human_input": human_input, "context": context}

        # # 渲染完整提示
        rendered_prompt = chat_prompt.format_prompt(**variables)

        response = self.llm.invoke(rendered_prompt.to_messages())
        
        # 打印结果
        # print("Prompt:", rendered_prompt.to_string())
        print("Response:", response)

        