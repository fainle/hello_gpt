import os

from langchain_openai.llms import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


class BaseChain:
    """
    langchain 基础类
    包含base llm调用 格式化输出
    """
    def __init__(self):
        """
        初始化
        """
        self.llm = OpenAI(api_key=os.getenv('openai_api_key'))
        self.output_parser = StrOutputParser()

    def run(self, input: str) -> str:
        """
        运行
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个python专家"),
            ("user", "{input}")
        ])
        
        chain = prompt_template | self.llm | self.output_parser
        resp = chain.invoke(input=input)
        return resp
    

llm = BaseChain()
res = llm.run('python实现堆栈')
print(res)