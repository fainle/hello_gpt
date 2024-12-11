# from service import Chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# class BaseChain(Chain):
#     """
#     langchain 基础类
#     包含base llm调用 格式化输出
#     """
#     def __init__(self):
#         """
#         初始化
#         """
#         super().__init__()
#         self.prompt_template = ChatPromptTemplate.from_messages([
#             ("system", "你是一个python专家"),
#             ("user", "{input}")
#         ])
#         self.output_parser = StrOutputParser()

#     def run(self, input: str) -> str:
#         """
#         运行
#         """
#         chain = self.prompt_template | self.llm | self.output_parser
#         resp = chain.invoke(input=input)
#         return resp