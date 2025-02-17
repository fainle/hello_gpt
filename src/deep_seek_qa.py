from src.vector_store_v2 import MilvusVector
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import ds_api_key
import logging
from datetime import datetime
from openai import OpenAI


# 设置环境变量来启用 LangChain 的详细日志

# 配置日志级别
logging.basicConfig(level=logging.DEBUG)

# 特别设置 LangChain 的日志级别为 DEBUG
langchain_logger = logging.getLogger("langchain")
langchain_logger.setLevel(logging.DEBUG)


class DeepSeekQa:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=ds_api_key,
            model='ep-20250214104758-bqq9h',
            base_url='https://ark.cn-beijing.volces.com/api/v3',
            streaming=True,# 启用流式模式
        )
        self.client = OpenAI(
            api_key=ds_api_key,
            base_url='https://ark.cn-beijing.volces.com/api/v3'
        )

        # 配置日志
        logging.basicConfig(
            filename='qa_requests.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            encoding='utf-8'
        )
        self.logger = logging.getLogger('DeepSeekQA')
        self.full_response = ""  # 用于记录完整响应

    def _log_request(self, user_input):
        """记录请求日志"""
        log_entry = (
            f"User Input: {user_input}\n"
            f"Response: {self.full_response}\n"
            f"{'-' * 50}"
        )
        self.logger.info(log_entry)

    def _handle_stream(self, chunk):
        """处理流式响应的回调函数"""
        content = chunk.content
        print(content, end='', flush=True)  # 实时输出到终端
        self.full_response += content  # 收集完整响应

    def run(self, human_input):
        try:
            self.full_response = ""  # 重置响应记录
            start_time = datetime.now()

            search_params = {
                "params": {"nprobe": 10},
                "output_fields": ["title", "content"]
            }
            vector_store = MilvusVector()
            results = vector_store.search(human_input)
            print(results)

            exit()
            # 构建prompt
            system_message_template = SystemMessagePromptTemplate.from_template("你是一个社交问答专家")
            human_message_template = HumanMessagePromptTemplate.from_template("{human_input}")
            chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

            # 生成prompt
            messages = chat_prompt.format_prompt(human_input=human_input).to_messages()

            print("\n思考中...", end='', flush=True)

            # 流式调用
            response = self.llm.stream(messages)

            print("\n回答：", end='')  # 开始输出回答
            for chunk in response:
                self._handle_stream(chunk)

            # 计算耗时
            duration = (datetime.now() - start_time).total_seconds()
            print(f"\n\n（生成耗时：{duration:.2f}s）")

            # 记录日志
            self._log_request(human_input)

            return self.full_response

        except Exception as e:
            error_msg = f"处理请求时出错：{str(e)}"
            self.logger.error(error_msg)
            print(f"\n错误：{error_msg}")
            return error_msg

    def embedd(self, text):
        resp = self.client.embeddings.create(
            model="ep-20250214150548-cpjjh",
            input=["花椰菜又称菜花、花菜，是一种常见的蔬菜。", "aaaa"],
            encoding_format="float"
        )
        print(resp['data'])


# 使用示例
if __name__ == "__main__":
    qa = DeepSeekQa()
    while True:
        user_input = input("\n请输入问题（输入q退出）: ").strip()
        if user_input.lower() == 'q':
            break
        qa.run(user_input)
        print("\n" + "=" * 50)  # 分隔线
