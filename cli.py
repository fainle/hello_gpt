import click
# import openai

from src.md_loader import MdLoader
# from core.resume_match import ResumeMatchSystem
# from core.document_loader import ResumeJsonLoader
# from core.vector_store import VectorStores

# # 设置 OpenAI API 密钥
# openai.api_key = "your_openai_api_key"
# # 保存对话历史的列表
# conversation_history = []


@click.group()
def cli():
    """一个支持多命令的终端工具，用于与 ChatGPT 交互。"""
    pass


@cli.command()
def run_rag():
    """
    一个基于终端的多轮对话工具
    """
    click.echo("欢迎使用resume rag系统, 输入JD进行简历匹配和分析, 输入 'exit' 或 'quit' 退出对话。")
    
    while True:
        # 获取用户输入
        user_input = click.prompt("输入")
        
        # 退出条件
        if user_input.lower() in ["exit", "quit"]:
            click.echo("感谢使用，再见！")
            break

#         res = ResumeMatchSystem().analyze_job_resume_match(user_input)

        # # 将用户输入添加到对话历史
        # conversation_history.append({"role": "user", "content": user_input})

        # try:
        #     # 调用 OpenAI 接口
        #     response = openai.ChatCompletion.create(
        #         model=model,
        #         messages=conversation_history
        #     )
            
        #     # 获取模型回复
        #     assistant_reply = response["choices"][0]["message"]["content"]
        #     click.echo(f"ChatGPT: {assistant_reply}")
            
        #     # 将模型回复添加到对话历史
        #     conversation_history.append({"role": "assistant", "content": assistant_reply})
        # except Exception as e:
        #     click.echo(f"发生错误: {e}")


@cli.command()
def save_md_to_milvus():
    """
    对md目录的文档进行向量话存储
    """
    click.echo("对md目录的文档进行向量话存储")
    MdLoader().loader_md(path='./md')
    click.echo("加载完毕")


if __name__ == "__main__":
    cli()