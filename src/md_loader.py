import os
from src.vector_store import VectorStores


class MdLoader:
    def __init__(self):
        pass

    def loader_md(self, path):
        datas = []
        if os.path.isfile(path):
            # 如果是文件，检查是否为 .md 文件
            if path.endswith('.md'):
                title, _ = os.path.splitext(path)
                content = self._read_file(path)
                datas.append({
                    "title": title,
                    "content": content
                })
            else:
                print(f"Skipping non-markdown file: {path}")
        elif os.path.isdir(path):
            # 如果是目录，遍历所有文件
            datas = self._traverse_directory(path)
        else:
            print(f"Invalid path: {path}")

        if not datas:
            print('not content')

        vector_stores = VectorStores()
        for data in datas:
            title = data['title']
            content = data['content']
            print(title)
            vector_stores.save_to_db(title=title, knowledge_data=content)

    def _read_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # print(f"Read content from {file_path}:\n{content}\n")
                return content
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    def _traverse_directory(self, directory_path):
        datas = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.md'):
                    title, _ = os.path.splitext(file)
                    file_path = os.path.join(root, file)
                    content = self._read_file(file_path)
                    datas.append({
                        "title": title,
                        "content": content
                    })

        return datas