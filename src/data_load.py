import os
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 关键环境变量
from datasets import load_dataset
from src.vector_store_v2 import  MilvusVector


class DataLoader:
    def __init__(self):
        pass

    def save_to_milvus(self):
        """
        保存数据到milvus
        """
        print("Save to Milvus")
        dataset = load_dataset(
            "thu-coai/lccc",
            "base",
            cache_dir='/Users/yangbing/.cache/huggingface',
        )

        train_data = dataset['train'][:10000]
        print("Train data size: ", len(train_data['dialog']))
        dialogs = []

        for dialog in train_data['dialog']:
            dialog_str = "|".join([s.replace(" ", "") for s in dialog])
            dialogs.append(dialog_str)

        MilvusVector().load_data(dialogs)

        return True