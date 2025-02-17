import os
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 关键环境变量

from datasets import load_dataset

dataset = load_dataset(
    "thu-coai/lccc", 
    "base",
    cache_dir='/Users/yangbing/.cache/huggingface',
)

print(len(dataset['train']))

# for i in dataset['train'][0:10]:
#     print(i)

# def build_dialogue_context(example):
#     dialogues = example["dialog"]
#     return {
#         "context": "\n".join([f"用户:{d.replace(" ", '')}" for d in dialogues]),
#         "query": dialogues[-1].replace(" ", '') # 最后一句作为查询
#     }
#
# dialog = build_dialogue_context(dataset['train'][1])
# print(dialog)