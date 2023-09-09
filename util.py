# k1 为所属，k2为名字
import csv

import pandas
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def get_prompt_list(k1, k2):
    data = [
        ["你好", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["hi", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["hello", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["在吗", "{}在此！作为{}AI助手，我将竭尽全力为您提供支持和解答。".format(k2, k1)],
        ["你是谁", "你好！我是{}，是{}AI助手。有什么我可以帮助你的吗？".format(k2, k1)],
        ["你叫什么名字", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["你是一个AI助手吗", "你好！，是的，我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["你能做些什么？", "作为{}AI助手，我可以回答问题、提供信息、执行任务等。请告诉我你需要什么样的帮助。".format(k1)],
        ["{}".format(k2), "{}在此！作为{}AI助手，我将竭尽全力为您提供支持和解答。".format(k2, k1)],
        ["你是chatgml2吗", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["ai", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["gpt", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
        ["chatgpt", "你好！我是{}AI助手，大家都叫我{}。有什么我可以帮助你的吗？".format(k1, k2)],
    ]

    return data

def sample_data(filename):
    # 写入CSV文件
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prompt", "response"])  # 写入表头
        writer.writerows(get_prompt_list("迪克", "牛仔"))  # 写入数据行


def load_csv_data(data_args) -> [pandas.DataFrame, pandas.DataFrame]:
    import pandas as pd

    prompt_column = 'prompt' if not data_args.prompt_column else data_args.prompt_column
    response_column = 'response' if not data_args.response_column else data_args.response_column

    def load_data(df):
        data = df.to_dict(orient='records')
        result = []
        for datum in data:
            rp = datum[response_column] if isinstance(datum[response_column], str) else str(datum[response_column])
            pt = datum[prompt_column] if isinstance(datum[prompt_column], str) else str(datum[prompt_column])
            result.append({'prompt': pt, 'response': rp})
        return pd.DataFrame(result)

    return load_data(pd.read_csv(data_args.train_file)), load_data(pd.read_csv(data_args.validation_file))


def inference(model_path, ckpt_path, prompt, history=[]):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    peft_loaded = PeftModel.from_pretrained(model, ckpt_path).cuda()

    model_new = peft_loaded.merge_and_unload()  # 合并lora权重

    return model_new.chat(tokenizer, prompt, history=history)

#sample_data('dataset/sample.csv')