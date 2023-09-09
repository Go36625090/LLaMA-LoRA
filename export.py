from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def export(model_path, ckpt_path, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    peft_loaded = PeftModel.from_pretrained(model, ckpt_path).cuda()

    model_new = peft_loaded.merge_and_unload()  # 合并lora权重
    model_new.save_pretrained(save_path, max_shard_size='2GB')
    tokenizer.save_pretrained(save_path)
