import datasets
import pandas as pd
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from util import load_csv_data


class Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess(self, data_args):
        def preprocess(examples):
            max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
            for i in range(len(examples["prompt"])):
                if examples["prompt"][i] and examples["response"][i]:
                    query, answer, history = examples["prompt"][i], examples["response"][i], examples["history"][i]

                    prompt = self.tokenizer.build_prompt(query, history)

                    a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                                  max_length=data_args.max_source_length)
                    b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                                  max_length=data_args.max_target_length)

                    context_length = len(a_ids)
                    input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
                    labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

                    pad_len = max_seq_length - len(input_ids)
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                    labels = labels + [self.tokenizer.pad_token_id] * pad_len
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)

            return model_inputs

        '''训练文件, 验证文件'''
        train_df, validation_df = load_csv_data(data_args)
        ds_train_raw = datasets.Dataset.from_pandas(train_df)
        ds_val_raw = datasets.Dataset.from_pandas(validation_df)

        ds_train = ds_train_raw.map(
            preprocess,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=ds_train_raw.column_names
        )

        ds_val = ds_val_raw.map(
            preprocess,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=ds_val_raw.column_names
        )

        # 管道
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=False
        )

        dl_train = DataLoader(ds_train, batch_size=1,
                              num_workers=data_args.preprocessing_num_workers, shuffle=True, collate_fn=data_collator)
        dl_val = DataLoader(ds_val, batch_size=1,
                            num_workers=data_args.preprocessing_num_workers, shuffle=False, collate_fn=data_collator)
        return dl_train, dl_val
