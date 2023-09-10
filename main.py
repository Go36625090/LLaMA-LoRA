import logging
import os
import sys

import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, HfArgumentParser, Seq2SeqTrainingArguments
import torch

from arguments import ModelArguments, DataTrainingArguments
from process import Processor
from trainer import Trainer

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0"

max_seq_length = 512

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, trust_remote_code=True, device_map=device)
        prefix_state_dict = torch.load(os.path.join(
            model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len(
                    "transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        if model_args.model_parallel_mode:
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path, config=config, trust_remote_code=True, device_map=device)
            logger.info(
                "use model parallel mode to training Lora")
        else:
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path, config=config, trust_remote_code=True, device_map=device)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    # 缓存机制可以用来存储和重用某些计算结果，以加速连续解码的过程。然而，关闭缓存可能有助于节省内存，尤其是在处理长序列时
    model.config.use_cache = True
    # 梯度检查点是一种内存优化技术，它可以在一定程度上减少模型训练过程中的内存使用量，但可能会增加一些计算开销。这种技术特别适用于训练大型模型和处理长序列。
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Trainer(model, lora_r=model_args.lora_r)

    processor = Processor(tokenizer)
    train_data, validation_data = processor.preprocess(data_args)

    '''训练'''
    if training_args.do_train:
        trainer.train(training_args.output_dir, train_data, validation_data,
                      lr=training_args.learning_rate,
                      epochs=training_args.num_train_epochs,
                      gradient_accumulation_steps=training_args.gradient_accumulation_steps)
    '''评估'''
    if training_args.do_eval:
        trainer.eval(validation_data)


if __name__ == "__main__":
    main()
