from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
from peft import get_peft_model, AdaLoraConfig, TaskType
from torchkeras import KerasModel
from accelerate import Accelerator


class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        # loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"], labels=batch["labels"]).loss

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics (stateful metrics)
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics


# 仅仅保存lora可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator=None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)


def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path, 'adapter_model.bin')), strict=False)
    self.from_scratch = False


KerasModel.StepRunner = StepRunner
KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt

# lr = 5e-3
# batch_size = 1
# gradient_accumulation_steps = 16  # 梯度累积


class Trainer:

    def __init__(self, model, lora_r=32, lora_alpha=32, lora_dropout=0.1):
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["query_key_value"]
        )

        peft_model = get_peft_model(model, peft_config)
        peft_model.is_parallelizable = True
        peft_model.model_parallel = True
        peft_model.print_trainable_parameters()
        self.peft_model = peft_model

    '''
    @gradient_accumulation_steps: 梯度累积 default: 16 
    '''
    def train(self, ckpt_path, dl_train, dl_val, lr=5e-3, epochs=30, gradient_accumulation_steps=16):
        optimizer = torch.optim.AdamW(self.peft_model.parameters(), lr=lr)
        keras_model = KerasModel(self.peft_model, loss_fn=None, optimizer=optimizer)

        keras_model.fit(train_data=dl_train,
                        val_data=dl_val,
                        epochs=epochs,
                        patience=20,
                        monitor='val_loss',
                        mode='min',
                        ckpt_path=ckpt_path,
                        mixed_precision='fp16',
                        gradient_accumulation_steps=gradient_accumulation_steps
                        )

    def eval(self, dl_val, quiet=False):
        optimizer = torch.optim.AdamW(self.peft_model.parameters())
        keras_model = KerasModel(self.peft_model, loss_fn=None, optimizer=optimizer)
        keras_model.evaluate(dl_val, quiet=quiet)

