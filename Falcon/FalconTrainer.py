import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import transformers
from Mysetting import wandb_api_key
import peft
from peft import LoraConfig, get_peft_model

os.environ["WANDB_API_KEY"] = wandb_api_key
import transformers
from transformers import EarlyStoppingCallback
import argparse
from trl import SFTTrainer
from Data.Data import prepare_data

# %%

class LLMTrainer():
    def __init__(self, model_name="tiiuae/falcon-7b", train_args=None, checkpoint_path=None):

        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.train_args = train_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # set the pad token to be the eos token for open ended generation
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add pad token to tokenizer
        self.bn_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_bit_compute_dtype=torch.float16, )
        if self.checkpoint_path is None:

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bn_config,
                device_map="auto",
                trust_remote_code=True,
            )

            self.lora_config = LoraConfig(
                r=64,  # scaling factor for the weight matrices
                lora_alpha=32,  # dimension of the low-rank matrices
                target_modules=["query_key_value",
                                "dense",
                                "dense_h_to_4h",
                                "dense_4h_to_h",
                                ],
                lora_dropout=0.05,
                bias="none",  # setting to 'none' for only training weight params instead of biases
                task_type="CAUSAL_LM"
            )
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()
            self.kbit_model = prepare_model_for_kbit_training(self.model)
            self.peft_model = get_peft_model(self.kbit_model, self.lora_config)
        else:
            self.peft_model = self.get_trained_model()
            self.peft_model.gradient_checkpointing_enable()

        self.print_trainable_parameters()
        self.default_training_args = vars(argparse.Namespace(num_train_epochs=1,
                                                             per_device_train_batch_size=2,
                                                             per_device_eval_batch_size=2,
                                                             gradient_accumulation_steps=1,
                                                             weight_decay=0.001,
                                                             warmup_ratio=0.03,
                                                             # warmup_steps=2,
                                                             group_by_length=True,
                                                             # max_steps=300,
                                                             learning_rate=2e-4,
                                                             max_grad_norm=0.3,
                                                             fp16=True,
                                                             bf16=False,
                                                             logging_steps=250,
                                                             optim="paged_adamw_32bit",
                                                             lr_scheduler_type='constant',
                                                             report_to=["wandb"],
                                                             load_best_model_at_end=True,
                                                             metric_for_best_model="eval_loss",
                                                             overwrite_output_dir=True,
                                                             output_dir="experiments",
                                                             save_strategy="steps",
                                                             evaluation_strategy="steps",
                                                             seed=42,))

    def get_trained_model(self):
        model_path = self.checkpoint_path
        config = peft.PeftConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            quantization_config=self.bn_config,
            device_map="auto",
        )

        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        trained_model = peft.PeftModel.from_pretrained(
            self.model, model_path, is_trainable=True)

        return trained_model

    def generate_answer(self, generate_args):
        print(
            "----------------------------------Question:-------------------------------------------------------------------")
        print(generate_args["question"])

        encoding = self.tokenizer("<question>:" + generate_args["question"] + "<answer>:", return_tensors="pt").to(
            self.device)
        output = self.peft_model.generate(input_ids=encoding.input_ids,attention_mask=encoding.attention_mask,max_new_tokens=50, top_k=10, top_p=0.95)

        print(
            "-------------------------------------------------------------------output-------------------------------------------------------------------")
        ans = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(ans)
        print(
            "-------------------------------------------------------------------Answer-------------------------------------------------------------------")
        print(generate_args["correct_answer"])

        return ans

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def prepare_data(self):
        train_dataset, val_dataset, test_dataset = prepare_data(self.tokenizer)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train(self, train_dataset, val_dataset):
        self.peft_model.use_cache = False


        trainer = SFTTrainer(
            model=falcon_trainer.model,
            peft_config=falcon_trainer.lora_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            max_seq_length=None,
            args=transformers.TrainingArguments(**self.default_training_args),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            packing=True,
        )

        trainer.train()
        trainer.model.save_pretrained("saved-models/falcon-7b-NOT-FOUND-REMOVED-1epoch-lr4e-4")


# %%
if __name__ == "__main__":
    falcon_trainer = LLMTrainer("tiiuae/falcon-7b",checkpoint_path=None)
    train_dataset, val_dataset, test_dataset = prepare_data(falcon_trainer.tokenizer, with_tags=False)

    falcon_trainer.train(train_dataset, val_dataset)

    generate_args = {}
    generate_args["question"] = test_dataset[0]["q"]
    generate_args["correct_answer"] = test_dataset[0]["a"]

    ans = falcon_trainer.generate_answer(generate_args)

    # %%

    generate_args = {}
    generate_args["question"] = test_dataset[0]["q"]
    generate_args["correct_answer"] = test_dataset[0]["a"]
    falcon_trainer.generate_answer(generate_args)


