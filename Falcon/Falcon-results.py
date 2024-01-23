# %%
import numpy as np
from tqdm import tqdm
import FalconTrainer

import FalconTrainer as ft


import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import transformers
print("Python interpreter path:", sys.executable)
print ("transformers version:", transformers.__version__)

# %%
falcon_trainer = ft.LLMTrainer("tiiuae/falcon-7b")
# %%
train_dataset, val_dataset, test_dataset = ft.prepare_data(falcon_trainer.tokenizer, with_tags=False)


# %%
print("train size:", train_dataset.data.shape)
print("val size:", val_dataset.data.shape)
print("test size:", test_dataset.data.shape)
train_dataset.data.head()
# %%


# model_path="falcon-7b-trained-question-answer-note-tag"
# model_path="falcon-7b-trained-question-answer-note2"
model_path = "saved-models/saved-models/falcon-7b-NOT-FOUND-REMOVED-1epoch-lr4e-4"
config = peft.PeftConfig.from_pretrained(model_path)
trained_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=falcon_trainer.bn_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = falcon_trainer.tokenizer
tokenizer.pad_token = tokenizer.eos_token
trained_model = peft.PeftModel.from_pretrained(
    trained_model, model_path)
falcon_trainer.peft_model = trained_model

# %%
prediction = []
for i in range(len(test_dataset)):
    print("{}/{}".format(i,len(test_dataset)))
    q = test_dataset[i]['q']
    a = test_dataset[i]['a']
    question = test_dataset.data['question'][i]
    encoding = falcon_trainer.tokenizer("<question>:" + question + "<answer>:", return_tensors="pt").to(falcon_trainer.device)
    output = falcon_trainer.peft_model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
                                                max_new_tokens=30,top_p=0.95, do_sample=True, num_return_sequences=1,)
    output = falcon_trainer.tokenizer.decode(output[0], skip_special_tokens=True)

    print(">> generated:", output[output.find("<answer>:"):].replace("<answer>:", ""))
    print(">> answer:", a)
    print("--------{:<}-----------".format(test_dataset.data.iloc[i]['report_type']))
    prediction.append(output[output.find("<answer>:"):].replace("<answer>:", ""))

    # print(">> generated:", output[output.find("<answer>:"):].replace("<answer>:",""))
# %%
# %%
test_dataset.data.columns

# %%
import pandas as pd
#
result_pd = pd.DataFrame(
    {'question': test_dataset.data['question'],
     'answer': test_dataset.data['answer'],
     'prediction': prediction,
     "report_type": test_dataset.report_type,
     "structure": test_dataset.data.structure})
result_pd.to_excel('../results/saved-models/falcon-7b-NOT-FOUND-REMOVED-1epoch-lr4e-4.xlsx', index=False)
# %%
