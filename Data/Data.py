import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
from sklearn.model_selection import train_test_split



class FalconDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        # self.encodings = tokenizer(data['input_text'].tolist(), padding=True,truncation=True, return_tensors='pt')
        self.encodings = tokenizer(data['input_text'].tolist(), padding=True, truncation=True, return_tensors='pt')
        self.label_encodings = tokenizer(data['answer'].tolist(), padding=True, truncation=True, return_tensors='pt')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item["labels"] = self.label_encodings["input_ids"][idx]
        item["q"] = self.data.iloc[idx]["question"]
        item["a"] = self.data.iloc[idx]["answer"]
        return item

    def __len__(self):
        return self.data.shape[0]
def clean_text(pandas_column):
    pandas_column = pandas_column.str.replace('\xa0', ' ')
    pandas_column = pandas_column.str.replace('\n', ' ')
    pandas_column = pandas_column.str.replace('•', ' ')
    pandas_column = pandas_column.str.lower()
    pandas_column = pandas_column.str.replace('cc:', ' ')
    pandas_column = pandas_column.str.replace('\s[\s]+', ' ', regex=True)
    pandas_column = pandas_column.str.replace('\s([;?.!"](;?:\s$))', r'\1', regex=True)
    pandas_column = pandas_column.str.replace('\.[\s]*\.', '.', regex=True)

    return pandas_column

def generate_prompt(df, with_tags=False):

    human = " <<{} report>>: {}  what is the integrity of the {} tendon?"
    assistant = "{}"
    prompts = []
    structures = ["Supraspinatus", "Infraspinatus", "Subscapularis", "Biceps", "Labrum"]
    for _, row in df.iterrows():
        if not with_tags:
            for s in structures:
                # -- radiology report prompt
                question = human.format("radiology", (row["Rad_Report"]), s)
                answer = assistant.format((row["Rad_" + s]).strip())
                input_text = ("<question>:" + question + "<answer>:" + answer).strip()
                prompts.append({"question": question, "answer": answer, "input_text": input_text, 'report_type': 'Rad',
                                "structure": s})

                # -- operation report prompt

                question = human.format("operation", row["Op_Report"], s)
                answer = assistant.format(row["Op_" + s].strip())
                input_text = ("<question>:" + question + "<answer>:" + answer).strip()

                prompts.append({"question": question, "answer": answer, "input_text": input_text, 'report_type': 'Op',
                                "structure": s})
        else:
            s = "Supraspinatus"
            question = human.format(s, "radiology", row["Rad_Report"])
            answer = assistant.format(row["Rad_Supra_Discript"])
            input_text = ("<question>:" + question + "<answer>:" + answer).strip()
            prompts.append(
                {"question": question, "answer": answer, "input_text": input_text, "text_ans": row["Rad_Supraspinatus"],
                 'report_type': 'Rad', "structure": s})

            # -- operation report prompt

            question = human.format(s, "operation", row["Op_Report"])
            answer = assistant.format(row["Op_Supra_Discript"])
            input_text = ("<question>:" + question + "<answer>:" + answer).strip()
            prompts.append(
                {"question": question, "answer": answer, "input_text": input_text, "text_ans": row["Op_Supraspinatus"],
                 'report_type': 'Op', "structure": s})
    # remove row when answer is "not found"
    prompts = pd.DataFrame(prompts)
    prompts = prompts[prompts.answer != "not found"]
    prompts.reset_index(drop=True, inplace=True)


    return prompts


def clean_data(data, with_tags=False):
    data.dropna(subset=["Rad_Report"], inplace=True)
    data.dropna(subset=["Op_Report"], inplace=True)
    if with_tags:
        data.drop(['DeMRN', 'DeAcc', 'Op_Acc'], axis=1, inplace=True)
        structures = ["Supraspinatus"]
        if "Rad_Supra_Number" in data.columns:
            data.drop(['Rad_Supra_Number'], axis=1, inplace=True)
        if "Op_Supra_Number" in data.columns:
            data.drop(['Op_Supra_Number'], axis=1, inplace=True)
        data.dropna(subset=["Op_Supra_Discript"], inplace=True)
        data.dropna(subset=["Rad_Supra_Discript"], inplace=True)
    else:
        data.drop(['DeMRN', 'DeAcc', 'Op_Acc'], axis=1, inplace=True)
        structures = ["Supraspinatus", "Infraspinatus", "Subscapularis", "Biceps", "Labrum"]
        data.drop(data[(data.Rad_Biceps.isnull()) & (data.Rad_Labrum.isnull()) & (data.Rad_Subscapularis.isnull()) & (
            data.Rad_Infraspinatus.isnull()) & (data.Rad_Supraspinatus.isnull())
                       & (data.Op_Biceps.isnull()) & (data.Op_Labrum.isnull()) & (data.Op_Subscapularis.isnull()) & (
                           data.Op_Infraspinatus.isnull()) & (data.Op_Supraspinatus.isnull())].index, inplace=True)

    for s in structures:
        data.loc[data["Rad_" + s].isnull(), "Rad_" + s] = "not found"
        data.loc[data["Op_" + s].isnull(), "Op_" + s] = "not found"

    for c in data.columns:
        data[c] = clean_text(data[c])

    return data


def prepare_data(tokenizer, with_tags=False):
    if with_tags:
        data = pd.read_excel("data_with_tags3.xlsx")
    else:
        data = pd.read_excel("data3.xlsx")

    data = clean_data(data, with_tags)

    train, test = train_test_split(data, test_size=0.4, random_state=42, shuffle=True)
    test, val = train_test_split(test, test_size=0.5, random_state=42, shuffle=True)

    train_df = generate_prompt(train, with_tags=with_tags)
    val_df = generate_prompt(val, with_tags=with_tags)
    test_df = generate_prompt(test, with_tags=with_tags)

    train_dataset = FalconDataset(train_df, tokenizer)
    val_dataset = FalconDataset(val_df, tokenizer)
    test_dataset = FalconDataset(test_df, tokenizer)

    return train_dataset, val_dataset, test_dataset
