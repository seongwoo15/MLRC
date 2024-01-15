from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from torch.optim import AdamW

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", max_length=512)

def tokenize_function_2(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=512)

def tokenize_function_3(examples):
    return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True, max_length=512)

def tokenize_function_4(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=512)

def tokenize_function_5(examples):
    return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset_list = ['wnli', 'sst2', 'rte', 'qnli', 'mrpc', 'cola', 'mnli', 'qqp']
dataset_name = 'wnli'

dataset = load_dataset("glue", dataset_name)
print(dataset)
print(dataset['train'][0])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
if(dataset_name in ['cola','sst2']):
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
elif(dataset_name in ['rte', 'wnli', 'mrpc']):
    tokenized_datasets = dataset.map(tokenize_function_2, batched=True)
elif(dataset_name == 'qnli'):
    tokenized_datasets = dataset.map(tokenize_function_3, batched=True)
elif(dataset_name == 'mnli'):
    tokenized_datasets = dataset.map(tokenize_function_4, batched=True)
elif(dataset_name == 'qqp'):
    tokenized_datasets = dataset.map(tokenize_function_5, batched=True)

shuffle_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
if(dataset_name == 'mnli'):
    shuffle_eval_dataset = tokenized_datasets["validation_mismatched"].shuffle(seed=42)
    num_labels = 3
else:
    shuffle_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
    num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = num_labels)

metric = evaluate.load("accuracy")

training_args = TrainingArguments(
    output_dir="train_" + dataset_name, 
    evaluation_strategy="epoch",
    num_train_epochs=1.0,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=shuffle_train_dataset,
    eval_dataset=shuffle_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()