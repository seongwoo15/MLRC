from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from torch.optim import AdamW
from utils import EarlyStoppingCallback
from transformers import BertConfig



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


def compute_metrics_acc(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def compute_metrics_acc_f1(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def compute_metrics_mcc(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return mcc_metric.compute(predictions=predictions, references=labels)


dataset_list = ['wnli', 'sst2', 'rte', 'qnli', 'mrpc', 'cola', 'mnli', 'qqp']
dataset_name = 'mrpc'

dataset = load_dataset("glue", dataset_name)
#print(dataset)
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

# Load the configuration for BERT, set the dropout rate to 0.1, and specify the number of labels
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.1, num_labels=num_labels)

# Now load the model with the updated configuration
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
mcc_metric = load_metric("matthews_correlation")
# Create an instance of your custom callback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)


training_args = TrainingArguments(
    output_dir="train_" + dataset_name, 
    evaluation_strategy="steps",
    save_strategy= 'steps',
    eval_steps=max(1,len(dataset['train'])//800),
    save_steps=max(1,len(dataset['train'])//800),
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    learning_rate=0.00005,
    num_train_epochs=4.0,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
)

print(training_args)
if(dataset_name in ['wnli', 'sst2', 'rte', 'qnli', 'mnli']):
    compute_metrics = compute_metrics_acc
elif(dataset_name in ['qqp','mrpc']):
    compute_metrics = compute_metrics_acc_f1
elif(dataset_name =='cola'):
    compute_metrics = compute_metrics_mcc
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=shuffle_train_dataset,
    eval_dataset=shuffle_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

trainer.train()
