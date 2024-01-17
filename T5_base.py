from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import numpy as np
import evaluate
from torch.optim import AdamW
from utils import EarlyStoppingCallback

def tokenize_function(examples):
    # T5 expects a task prefix, for example, "classify: "
    return tokenizer("classify: " + examples["sentence"], padding="max_length", max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset_list = ['qasc', 'wiki_qa', 'quartz', 'paws', 'story_cloze', 'winogrande', 'wsc']
dataset_name = dataset_list[5]
# Load WSC dataset
if(dataset_name == 'wsc'):
    dataset = load_dataset("super_glue", 'wsc')
elif(dataset_name == 'paws'):
    dataset = load_dataset("paws", "labeled_final")
elif(dataset_name == 'story_cloze'):
    dataset = load_dataset("story_cloze",  data_dir="<path/to/manual/data>")
elif(dataset_name == 'winogrande'):
    dataset = load_dataset("winogrande", "winogrande_xl")
else:
    dataset = load_dataset(dataset_name)

print(dataset)
print(dataset['train'][0])
exit()
tokenizer = T5Tokenizer.from_pretrained("t5-base")
if(dataset_name in ['qasc']):
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

shuffle_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
shuffle_eval_dataset = tokenized_datasets["validation_mismatched"].shuffle(seed=42)

model = T5ForConditionalGeneration.from_pretrained("t5-base")

metric = evaluate.load("accuracy")
# Create an instance of your custom callback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

training_args = TrainingArguments(
    output_dir="train_" + dataset_name, 
    evaluation_strategy="steps",
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    learning_rate=0.0001,
    max_steps=75000,
    num_train_epochs=10000000.0,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
)

#print(training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=shuffle_train_dataset,
    eval_dataset=shuffle_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

trainer.train()
