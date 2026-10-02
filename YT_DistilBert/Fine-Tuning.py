import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
import gc
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

if __name__ == '__main__':
    label_mapping = {'Negative': 0, 'Positive': 1}

    TrainSet = pd.read_csv('./PreprocecedDatasets/TrainSet.csv', keep_default_na=False)
    valSet = pd.read_csv('./PreprocecedDatasets/valSet.csv', keep_default_na=False)

    train_dataset = Dataset.from_pandas(TrainSet.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(valSet.reset_index(drop=True))

    del TrainSet, valSet
    gc.collect()

    Model_Name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(Model_Name)

    def tokenize_function(batch):
        return tokenizer(batch['FullText'], truncation=True, padding='max_length', max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    num_labels = len(label_mapping)
    model = DistilBertForSequenceClassification.from_pretrained(Model_Name, num_labels=num_labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,               
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,    
        num_train_epochs=4,               
        weight_decay=0.01,
        eval_strategy="epoch",            
        save_strategy="epoch",
        load_best_model_at_end=True,      
        metric_for_best_model="f1",
        logging_steps=100,
        bf16=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        report_to="none"                  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    TrainHistory = trainer.state.log_history

    train_loss = []
    train_steps = []
    val_loss = []
    val_steps = []
    
    for log in TrainHistory:
        if 'loss' in log:
            train_loss.append(log['loss'])
            train_steps.append(log['epoch'])
        if 'eval_loss' in log:
            val_loss.append(log['eval_loss'])
            val_steps.append(log['epoch'])
            
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_steps, train_loss, label="Training Loss", color="royalblue", lw=2)
    
    if val_loss:
        plt.plot(val_steps, val_loss, label="Validation Loss", color="crimson", marker="o", linestyle="--", lw=2)
    
    plt.title("DistilBERT Fine-Tuning Loss Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('./Figs/train_val_loss.png', dpi=300)
    plt.close()

    model.save_pretrained("./fine_tuned_youtube_sentiment")
    tokenizer.save_pretrained("./fine_tuned_youtube_sentiment")