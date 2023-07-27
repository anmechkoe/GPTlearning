import torch
import evaluate
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, AutoTokenizer


def training():

    # Initialize the Accelerator object for distributed training
    accelerator = Accelerator()
    
    # Load the pretrained model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    # Define the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)


    # Define a function to tokenize the text examples
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

        
    # Load the yelp review dataset and apply the tokenizer
    dataset = load_dataset("yelp_review_full")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    # Create data loaders for training and evaluation
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)
    
    # Prepare the model, optimizer, and data loaders for distributed training
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )
    
    # Set the number of epochs and the learning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
    )
    
    # Create a progress bar to track the training steps
    progress_bar = tqdm(range(num_training_steps))

    # Load the accuracy metric
    metric = evaluate.load("accuracy")
    
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # In training mode
        model.train()
        
        # Loop over the batches in the training data loader
        for batch in train_dataloader:
            # Forward pass: compute the model outputs and loss
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass: compute the gradients using Accelerator's backward method
            accelerator.backward(loss)
    
            # Update the model parameters and the learning rate
            optimizer.step()
            lr_scheduler.step()
            
            # Zero out the gradients for the next batch
            optimizer.zero_grad()
            
            # Update the progress bar
            progress_bar.update(1)
            
        # In evaluation mode
        model.eval()

        # Perform evaluation every epoch
        for batch in eval_dataloader:
            # Forward pass: compute the prediction digits
            outputs = model(**batch)
            # Gather all predictions and targets
            all_outputs, all_labels = accelerator.gather_for_metrics((outputs, batch['labels']))
            # Get the predictions from the logits
            all_logits = all_outputs.logits
            all_predictions = torch.argmax(all_logits, dim=-1)
            # Example of use with a *Datasets.Metric*
            metric.add_batch(predictions=all_predictions, references=all_labels)

        # Compute the metric value over all batches and update the progress bar
        progress_bar.set_postfix({'Accuracy': metric.compute()['accuracy']})


if __name__ == "__main__":
    training()
