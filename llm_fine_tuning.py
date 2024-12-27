import pandas as pd
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Define token limit
TOKEN_LIMIT = 512

def create_aggregates(df, tokenizer, token_limit, num_aggregates=30):
    import random
    from tqdm import tqdm
    
    aggregates = []
    
    # Group by MatchID and PeriodID
    for (match_id, period_id), group in tqdm(df.groupby(['MatchID', 'PeriodID']), desc="Aggregating Tweets"):
        tweets = group['Tweet'].tolist()
        event_type = group['EventType'].iloc[0]  # Binary target for the period

        # Generate aggregates for this group
        for _ in range(num_aggregates):
            random.shuffle(tweets)  # Shuffle tweets for randomness
            aggregate = ""
            token_count = 0

            for tweet in tweets:
                # Tokenize tweet and count tokens
                tokenized_tweet = tokenizer.encode(tweet, add_special_tokens=False)
                if token_count + len(tokenized_tweet) > token_limit:
                    break  # Stop adding tweets if token limit is reached
                
                # Add the tweet to the aggregate
                aggregate += tweet + " "
                token_count += len(tokenized_tweet)

            # Save the aggregate and its label
            aggregates.append({
                'text': aggregate.strip(),
                'label': event_type,
                'match_id': match_id,
                'period_id': period_id
            })
    
    return pd.DataFrame(aggregates)


df = pd.read_csv('cleaned_llm_tweets.csv')
df.dropna(subset=['Tweet'], inplace=True)

aggregates_df = create_aggregates(df, tokenizer, TOKEN_LIMIT)

# Split into train and validation sets
train_df, val_df = train_test_split(aggregates_df, test_size=0.2, random_state=42)

# Prepare Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",  # Pad to max length
        truncation=True,       # Truncate if longer than model's max length
        max_length=TOKEN_LIMIT # Ensure length doesn't exceed model's limit
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Initialize model
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', num_labels=2, ignore_mismatched_sizes=True)

# Training arguments

training_args = TrainingArguments(
    output_dir="./results",  # Save checkpoints to local /tmp directory
    save_steps=500,             # Save every 500 steps
    save_total_limit=3,         # Keep the last 3 checkpoints
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",    # Save logs to local /tmp directory
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the trained model
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')