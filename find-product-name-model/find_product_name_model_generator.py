# Run the following pip install command in the terminal to install the necessary libraries:
# pip install -r requirements.txt
# pip install tf-keras

# For CUDA-supported versions, you can run the following pip install command in the terminal:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# You can use the following command in the terminal or command line to run this file:
# python find_product_name_model_generator.py

import torch
from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "savasy/mt5-mlsum-turkish-summarization"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

model.to(device)

# 1. Load the dataset in JSON format
def load_custom_dataset(json_file):
    # Load the dataset in JSON format
    dataset = load_dataset('json', data_files=json_file)
    return dataset

# 2. Data processing (Tokenization)
def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['output_text']
    
    # Perform tokenization
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load and prepare the dataset
dataset = load_custom_dataset("findProdName-io.json")  # Specify the path to the JSON file
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 3. Split into training and validation datasets
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,  # It might be beneficial to try a lower number of epochs initially
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,  # Maximum number of checkpoints to keep
)

# 5. Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 6. Start training
trainer.train()

# 7. Save the model
trainer.save_model(f"./fine_tuned_{model_name}")

# Inference for testing (for a single sentence)
def generate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    output_sequences = model.generate(input_ids=inputs["input_ids"], max_new_tokens=50)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# Perform a test (inference)
sample_input = "700 gram kuşburnu marmeladı, yüzde 100 doğal. Kapağını açtıktan sonra buzdolabında saklayın."
print("Input:", sample_input)
print("Model output:", generate_text(sample_input))
