# Gerekli Kütüphaneleri Yüklemek için terminalden aşağıdaki pip install komutunu çalıştırın:
# pip install -r requirements.txt

# CUDA destekli versiyonlar için terminalden aşağıdaki pip install komutunu ayrıca çalıştırabilirsiniz:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Bu dosyayı çalıştırmak için terminal veya komut satırında şu komutu kullanabilirsiniz:
# python text_improvement_model_generater.py

import torch
from datasets import load_dataset, Dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model ve tokenizer yükleme
model_name = "ozcangundes/mt5-small-turkish-summarization"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)  # Değişiklik yapıldı

model.to(device)

# 1. JSON formatındaki veri setini yükle
def load_custom_dataset(json_file):
    # JSON formatındaki veri setini yükleme
    dataset = load_dataset('json', data_files=json_file)
    return dataset

# 2. Veri işleme (Tokenizasyon)
def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['output_text']
    
    # Tokenizasyon işlemi
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Veri setini yükle ve hazırla
dataset = load_custom_dataset("textImprove-io.json")  # JSON dosyasının yolunu belirt
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 3. Eğitim ve doğrulama veri setlerini ayırma
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 4. Eğitim argümanları
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,  # Maksimum kaç checkpoint kaydedilsin
)

# 5. Trainer objesi
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 6. Eğitimi başlat
trainer.train()

# 7. Modeli kaydetme
#trainer.save_model(f"./fine_tuned_{model_name}")
model_save_path = f"./fine_tuned_{model_name}"
# Modeli ve tokenizer'ı kaydet
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Test amaçlı inference (tek cümle için)
def generate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    output_sequences = model.generate(input_ids=inputs["input_ids"], max_new_tokens=100)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# Test yap (inference)
sample_input = "700 gram kuşburnu marmeladı, yüzde 100 doğal. Kapağını açtıktan sonra buzdolabında saklayın."
print("Girdi:", sample_input)
print("Model çıktısı:", generate_text(sample_input))
