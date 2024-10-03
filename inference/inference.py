import gradio as gr
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from PIL import Image as img
import cv2
import numpy as np
import os
import sys

# image_similarity modülünü içe aktarın
sys.path.append(os.path.abspath('../image-improvement-model/'))
import image_similarity

# Ürün adı oluşturma modeli yükleme
product_name_model_path = "../find-product-name-model/fine_tuned_savasy/mt5-mlsum-turkish-summarization"
product_name_tokenizer = MT5Tokenizer.from_pretrained(product_name_model_path)
product_name_model = MT5ForConditionalGeneration.from_pretrained(product_name_model_path)

# Metin iyileştirme modeli yükleme
text_improvement_model_path = "../text-improvement-model/fine_tuned_ozcangundes/mt5-small-turkish-summarization"
text_improvement_tokenizer = MT5Tokenizer.from_pretrained(text_improvement_model_path)
text_improvement_model = MT5ForConditionalGeneration.from_pretrained(text_improvement_model_path)

# Ürün adı oluşturma fonksiyonu
def generate_product_name(input_text):
    inputs = product_name_tokenizer.encode("generate name: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = product_name_model.generate(inputs, max_length=64, num_beams=5, early_stopping=True)
    generated_name = product_name_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_name

# Metin iyileştirme fonksiyonu
def improve_text(input_text):
    inputs = text_improvement_tokenizer.encode("improve: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = text_improvement_model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    improved_text = text_improvement_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return improved_text

# Görsel karşılaştırma fonksiyonu
def image_compare(image_file):
    user_image_path = image_file
    hr_images_folder = "../image-improvement-model/hr_images"
    
    # image_similarity.py dosyasındaki find_most_similar_image fonksiyonunu çağırın
    _, best_match_image, best_match_filename, best_similarity = image_similarity.find_most_similar_image(user_image_path, hr_images_folder)
    
    # En benzer resmi Gradio formatında döndürün (RGB formatına çevirin)
    best_match_image = cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB)
    
    # Sonucu metin olarak formatla
    result_text = f"En benzer resim: {best_match_filename}, Cosine Benzerlik Skoru: {best_similarity:.4f}"
    
    return best_match_image, result_text

# Gradio arayüz fonksiyonu
def process(input_text, image_file):
    results = {}
    image_result = None
    similarity_info = None
    
    if input_text:
        results['Generated Product Name'] = generate_product_name(input_text)
        results['Improved Text'] = improve_text(input_text)
    
    if image_file:
        similar_image, similarity_info = image_compare(image_file)
        image_result = similar_image
    
    return results, image_result, similarity_info

# Gradio arayüzü
demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Textbox(lines=2, placeholder="Lütfen ürün tanımlayan kısa bir açıklama girin", label="Ürün Açıklaması"),
        gr.Image(type="filepath", label="Resim Yükle")
    ],
    outputs=[
        gr.JSON(label="Sonuçlar"),
        gr.Image(type="numpy", label="En Benzer Resim"),
        gr.Textbox(label="Benzerlik Sonucu")
    ],
    title="Gen-AREL Ürün Adı ve Görsel Benzerliği Arayüzü",
    description="Bu arayüz ürün adı oluşturma, metin iyileştirme ve benzer görsel bulma işlemleri yapar."
)

# Uygulamayı başlat
if __name__ == "__main__":
    demo.launch(share=True)