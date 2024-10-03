# Inference Folder

This folder contains the necessary scripts and dependencies to run the integrated models and algorithms for product name generation, text improvement, and image similarity comparison through a Gradio interface.

## Files Overview

### 1. **inference.py**

- This script brings together the product name generation model, text improvement model, and image similarity algorithm into a single Gradio interface. It allows users to input product descriptions and upload images, generating both a product name and improved text for the description, as well as finding visually similar images.

**Key Functions:**

- **generate_product_name(input_text):** Generates a product name from a short description using the fine-tuned MT5 model.
- **improve_text(input_text):** Enhances a given product description using another MT5 model.
- **image_compare(image_file):** Finds the most visually similar image from a predefined folder of high-resolution images, using cosine similarity.

**Gradio Interface:**

- Accepts two inputs: a product description (text) and an image.
- Outputs:
  - A JSON object containing the generated product name and improved description.
  - The most similar image (if an image was provided).
  - A text description of the similarity score.

### 2. **requirements.txt**

- Lists the dependencies required to run the inference pipeline.

```
transformers
torch --extra-index-url https://download.pytorch.org/whl/cu124
torchvision --extra-index-url https://download.pytorch.org/whl/cu124
torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
sentencepiece
datasets
pandas
accelerate>=0.26.0
gradio
```

## Usage

To run the inference pipeline, ensure that all dependencies are installed. You can install them by running:

```bash
pip install -r requirements.txt
```

Once the dependencies are installed, run the Gradio interface by executing `inference.py`:

```bash
python inference.py
```

This will launch a local server with a Gradio interface, allowing you to input product descriptions and images for product name generation, text improvement, and image comparison.
