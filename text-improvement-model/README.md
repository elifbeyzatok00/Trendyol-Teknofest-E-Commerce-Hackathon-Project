# Text Improvement Model

## File Overview

| Filename                                  | Description                                                                                                          |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `evaluation_text_improvement_model.ipynb` | Jupyter Notebook for evaluating the performance of the fine-tuned model using ROUGE metrics and visualizing results. |
| `requirements.txt`                        | List of Python dependencies required for the project.                                                                |
| `text_improvement_model_generator.py`     | Script for fine-tuning the MT5 model and saving the trained model and tokenizer.                                     |
| `textImprove-io.json`                     | Dataset consisting of input-output pairs used for training the model.                                                |

## Overview

The Text Improvement Model is a fine-tuned MT5 (Multilingual T5) model designed to enhance product descriptions in Turkish. This repository includes code and resources for training and evaluating the model, allowing users to improve the quality of textual product descriptions.

## Installation

To set up the project, ensure you have Python installed and then install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

For CUDA support, you can install the GPU-compatible versions of PyTorch by running:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Usage

1. **Training the Model**:
   To train the model, run the following command in your terminal or command prompt:

   ```bash
   python text_improvement_model_generator.py
   ```

   This script will load the dataset, preprocess the data, train the model, and save the trained model and tokenizer.

2. **Evaluating the Model**:
   To evaluate the model's performance, run the Jupyter Notebook:

   ```bash
   jupyter notebook evaluation_text_improvement_model.ipynb
   ```

   This notebook will load the trained model, evaluate it on the validation dataset, and visualize the ROUGE scores and training/validation loss.

## Dataset Format

The dataset `textImprove-io.json` is formatted as a JSON array, where each entry contains:

- `input_text`: The original product description.
- `output_text`: The improved product description.

Example:

```json
{
  "input_text": "700 gram kuşburnu marmeladı, yüzde 100 doğal. Kapağını açtıktan sonra buzdolabında saklayın.",
  "output_text": "Bu %100 doğal kuşburnu reçeli, 700g’lık kavanozlarda mevcuttur ve katkı maddesi içermez, saf meyve lezzeti sunar. Ürünün tazeliğini sağlamak için kapağı açıldıktan sonra buzdolabında saklayın. Kahvaltıda veya tatlılar için kullanabilirsiniz."
}
```

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index): For providing the MT5 model and training utilities.
- [PyTorch](https://pytorch.org/): For the deep learning framework used in this project.
