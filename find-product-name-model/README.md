# Find Product Name Model

This project is designed to develop a product name generation model. The project fine-tunes a pre-trained MT5 model and utilizes a dataset for predicting product names.

## Files and Their Functions

| File Name                                  | Description                                                                                                          |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `evaluation_find_product_name_model.ipynb` | Code for model evaluation. Analyzes model performance using ROUGE metrics and visualizes the results in a graph.     |
| `find_product_name_model_generator.py`     | Code for fine-tuning and saving the pre-trained model. Includes a function for generating text for testing purposes. |
| `findProdName-io.json`                     | A dataset consisting of input and output pairs. Contains suitable product name suggestions for each input.           |
| `requirements.txt`                         | A file listing the dependencies required to run the project.                                                         |

## Usage

### Requirements

To install the necessary libraries for the project, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

For CUDA-supported versions:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Start Training

To train the model, execute the following command in the terminal or command line:

```bash
python find_product_name_model_generator.py
```

### Evaluate the Model

To evaluate the trained model, open and run the following Jupyter Notebook:

```bash
evaluation_find_product_name_model.ipynb
```

### Testing

You can see the product suggested by the model for a given input text using the `generate_text` function found in the `find_product_name_model_generator.py` file.
