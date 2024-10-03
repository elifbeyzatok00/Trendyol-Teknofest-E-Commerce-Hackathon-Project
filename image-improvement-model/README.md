# Image Similarity Detection using VGG16

This project uses the pre-trained VGG16 model to extract feature vectors from images and compute the cosine similarity between a user-provided image and a collection of high-resolution (HR) images. The goal is to find the HR image that is most similar to the user's image.

## Requirements

To run this project, you will need to install the following libraries:

```bash
pip install numpy opencv-python tensorflow scikit-learn matplotlib
```

## Libraries Used
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy**
- **scikit-learn**
- **Matplotlib**

## Project Structures
#### ```load_vgg_model()```: Loads the VGG16 model, with the output taken from the 'fc1' layer (first fully connected layer), which provides a 4096-dimensional feature vector.
#### ```extract_features(model, image)```: Takes an image as input, resizes it to (224, 224) to match VGG16's expected input size, applies preprocessing, and returns a flattened feature vector from the VGG16 model.
#### ```calculate_cosine_similarity(user_features, hr_features)```: Uses cosine similarity to compare two feature vectors.
#### ```load_user_image(filepath)```: Reads the user’s image from the provided file path using OpenCV.
#### ```load_hr_images(folder_path)```: Loads all images from the specified folder and stores them in a list.
#### ```show_most_similar_image(user_img, best_match_img)```: Displays the user image and the most similar image side by side using Matplotlib.
#### ```find_most_similar_image(user_image_path, hr_images_folder)```: The main function that takes the path of a user's image and a folder of HR images, finds the most similar image from the folder, and returns the user's image, the most similar image, the filename of the best match, and the similarity score.

## Usage
**Load a User Image:** The image is loaded using the ```load_user_image()``` function.

**Load HR Images:** All images in the hr_images_folder are loaded using the ```load_hr_images()``` function.

**Extract Features:** Both the user’s image and the HR images are converted into 4096-dimensional feature vectors using the ```extract_features()``` function.

**Calculate Similarity:** The cosine similarity between the user's image and each HR image is calculated with ```calculate_cosine_similarity()```.

**Find Best Match:** The HR image with the highest similarity score is determined and displayed using ```show_most_similar_image()```.