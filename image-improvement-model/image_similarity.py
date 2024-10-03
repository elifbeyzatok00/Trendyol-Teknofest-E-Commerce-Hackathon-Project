import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

def load_vgg_model():
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    return model

# Convert images to features vectors with VGG16
def extract_features(model, image):
    # Görüntü boyutunu VGG16 için uygun hale getir ve ön işlem uygula
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    image_array = preprocess_input(image_array)

    # Extract the features vectors
    features = model.predict(image_array)
    return features.flatten()

def calculate_cosine_similarity(user_features, hr_features):
    similarity = cosine_similarity([user_features], [hr_features])
    return similarity[0][0]

# Take the Image File From User
def load_user_image(filepath):
    return cv2.imread(filepath)

# Upload hr_images folder's images
def load_hr_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((filename, img))
    return images

# Show most similar image
def show_most_similar_image(user_img, best_match_img):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title("User Image")
    plt.imshow(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Most Similar Image")
    plt.imshow(cv2.cvtColor(best_match_img, cv2.COLOR_BGR2RGB))

    plt.show()

# Find Most Similar Image
def find_most_similar_image(user_image_path, hr_images_folder):
    vgg_model = load_vgg_model()

    # Upload User Image
    user_image = load_user_image(user_image_path)

    # Extract User's Image features
    user_features = extract_features(vgg_model, user_image)

    # Upload images from hr_images folder and calculate similarity scores
    hr_images = load_hr_images(hr_images_folder)
    similarity_scores = []
    
    for filename, hr_image in hr_images:
        hr_features = extract_features(vgg_model, hr_image)
        similarity = calculate_cosine_similarity(user_features, hr_features)
        similarity_scores.append((filename, similarity))

    # Sort images according to similarity scores and take the which one is have most similarity score
    best_match_filename, best_similarity = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[0]
    
    best_match_image = cv2.imread(os.path.join(hr_images_folder, best_match_filename))
    
    return user_image, best_match_image, best_match_filename, best_similarity