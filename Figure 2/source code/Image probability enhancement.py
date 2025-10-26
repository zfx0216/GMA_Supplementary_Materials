import json
import shutil
import matplotlib.pyplot as plt
from torchvision.models import alexnet
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

def predict_image_path_specified_class_probability(image_path, index_path, weight_path, index, model_cnn):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Load the class index mapping
    with open(index_path, "r") as f:
        class_indict = json.load(f)

    # Initialize the model and load weights
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()

        # Apply softmax to get probabilities
        classification_probability = torch.softmax(output, dim=0)

    # Get the predicted class index and the probability for the specified class
    predicted_class_index = torch.argmax(classification_probability).item()
    specified_class_probability = classification_probability[index].item()

    return specified_class_probability


# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# Calculate the pixel weight matrix of a single specified image with category index number "index"
def pixel_weight_matrix_image_path(image_path, weight_path, index, model_cnn):
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Create model
    model = model_cnn(num_classes=1000).to(device)

    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    classification_probability = torch.softmax(output, dim=0)

    top_probs, top_indices = torch.topk(classification_probability, 3)
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, index]
    pred_score.backward(retain_graph=True)
    gradients = img.grad

    channel_r = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_g = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_b = gradients[0, 2, :, :].cpu().detach().numpy()

    return channel_r, channel_g, channel_b


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The absolute path of the original image
actual_image_absolute_path = "..."

# Absolute path of weight file
weight_file_absolute_path = "..."

# The absolute path of index file
index_file_absolute_path = "..."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# The absolute path of the folder where the result image is saved
output_directory = "..."

attack_index = 455
flag = 0
iterative_image_path = actual_image_absolute_path
designated_probability_value = 0.5
num_iterative = 0
degree = 8
model_current = alexnet

image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
actual_image = np.array(image)

# The R, G, B three channel matrix of the actual image
actual_image_channel_R = actual_image[:, :, 0]
actual_image_channel_R = actual_image_channel_R.astype(np.float64)
actual_image_channel_G = actual_image[:, :, 1]
actual_image_channel_G = actual_image_channel_G.astype(np.float64)
actual_image_channel_B = actual_image[:, :, 2]
actual_image_channel_B = actual_image_channel_B.astype(np.float64)


while designated_probability_value < 0.95:
    num_iterative = num_iterative + 1
    iterative_image_pixel_weight_matrix_for_attack_category_R, iterative_image_pixel_weight_matrix_for_attack_category_G, iterative_image_pixel_weight_matrix_for_attack_category_B = pixel_weight_matrix_image_path(
        iterative_image_path, weight_file_absolute_path, attack_index, model_current
    )
    ###########################################################################################################
    image = Image.open(iterative_image_path)
    image = image.resize((224, 224))
    iterative_image = np.array(image)

    iterative_image_channel_R = iterative_image[:, :, 0]
    iterative_image_channel_R = iterative_image_channel_R.astype(np.float64)
    iterative_image_channel_G = iterative_image[:, :, 1]
    iterative_image_channel_G = iterative_image_channel_G.astype(np.float64)
    iterative_image_channel_B = iterative_image[:, :, 2]
    iterative_image_channel_B = iterative_image_channel_B.astype(np.float64)

    iterative_image_standardized_matrix_R = ((iterative_image_channel_R / 255) - 0.485) / 0.229
    iterative_image_standardized_matrix_G = ((iterative_image_channel_G / 255) - 0.456) / 0.224
    iterative_image_standardized_matrix_B = ((iterative_image_channel_B / 255) - 0.406) / 0.225

    # Iteration step size
    iteration_step_size = 0.001
    #############################################################################################
    standardized_matrix_R_increase = iterative_image_standardized_matrix_R.copy()
    standardized_matrix_G_increase = iterative_image_standardized_matrix_G.copy()
    standardized_matrix_B_increase = iterative_image_standardized_matrix_B.copy()

    addend = 0
    for i in range(224):
        for j in range(224):
            standardized_matrix_R_increase[i][j] = standardized_matrix_R_increase[i][j] + iteration_step_size
            standardized_matrix_G_increase[i][j] = standardized_matrix_G_increase[i][j] + iteration_step_size
            standardized_matrix_B_increase[i][j] = standardized_matrix_B_increase[i][j] + iteration_step_size
    ##############################################################################################
    standardized_matrix_R_decrease = iterative_image_standardized_matrix_R.copy()
    standardized_matrix_G_decrease = iterative_image_standardized_matrix_G.copy()
    standardized_matrix_B_decrease = iterative_image_standardized_matrix_B.copy()

    for i in range(224):
        for j in range(224):
            standardized_matrix_R_decrease[i][j] = standardized_matrix_R_decrease[i][j] - iteration_step_size
            standardized_matrix_G_decrease[i][j] = standardized_matrix_G_decrease[i][j] - iteration_step_size
            standardized_matrix_B_decrease[i][j] = standardized_matrix_B_decrease[i][j] - iteration_step_size
    ########################################################################################################################
    change_matrix_R = iterative_image_channel_R.copy()
    change_matrix_R = change_matrix_R.astype(np.float64)

    change_matrix_G = iterative_image_channel_G.copy()
    change_matrix_G = change_matrix_G.astype(np.float64)

    change_matrix_B = iterative_image_channel_B.copy()
    change_matrix_B = change_matrix_B.astype(np.float64)

    for i in range(224):
        for j in range(224):
            ##################################################################################################################################
            if iterative_image_standardized_matrix_R[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] > 0:
                change_matrix_R[i][j] = math.ceil((standardized_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)


            if iterative_image_standardized_matrix_G[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] > 0:
                change_matrix_G[i][j] = math.ceil((standardized_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] > 0:
                change_matrix_B[i][j] = math.ceil((standardized_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)

            #########################################################################################################

            if iterative_image_standardized_matrix_R[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] < 0:
                change_matrix_R[i][j] = int((standardized_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_standardized_matrix_G[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] < 0:
                change_matrix_G[i][j] = int((standardized_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] < 0:
                change_matrix_B[i][j] = int((standardized_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
            ########################################################################################################
            if iterative_image_standardized_matrix_R[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] > 0:
                change_matrix_R[i][j] = math.ceil((standardized_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_standardized_matrix_G[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] > 0:
                change_matrix_G[i][j] = math.ceil((standardized_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] > 0:
                change_matrix_B[i][j] = math.ceil((standardized_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
            ###########################################################################################################
            if iterative_image_standardized_matrix_R[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] < 0:
                change_matrix_R[i][j] = int((standardized_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_standardized_matrix_G[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] < 0:
                change_matrix_G[i][j] = int((standardized_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] < 0:
                change_matrix_B[i][j] = int((standardized_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)

    image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
    image_rgb = image_rgb.astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)
    image_pil.save("Targeted attack images.png")

    iterative_image_path = "Targeted attack images.png"
    designated_probability_value = predict_image_path_specified_class_probability(
        iterative_image_path, index_file_absolute_path, weight_file_absolute_path, attack_index, model_current)

    print(designated_probability_value)

success_image_path = f"{output_directory}\\(b) attack image X'.png"
shutil.copy(iterative_image_path, success_image_path)
