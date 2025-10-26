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
import subprocess
import ast


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)


def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    # Load model weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()
    return(predicted_class_index,output[index])


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


def compare_numbers(random_array_1, random_array_2, random_array_3, random_array_4):
    result = subprocess.run(
        [r"The absolute path of the judge.exe file", random_array_1, random_array_2, random_array_3, random_array_4],  # 将数组传递为空格分隔的字符串
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The absolute path of the original image
actual_image_absolute_path = "..."

# The absolute path to the folder where adversarial samples are saved
output_directory = "..."

# The absolute path of index file
index_file_absolute_path = "..."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# The absolute path of weight file for convolutional neural network model
weight_file_absolute_path = "..."

# The maximum degree of tampering of a single pixel
degree = 8

# Iterative step size
iteration_step_size = 0.001

# Calculate the maximum number of iterations
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

# The currently used convolutional neural network model
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

# Standardize the actual image
actual_image_transform_matrix_R = ((actual_image_channel_R / 255) - 0.485) / 0.229
actual_image_transform_matrix_G = ((actual_image_channel_G / 255) - 0.456) / 0.224
actual_image_transform_matrix_B = ((actual_image_channel_B / 255) - 0.406) / 0.225
####################################################################################################
# Perform category prediction on the original image
actual_image_index, x = predict_image_path(actual_image_absolute_path, index_file_absolute_path, weight_file_absolute_path, 0, model_current)
print(f"Top-1 category index number of the actual image:{actual_image_index}")

attack_index = 455
print(f"The target attack category is：{attack_index}")
####################################################################################################
# Mark whether the target attack was successful
flag = 0

# The first iteration image is the original image
iterative_image_path = actual_image_absolute_path

# Number of iterations
num_iterative = 0

# Perform forward iteration tampering
while flag == 0 and num_iterative <= max_num_iterative:

    num_iterative = num_iterative + 1
    print(f"Current iteration count：{num_iterative}")

    # Calculate the Top-1 category of the iterative image in this round
    iterative_image_top1_index, x = predict_image_path(iterative_image_path, index_file_absolute_path, weight_file_absolute_path, 0, model_current)
    print(f"The Top-1 category of the current iterative image：{iterative_image_top1_index}")

    # Calculate the pixel weight matrix of the current iterative image in the Top1 category
    iterative_image_pixel_weight_matrix_in_iterative_top1_label_R, iterative_image_pixel_weight_matrix_in_iterative_top1_label_G, iterative_image_pixel_weight_matrix_in_iterative_top1_label_B = pixel_weight_matrix_image_path(
        iterative_image_path, weight_file_absolute_path, iterative_image_top1_index, model_current
    )
    #################################################################################################
    # Calculate the pixel weight matrix of the current iterative image in the target category
    iterative_image_pixel_weight_matrix_in_attack_label_R, iterative_image_pixel_weight_matrix_in_attack_label_G, iterative_image_pixel_weight_matrix_in_attack_label_B = pixel_weight_matrix_image_path(
        iterative_image_path, weight_file_absolute_path, attack_index, model_current
    )
    ########################################################################################################
    # Calculate the RGB three channel matrix of iterative images
    image = Image.open(iterative_image_path)
    image = image.resize((224, 224))
    iterative_image = np.array(image)

    # The RGB three channel matrix of the image in this iteration
    iterative_image_channel_R = iterative_image[:, :, 0]
    iterative_image_channel_R = iterative_image_channel_R.astype(np.float64)
    iterative_image_channel_G = iterative_image[:, :, 1]
    iterative_image_channel_G = iterative_image_channel_G.astype(np.float64)
    iterative_image_channel_B = iterative_image[:, :, 2]
    iterative_image_channel_B = iterative_image_channel_B.astype(np.float64)

    # Calculate the standardized matrix of iterative image RGB three channels
    iterative_image_transform_matrix_R = (iterative_image_channel_R / 255 - 0.485) / 0.229
    iterative_image_transform_matrix_G = (iterative_image_channel_G / 255 - 0.456) / 0.224
    iterative_image_transform_matrix_B = (iterative_image_channel_B / 255 - 0.406) / 0.225

    # Calculate the contribution matrix of iterative images to the top 1 labels in the iteration
    iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R = iterative_image_transform_matrix_R * iterative_image_pixel_weight_matrix_in_iterative_top1_label_R
    iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G = iterative_image_transform_matrix_G * iterative_image_pixel_weight_matrix_in_iterative_top1_label_G
    iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B = iterative_image_transform_matrix_B * iterative_image_pixel_weight_matrix_in_iterative_top1_label_B

    # Calculate the contribution matrix of iterative images in the target category
    iterative_image_pixel_classification_contribution_matrix_in_attack_label_R = iterative_image_transform_matrix_R * iterative_image_pixel_weight_matrix_in_attack_label_R
    iterative_image_pixel_classification_contribution_matrix_in_attack_label_G = iterative_image_transform_matrix_G * iterative_image_pixel_weight_matrix_in_attack_label_G
    iterative_image_pixel_classification_contribution_matrix_in_attack_label_B = iterative_image_transform_matrix_B * iterative_image_pixel_weight_matrix_in_attack_label_B
    ############################################################################################################
    # Calculate the tampering matrix of iterative images based on the iteration step size
    iterative_image_transform_matrix_R_increase = iterative_image_transform_matrix_R.copy()
    iterative_image_transform_matrix_G_increase = iterative_image_transform_matrix_G.copy()
    iterative_image_transform_matrix_B_increase = iterative_image_transform_matrix_B.copy()

    iterative_image_transform_matrix_R_decrease = iterative_image_transform_matrix_R.copy()
    iterative_image_transform_matrix_G_decrease = iterative_image_transform_matrix_G.copy()
    iterative_image_transform_matrix_B_decrease = iterative_image_transform_matrix_B.copy()

    for i in range(224):
        for j in range(224):
            iterative_image_transform_matrix_R_increase[i][j] = iterative_image_transform_matrix_R_increase[i][j] + iteration_step_size
            iterative_image_transform_matrix_G_increase[i][j] = iterative_image_transform_matrix_G_increase[i][j] + iteration_step_size
            iterative_image_transform_matrix_B_increase[i][j] = iterative_image_transform_matrix_B_increase[i][j] + iteration_step_size

            iterative_image_transform_matrix_R_decrease[i][j] = iterative_image_transform_matrix_R_decrease[i][j] - iteration_step_size
            iterative_image_transform_matrix_G_decrease[i][j] = iterative_image_transform_matrix_G_decrease[i][j] - iteration_step_size
            iterative_image_transform_matrix_B_decrease[i][j] = iterative_image_transform_matrix_B_decrease[i][j] - iteration_step_size
    ##############################################################################################################
    # Define a matrix to retain tampered information
    change_matrix_R = iterative_image_channel_R.copy()
    change_matrix_R = change_matrix_R.astype(np.float64)

    change_matrix_G = iterative_image_channel_G.copy()
    change_matrix_G = change_matrix_G.astype(np.float64)

    change_matrix_B = iterative_image_channel_B.copy()
    change_matrix_B = change_matrix_B.astype(np.float64)

    # Perform positive absolute monotonic tampering
    for i in range(224):
        for j in range(224):
            ############################################################################################################################################################################################
            if iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j] < 0:

                if iterative_image_transform_matrix_R[i][j] > 0:
                    change_matrix_R[i][j] = math.ceil(
                        (iterative_image_transform_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                if iterative_image_transform_matrix_R[i][j] < 0:
                    change_matrix_R[i][j] = int(
                        (iterative_image_transform_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j] < 0:
                if iterative_image_transform_matrix_G[i][j] > 0:
                    change_matrix_G[i][j] = math.ceil(
                        (iterative_image_transform_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                if iterative_image_transform_matrix_G[i][j] < 0:
                    change_matrix_G[i][j] = int(
                        (iterative_image_transform_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j] < 0:
                if iterative_image_transform_matrix_B[i][j] > 0:
                    change_matrix_B[i][j] = math.ceil(
                        (iterative_image_transform_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                if iterative_image_transform_matrix_B[i][j] < 0:
                    change_matrix_B[i][j] = int(
                        (iterative_image_transform_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
            ############################################################################################################################################################################################
            if iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j] > 0:
                if iterative_image_transform_matrix_R[i][j] < 0:
                    change_matrix_R[i][j] = math.ceil(
                        (iterative_image_transform_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                if iterative_image_transform_matrix_R[i][j] > 0:
                    change_matrix_R[i][j] = int(
                        (iterative_image_transform_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)


            if iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j] > 0:
                if iterative_image_transform_matrix_G[i][j] < 0:
                    change_matrix_G[i][j] = math.ceil(
                        (iterative_image_transform_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                if iterative_image_transform_matrix_G[i][j] > 0:
                    change_matrix_G[i][j] = int(
                        (iterative_image_transform_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)


            if iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j] > 0:
                if iterative_image_transform_matrix_B[i][j] < 0:
                    change_matrix_B[i][j] = math.ceil(
                        (iterative_image_transform_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                if iterative_image_transform_matrix_B[i][j] > 0:
                    change_matrix_B[i][j] = int(
                        (iterative_image_transform_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)

    # Generate images that have been tampered with in this iteration
    image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
    image_rgb = image_rgb.astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)
    image_pil.save("target attack image.png")
##########################################################################################################
    # Perform forward calibration
    forward_image_optimize_path = "target attack image.png"
    image = Image.open(forward_image_optimize_path)
    image = image.resize((224, 224))
    forward_image_optimize = np.array(image)

    # The R, G, B three channel matrix of the positive optimized image
    forward_image_optimize_channel_R = forward_image_optimize[:, :, 0]
    forward_image_optimize_channel_R = forward_image_optimize_channel_R.astype(np.float64)
    forward_image_optimize_channel_G = forward_image_optimize[:, :, 1]
    forward_image_optimize_channel_G = forward_image_optimize_channel_G.astype(np.float64)
    forward_image_optimize_channel_B = forward_image_optimize[:, :, 2]
    forward_image_optimize_channel_B = forward_image_optimize_channel_B.astype(np.float64)

    # Calculate the standardized matrix of Xt+1
    forward_image_optimize_transform_matrix_R = ((forward_image_optimize_channel_R / 255) - 0.485) / 0.229
    forward_image_optimize_transform_matrix_G = ((forward_image_optimize_channel_G / 255) - 0.456) / 0.224
    forward_image_optimize_transform_matrix_B = ((forward_image_optimize_channel_B / 255) - 0.406) / 0.225

    # Calculate the pixel weight matrix of Xt+1 in the top 1 category of iterative images
    forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R, forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G, forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B = pixel_weight_matrix_image_path(forward_image_optimize_path, weight_file_absolute_path, iterative_image_top1_index, model_current)
    # Calculate the contribution matrix of Xt+1 in the top 1 category of iterative images
    forward_image_optimize_contribution_matrix_in_iterative_top1_label_R = forward_image_optimize_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R
    forward_image_optimize_contribution_matrix_in_iterative_top1_label_G = forward_image_optimize_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G
    forward_image_optimize_contribution_matrix_in_iterative_top1_label_B = forward_image_optimize_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B

    # Calculate the pixel weight matrix of Xt+1 in the target category
    forward_image_optimize_pixel_weight_matrix_in_object_label_R, forward_image_optimize_pixel_weight_matrix_in_object_label_G, forward_image_optimize_pixel_weight_matrix_in_object_label_B = pixel_weight_matrix_image_path(forward_image_optimize_path, weight_file_absolute_path, attack_index, model_current)
    # Calculate the contribution matrix of Xt+1 in the target category
    forward_image_optimize_contribution_matrix_in_object_label_R = forward_image_optimize_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_object_label_R
    forward_image_optimize_contribution_matrix_in_object_label_G = forward_image_optimize_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_object_label_G
    forward_image_optimize_contribution_matrix_in_object_label_B = forward_image_optimize_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_object_label_B

    # Calculate the contribution value matrix of the top 1 categories in the iterative image when X is the original image
    actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R = actual_image_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R
    actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G = actual_image_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G
    actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B = actual_image_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B

    # Calculate the contribution matrix on the target category when X is the original image
    actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_R = actual_image_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_object_label_R
    actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_G = actual_image_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_object_label_G
    actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_B = actual_image_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_object_label_B

    # Calculate the growth rate of the contribution value in the Top1 category of the iterative image after hypothesis correction
    if_optimize_growth_contribution_in_iterative_top1_label_R = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R - forward_image_optimize_contribution_matrix_in_iterative_top1_label_R
    if_optimize_growth_contribution_in_iterative_top1_label_G = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G - forward_image_optimize_contribution_matrix_in_iterative_top1_label_G
    if_optimize_growth_contribution_in_iterative_top1_label_B = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B - forward_image_optimize_contribution_matrix_in_iterative_top1_label_B

    # Calculate the growth rate of contribution value in the target category after hypothesis correction
    if_optimize_growth_contribution_in_object_label_R = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_R - forward_image_optimize_contribution_matrix_in_object_label_R
    if_optimize_growth_contribution_in_object_label_G = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_G - forward_image_optimize_contribution_matrix_in_object_label_G
    if_optimize_growth_contribution_in_object_label_B = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_B - forward_image_optimize_contribution_matrix_in_object_label_B

    if_optimize_growth_contribution_in_object_label = np.stack([
        if_optimize_growth_contribution_in_object_label_R,
        if_optimize_growth_contribution_in_object_label_G,
        if_optimize_growth_contribution_in_object_label_B
    ])

    if_optimize_growth_contribution_in_object_label = if_optimize_growth_contribution_in_object_label.flatten()
    file_name_txt = "if_optimize_growth_contribution_in_object_label.txt"
    with open(file_name_txt, 'w') as file:
        file.write(" ".join(map(str, if_optimize_growth_contribution_in_object_label)))


    if_optimize_growth_contribution_in_iterative_top1_label = np.stack([
        if_optimize_growth_contribution_in_iterative_top1_label_R,
        if_optimize_growth_contribution_in_iterative_top1_label_G,
        if_optimize_growth_contribution_in_iterative_top1_label_B
    ])
    if_optimize_growth_contribution_in_iterative_top1_label = if_optimize_growth_contribution_in_iterative_top1_label.flatten()
    file_name_txt = "if_optimize_growth_contribution_in_iterative_top1_label.txt"
    with open(file_name_txt, 'w') as file:
        file.write(" ".join(map(str, if_optimize_growth_contribution_in_iterative_top1_label)))


    forward_image_optimize_channel = np.stack([
        forward_image_optimize_channel_R,
        forward_image_optimize_channel_G,
        forward_image_optimize_channel_B
    ])
    # 将幅度矩阵保存为.txt文件
    forward_image_optimize_channel = forward_image_optimize_channel.flatten()
    file_name_txt = "forward_image_optimize_channel.txt"  # 文件名
    with open(file_name_txt, 'w') as file:
        # 将数组转换为字符串，以空格分隔
        file.write(" ".join(map(str, forward_image_optimize_channel)))


    actual_image_channel = np.stack([
        actual_image_channel_R,
        actual_image_channel_G,
        actual_image_channel_B
    ])
    # 将幅度矩阵保存为.txt文件
    actual_image_channel = actual_image_channel.flatten()
    file_name_txt = "actual_image_channel.txt"  # 文件名
    with open(file_name_txt, 'w') as file:
        # 将数组转换为字符串，以空格分隔
        file.write(" ".join(map(str, actual_image_channel)))

    random_array_1_path = "if_optimize_growth_contribution_in_object_label.txt"

    random_array_2_path = "if_optimize_growth_contribution_in_iterative_top1_label.txt"

    random_array_3_path = "forward_image_optimize_channel.txt"

    random_array_4_path = "actual_image_channel.txt"

    compare_result_str = compare_numbers(random_array_1_path, random_array_2_path, random_array_3_path, random_array_4_path)

    compare_result_list = ast.literal_eval(compare_result_str)
    forward_image_optimize_channel = np.array(compare_result_list, dtype=np.float32)


    # Generate images that have been tampered with in this iteration
    image_rgb = np.stack([forward_image_optimize_channel[0], forward_image_optimize_channel[1], forward_image_optimize_channel[2]], axis=-1)
    image_rgb = image_rgb.astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)
    image_pil.save("target attack image.png")

    # Calculate and query whether the current attack image has been successfully attacked
    attack_image_path = "target attack image.png"
    attack_image_index, x = predict_image_path(attack_image_path, index_file_absolute_path, weight_file_absolute_path, actual_image_index, model_current)
    if attack_image_index == attack_index:
        flag = 1
        iterative_image_path = f"{output_directory}\\attack image.png"
        shutil.copy(attack_image_path, iterative_image_path)
    else:
        iterative_image_path = f"{output_directory}\\attack image.png"
        shutil.copy(attack_image_path, iterative_image_path)