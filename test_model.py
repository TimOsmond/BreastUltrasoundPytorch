"""
Load a saved model and run it. Output all data to Neptune.ai
"""
# Import required libraries
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # for dataframes
from sklearn.metrics import classification_report
import seaborn as sns  # for plotting
from sklearn.metrics import confusion_matrix, roc_auc_score
import neptune  # for neptune logging online
from neptune_login import api_token  # for neptune
from neptune.utils import stringify_unsupported  # for neptune utilities
from GPUtil import showUtilization as gpu_usage
from numba import cuda  # for clearing GPU memory

api = api_token


def free_gpu_cache():
    print("\nInitial GPU Usage")
    gpu_usage()
    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


free_gpu_cache()
print("\n1. resnet 18\n2. resnet 152\n3. alexnet\n4. vgg 11\n5. squeezenet v1\n6. densenet 121\n7. inception v3\n")
input_model = input("Which model is being tested: ")
if input_model == "1":
    model_name = "resnet18"
elif input_model == "2":
    model_name = "resnet152"
elif input_model == "3":
    model_name = "alexnet"
elif input_model == "4":
    model_name = "vgg11"
elif input_model == "5":
    model_name = "squeezenet"
elif input_model == "6":
    model_name = "densenet121"
elif input_model == "7":
    model_name = "inception_v3"
else:
    exit()

# Start Neptune run to log data_mammogram
run = neptune.init_run(
    project="tim-osmond/TESTS",
    api_token=api,
)
params = {
    "model name": model_name,
}
run["parameters"] = params

# Define the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the TorchScript model
model = torch.jit.load('saved_model.pt').to(device)
model.eval()

# Define the image transformations
if model_name == "resnet18":
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

elif model_name == "resnet152":
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

elif model_name == "alexnet":
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

elif model_name == "vgg11":
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

elif model_name == "squeezenet":
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

elif model_name == "densenet121":
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

elif model_name == "inception_v3":
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load the test data
test_dir = 'us_test_images'
test_images = []
test_labels = []
total_identity_malignant = int()
total_identity_normal = int()
for label in os.listdir(test_dir):
    for file in os.listdir(os.path.join(test_dir, label)):
        image_path = os.path.join(test_dir, label, file)
        test_images.append(image_path)
        if label == 'malignant':
            identity = 0
            total_identity_malignant += 1
        else:
            identity = 1
            total_identity_normal += 1
        test_labels.append(int(identity))


# Define the evaluation function
def evaluate(model, images, labels):
    y_true = np.array(labels)
    y_pred = np.zeros_like(y_true)

    for i, image_path in enumerate(images):
        # Load the image
        image = Image.open(image_path)

        # Apply the image transformations
        image = transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Run the model on the input
        with torch.no_grad():
            output = model(image.to(device)).cpu()

        # Get the predicted label
        predicted_label = torch.argmax(output).item()
        y_pred[i] = predicted_label

    # Compute the evaluation metrics
    accuracy = (y_pred == y_true).mean()
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovo')
    precision, recall, f1, _ = classification_report(y_true, y_pred, output_dict=True)['weighted avg'].values()

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['malignant', 'normal'], columns=['malignant', 'normal'])

    # confusion matrix
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1.8)

    group_totals = [total_identity_malignant, total_identity_malignant, total_identity_normal, total_identity_normal]
    group_names = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
    group_counts = ['{0: 0.0f}'.format(value) for value in cm.flatten()]
    group_counts_int = [int(i) for i in group_counts]
    group_percentages_list = [x / y for x, y in zip(group_counts_int, group_totals)]
    group_percentages = ['{0:.2%}'.format(value) for value in (group_percentages_list * 100)]

    labels = [f'{v1}\n\n {v2}\n\n {v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    heatmap = sns.heatmap(df_cm, annot=labels, cmap=plt.cm.Blues, fmt="")
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    plt.title("Confusion Matrix\n")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig('test_confusion_matrix.png')

    # Print the evaluation results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'ROC AUC Score: {roc_auc:.2f}')
    print(f'Confusion Matrix:\n {cm:}')

    # Export logs to Neptune
    run["results/confusion_matrix"] = stringify_unsupported(confusion_matrix)
    run["val/conf_matrix"].upload("test_confusion_matrix.png")  # Upload confusion matrix image to Neptune
    run["results/accuracy"] = stringify_unsupported(accuracy)
    run["results/precision"] = stringify_unsupported(precision)
    run["results/recall"] = stringify_unsupported(recall)
    run["results/F1"] = stringify_unsupported(f1)
    run["results/ROC"] = stringify_unsupported(roc_auc)


# Evaluate the model on the test data
evaluate(model, test_images, test_labels)

# Finish export to Neptune
run.stop()
