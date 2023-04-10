"""
Using a menu structure to command the use of CNN models in training and validating from different directories.
Training can be changed from fine-tuning to feature extraction and batch number and epochs entered.
Neptune.ai is utilised for collecting data and drawing graphs.
"""

from __future__ import print_function
from __future__ import division
import torch  # for pytorch
import torch.nn as nn  # for neural network
import torch.optim as optim  # for optimizer
from neptune.types import File
import torchvision  # for vision
import matplotlib.pyplot as plt  # for plotting
import neptune  # for neptune logging online
from neptune.utils import stringify_unsupported  # for neptune utilities
from torchvision import datasets, models, transforms
from GPUtil import showUtilization as gpu_usage
from numba import cuda  # for clearing GPU memory
import pydicom as dicom  # for reading dicom files
import cv2  # for DICOM image processing
from pathlib import Path  # for making file paths
import time
import os
import copy
from torchvision.models import ResNet152_Weights, AlexNet_Weights, VGG11_BN_Weights, SqueezeNet1_0_Weights, \
    Inception_V3_Weights, DenseNet121_Weights, ResNet18_Weights  # for model weights
from torch.utils.data import DataLoader
import pandas as pd  # for dataframes
import seaborn as sns  # for plotting
from neptune_login import api_token  # for neptune

api = api_token

LOSS_FUNCTION = nn.CrossEntropyLoss
OPTIMIZER = optim.SGD
# Optimizer momentum - momentum original 0.9
MOMENTUM = 0.9
# learning rate original 0.001
# https://www.cs.toronto.edu/~lczhang/360/lec/w02/training.html
LEARNING_RATE = 0.001

# Convert dicom images from the dataset TODO
convert = input("Convert DICOM images to PNG or JPG format? Enter Y or N: ").upper()


def convert_dicom(dcm_path, new_image_path):
    images_path = os.listdir(dcm_path)
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(dcm_path, image))
        pixel_array_numpy = ds.pixel_array
        if not PNG:
            image = image.replace('.dcm', '.jpg')
        else:
            image = image.replace('.dcm', '.png')
        # Create the new folder if it doesn't exist
        Path(new_image_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(new_image_path, image), pixel_array_numpy)
        if n % 50 == 0:
            print('{} image converted'.format(n))


if convert == "Y":
    data_dir = input("Enter the path to the dataset folder: ")
    file_type = input("Enter the file type to convert to\n1 PNG\n2 JPG ")
    # make it True if PNG format is required
    if file_type == "1":
        PNG = True
    else:
        PNG = False

    # Specify the .dcm folder path
    train_normal = data_dir + "/train/normal"
    # Specify the output jpg/png folder path
    path_train_normal = data_dir + "/jpg/train/normal"
    convert_dicom(train_normal, path_train_normal)

    train_malignant = data_dir + "/train/malignant"
    path_train_malignant = data_dir + "/jpg/train/malignant"
    convert_dicom(train_malignant, path_train_malignant)

    val_normal = data_dir + "/val/normal"
    path_val_normal = data_dir + "/jpg/val/normal"
    convert_dicom(val_normal, path_val_normal)

    val_malignant = data_dir + "/val/malignant"
    path_val_malignant = data_dir + "/jpg/val/malignant"
    convert_dicom(val_malignant, path_val_malignant)


# Clear memory
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

# Detect if GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
print(f"GPU is available: {torch.cuda.is_available()}")
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print("*" * 104)
print(f"Check settings for optimizer ({OPTIMIZER}), momentum ({MOMENTUM}) and learning rate ({LEARNING_RATE})")
print("*" * 104)
print("\n1. resnet 18\n2. resnet 152\n3. alexnet\n4. vgg11\n5. squeezenet v1\n6. densenet 121\n7. inception v3\n")
extraction_method = input("Enter the model you want to train: ")
if extraction_method == "1":
    model_name = "resnet18"
elif extraction_method == "2":
    model_name = "resnet152"
elif extraction_method == "3":
    model_name = "alexnet"
elif extraction_method == "4":
    model_name = "vgg11"
elif extraction_method == "5":
    model_name = "squeezenet"
elif extraction_method == "6":
    model_name = "densenet121"
elif extraction_method == "7":
    model_name = "inception"
else:
    exit()

# num_classes is the number of classes in the dataset
# feature_extract is a boolean that defines if fine-tuning or feature extracting. If feature_extract = False,
# the model is fine-tuned and all model parameters are updated.
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.

# Top level data_mammogram directory. Format of the directory conforms must be same as the image folder structure
print("\n1. Trained on full mammogram images, tested on full mammogram\n2. Trained on ROI mammogram images, tested on "
      "ROI mammogram\n3. Trained on ultrasound images, tested on ultrasound\n")
extraction_method = input("Enter the image training type required: ")
if extraction_method == "1":
    data_dir = "data_mammogram/mammogram/jpg"
    num_classes = 2
elif extraction_method == "2":
    # DICOM medical images
    data_dir = "data_mammogram/mgroi/jpg"
    num_classes = 2
elif extraction_method == "3":
    data_dir = "data_ultrasound"
    num_classes = 2
else:
    exit()

batch_size = int(input("Enter batch size: "))
num_epochs = int(input("Enter number of epochs: "))

# Feature extracting. When False, fine-tune the whole model,
# when True only update the reshaped layer params
print("\n1. Fine-tune whole model\n2. Feature extract / reshape final layer parameters\n")
extraction_method = input("Enter the training required: ")
if extraction_method == "1":
    feature_extract = False
elif extraction_method == "2":
    feature_extract = True
else:
    exit()

# Start Neptune run to log data_mammogram
run = neptune.init_run(
    project="tim-osmond/Retrain-ROI-MG-Test-ROI-MG",
    api_token=api,
)

# Neptune parameters to log
params = {
    "epochs": num_epochs,
    "model name": model_name,
    "data_mammogram directory": data_dir,
    "batch size": batch_size,
    "feature extract": feature_extract,
    # Neptune does not support using static names so added as run commands as work around
    "optimizer": stringify_unsupported(OPTIMIZER),
    "criterion": stringify_unsupported(LOSS_FUNCTION),
    "learning rate": stringify_unsupported(LEARNING_RATE),
    "momentum": stringify_unsupported(MOMENTUM)
}
run["parameters"] = params


# The train_model function takes a PyTorch model, a dictionary of dataloaders, a loss function, an optimizer,
# the specified number of epochs to train and validate for, and a boolean flag for when the model is an Inception model.
# The is_inception flag is used to accommodate the Inception v3 model, as that architecture uses an auxiliary output
# and the overall model loss respects both the auxiliary output and the final output. The function trains for the
# specified number of epochs and after each epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of training returns the best performing model.
# After each epoch, the training and validation accuracies are printed.

# TODO check why epochs=25
def train_model(model, dataloaders, criterion, optimizer, epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    # mode calculate the loss by summing the final output and the auxiliary output
                    # but in testing only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Neptune stats
            if phase == 'train':
                run["training loss"].append(epoch_loss)
                run["training accuracy"].append(epoch_acc)

            if phase == 'val':
                run["validation loss"].append(epoch_loss)
                run["validation accuracy"].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# This helper function sets the .requires_grad attribute of the parameters in the model to False when feature extracting.
# By default, when loading a pretrained model all the parameters have .requires_grad=True,
# which is fine if training from scratch or fine-tuning. However, if feature extracting and only want
# to compute gradients for the newly initialized layer then we want all the other parameters to not require
# gradients.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    # Reshaping of each network. This is not an automatic procedure and is unique to each model.
    # The final layer of a CNN model, which is often times an FC layer, has the same number of nodes as the number of
    # output classes in the dataset. Since all the models have been pretrained on Imagenet, they all have output layers
    # of size 1000, one node for each class. The goal here is to reshape the last layer to have the same number of
    # inputs as before, AND to have the same number of outputs as the number of classes in the dataset.
    # In the following alter the architecture of each model individually.
    #
    # When feature extracting, we only want to update the parameters of the last layer, or in other words, we only want
    # to update the parameters for the layer(s) we are reshaping. Therefore, we do not need to compute the gradients of
    # the parameters that we are not changing, so for efficiency we set the .requires_grad attribute to False. This is
    # important because by default, this attribute is set to True. Then, when we initialize the new layer and by default
    # the new parameters have .requires_grad=True so only the new layer’s parameters will be updated. When we are
    # fine-tuning we can leave all the .required_grad’s set to the default of True.
    #
    # Finally, notice that inception_v3 requires the input size to be (299,299), whereas all the other models
    # expect (224,224).

    # Resnet was introduced in the paper Deep Residual Learning for Image Recognition. There are several variants of
    # different sizes, including Resnet18, Resnet34, Resnet50, Resnet101, and Resnet152, all of which are available
    # from torchvision models. Here we use Resnet18, as our dataset is small and only has two classes. When we print
    # the model, we see that the last layer is a fully connected layer as shown below:
    # (fc): Linear(in_features=512, out_features=1000, bias=True)
    # model.fc = nn.Linear(512, num_classes)

    # Alexnet was introduced in the paper ImageNet Classification with Deep Convolutional Neural Networks and was the
    # first very successful CNN on the ImageNet dataset. When we print the model architecture, we see the model output
    # comes from the 6th layer of the classifier
    # (classifier): Sequential(
    #     ...
    #     (6): Linear(in_features=4096, out_features=1000, bias=True)
    #  )
    # model.classifier[6] = nn.Linear(4096, num_classes)

    # VGG was introduced in the paper Very Deep Convolutional Networks for Large-Scale Image Recognition. Torchvision
    # offers eight versions of VGG with various lengths and some that have batch normalizations layers.
    # Here we use VGG-11 with batch normalization. The output layer is similar to Alexnet, i.e.
    # (classifier): Sequential(
    #     ...
    #     (6): Linear(in_features=4096, out_features=1000, bias=True)
    #  )
    # model.classifier[6] = nn.Linear(4096, num_classes)

    # The Squeeznet architecture is described in the paper SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    # and <0.5MB model size and uses a different output structure than any of the other models shown here. Torchvision
    # has two versions of Squeezenet, we use version 1.0. The output comes from a 1x1 convolutional layer which is the
    # 1st layer of the classifier:
    # (classifier): Sequential(
    #     (0): Dropout(p=0.5)
    #     (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
    #     (2): ReLU(inplace)
    #     (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
    #  )
    # model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    # Densenet was introduced in the paper Densely Connected Convolutional Networks. Torchvision has four variants of
    # Densenet but here we only use Densenet-121. The output layer is a linear layer with 1024 input features:
    # (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    # model.classifier = nn.Linear(1024, num_classes)

    # Finally, Inception v3 was first described in Rethinking the Inception Architecture for Computer Vision. This
    # network is unique because it has two output layers when training. The second output is known as an auxiliary
    # output and is contained in the AuxLogits part of the network. The primary output is a linear layer at the end of
    # the network. Note, when testing we only consider the primary output. The auxiliary output and primary output of the
    # loaded model are printed as:
    # (AuxLogits): InceptionAux(
    #     ...
    #     (fc): Linear(in_features=768, out_features=1000, bias=True)
    #  )
    #  ...
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # model.AuxLogits.fc = nn.Linear(768, num_classes)
    # model.fc = nn.Linear(2048, num_classes)


# Check out the printed model architecture of the reshaped network and make sure the number of output features is the
# same as the number of classes in the dataset.

def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        # Resnet18
        # model_ft = models.resnet18(pretrained=use_pretrained) # deprecated
        model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        # Resnet152
        model_ft = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        # Alexnet
        model_ft = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg11":
        # VGG11_bn
        model_ft = models.vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        # Squeezenet
        model_ft = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet121":
        # Densenet
        model_ft = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inceptionV3":
        # Inception v3
        # Be careful, expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model just instantiated
print(model_ft)

# Initialize the transforms, image datasets, and the dataloaders.
# Notice, the models were pretrained with the hard-coded normalization values, as described here.

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("\nInitializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4,
                                   drop_last=True) for x in ['train', 'val']}

# Create an optimizer that only updates the desired parameters. After loading the pretrained model,
# but before reshaping, if feature_extract=True we manually set all the parameter’s. requires_grad attributes to False.
# Then the reinitialized layer’s parameters have .requires_grad=True by default. So now we know that all parameters
# that have .requires_grad=True should be optimized. Next, we make a list of such parameters and input this list to
# the optimizer algorithm constructor.
# To verify this, check out the printed parameters to learn. When fine-tuning, this list should be long and include
# all the model parameters. However, when feature extracting this list should be short and only include the
# weights and biases of the reshaped layers.
# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
# fine-tuning we will be updating all parameters. However, if we are
# doing feature extract method, we will only update the parameters
# that we have just initialized, i.e. the parameters with requires_grad
# is True.
params_to_update = model_ft.parameters()
print("Params to learn:\n")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer = OPTIMIZER(params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

# Set up the loss fxn
criterion = LOSS_FUNCTION()

# Run the training and validation function for the set number of epochs.
# The default learning rate is not optimal for all the models, so to achieve maximum accuracy it would be
# necessary to tune for each model separately.

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer, epochs=num_epochs,
                             is_inception=(model_name == "inception"))

# Save the current model
model_scripted = torch.jit.script(model_ft)  # Export to TorchScript
model_scripted.save('first.pt')  # Save

# **********************************************************************************************************************
# Create and view statistics for the model
confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders_dict['val']):
        inputs = inputs.to(device)
        classes = labels.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, dim=1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

scores = pd.DataFrame(index=['negative', 'positive', 'average'], columns=['precision', 'recall', 'F1-Score'])
for i, label in enumerate(["negative", "positive"]):
    p = scores.loc[label, 'precision'] = (confusion_matrix[i, i] / confusion_matrix[i].sum()).item()
    r = scores.loc[label, 'recall'] = (confusion_matrix[i, i] / confusion_matrix[:, i].sum()).item()
    scores.loc[label, 'F1-Score'] = (2 * p * r) / (p + r)
scores.loc['average'] = scores.mean().values
for i, label in enumerate(["positive"]):
    precision = scores.loc[label, 'precision'] = (confusion_matrix[i, i] / confusion_matrix[i].sum()).item()
    recall = scores.loc[label, 'recall'] = (confusion_matrix[i, i] / confusion_matrix[:, i].sum()).item()
    f1 = scores.loc[label, 'F1-Score'] = (2 * p * r) / (p + r)
print(f"\nOverall...\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {f1:.2f}\n")
print(scores)
print()

label2class = {0: 'negative', 1: 'positive'}

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.8)

class_names = list(label2class.values())
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
# plt.show()
# **********************************************************************************************************************

# Export to Neptune
run["results/confusion_matrix"] = stringify_unsupported(confusion_matrix)
run["val/conf_matrix"].upload("confusion_matrix.png")  # Upload confusion matrix image to Neptune
run["results/scores"] = stringify_unsupported(scores)
run["results/precision"] = stringify_unsupported(precision)
run["results/recall"] = stringify_unsupported(recall)
run["results/F1"] = stringify_unsupported(f1)

# Finish export to Neptune
run.stop()
