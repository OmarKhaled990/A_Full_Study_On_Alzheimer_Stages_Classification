# AFull Study On Alzheimer Stages Classification

This project involves the analysis and classification of Alzheimer's disease using brain MRI images. The project utilizes various machine learning models and techniques to achieve accurate classification results. Below is an overview of the project structure and the different approaches used.

# Project Dependencies

The following libraries and modules are required to run the code successfully:

pandas for data manipulation

numpy for numerical operations

seaborn and matplotlib.pyplot for data visualization

cv2 and PIL for image processing

tensorflow and keras for deep learning models

torch and torchvision for neural network models

imblearn for oversampling

splitfolders for dataset splitting

scikit-learn for logistic regression, SVM, and evaluation metrics

albumentations for image augmentation

tqdm for progress tracking

# Data Preparation

The Alzheimer's disease dataset is located at ../input/alzheimer-mri-dataset/Dataset. The dataset is split into train, test, and validation sets using the splitfolders package. The images are resized to a height and width of 128 pixels.

# Exploratory Data Analysis

The class distribution of the dataset is visualized using a bar plot. Additionally, sample images from each class are displayed using the sample_bringer function.

# Approach 1: Convolutional Neural Network (CNN) Model
A CNN model is implemented using the Keras library. The model consists of several convolutional layers, pooling layers, dropout layers, and fully connected layers. The model is compiled with the Adam optimizer and trained on the train_ds dataset. The model's performance is evaluated on the validation set (val_ds) for 100 epochs.

# Approach 2: EfficientNet Model
An EfficientNet model is used for classification. The base model is pre-trained on the ImageNet dataset and fine-tuned on the Alzheimer's dataset. The model is trained for 100 epochs on the train_ds dataset and evaluated on the validation set (val_ds).

# Approach 3: Logistic Regression
Logistic regression is applied to classify the images. The images are flattened and preprocessed using a pipeline that includes standardization and dimensionality reduction through PCA. The model is trained using the training set and evaluated on the test set.

# Approach 4: Support Vector Machine (SVM)
An SVM model is trained on the preprocessed images. The hyperparameters for the SVM model are optimized using grid search with cross-validation. The model's performance is evaluated on the test set.

# Results and Evaluation

Approach 1: The CNN model achieves an accuracy score and loss value after 100 epochs. The training and validation progress is stored in the hist variable.

Approach 2: The EfficientNet model's training progress and performance on the validation set are stored in the history variable.

Approach 3: The logistic regression model's evaluation results, including the classification report and confusion matrix, are displayed.

Approach 4: The SVM model's best parameters and accuracy score on the test set are shown, followed by the confusion matrix.
