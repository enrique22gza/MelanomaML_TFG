import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, f1_score, precision_score, roc_curve, auc, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Parameters for images
img_height, img_width = 224, 224

# Evaluation image directories
eval_melanoma_dir = ''
eval_no_melanoma_dir = ''

# Create the dataframe to use with flow_from_dataframe
eval_filepaths = []
eval_labels = []

valid_extensions = ['.jpg', '.jpeg', '.png']

# Filter and add only valid files
for filename in os.listdir(eval_melanoma_dir):
    if any(filename.lower().endswith(ext) for ext in valid_extensions):
        eval_filepaths.append(os.path.join(eval_melanoma_dir, filename))
        eval_labels.append('melanoma')

for filename in os.listdir(eval_no_melanoma_dir):
    if any(filename.lower().endswith(ext) for ext in valid_extensions):
        eval_filepaths.append(os.path.join(eval_no_melanoma_dir, filename))
        eval_labels.append('no_melanoma')

# Create the dataframe
eval_df = pd.DataFrame({
    'filename': eval_filepaths,
    'class': eval_labels
})

# Display the first rows of the dataframe and a summary of the classes
print(eval_df.head())
print("\nClass summary:")
print(eval_df['class'].value_counts())

# Load the pre-trained model
model = load_model('/Users/enriquegonzalezardura/Documents/modelos/model_fold_10.keras')

# Prepare the evaluation data generator
eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = eval_datagen.flow_from_dataframe(
    eval_df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    class_mode='binary',  # Or 'categorical' if you have more than two classes
    shuffle=False
)

# Reverse the class indices manually
class_mapping = {'melanoma': 1, 'no_melanoma': 0}

# Print class indices to verify
print("Original class indices:", eval_generator.class_indices)
eval_generator.class_indices = class_mapping
print("Updated class indices:", eval_generator.class_indices)

# Make predictions
predictions = model.predict(eval_generator)
predicted_classes = (predictions > 0.5).astype(int)  # Threshold of 0.5 to convert probabilities to classes

# True labels
true_classes = eval_generator.classes

# Remap true and predicted classes
true_classes = np.array([class_mapping['melanoma'] if cls == 0 else class_mapping['no_melanoma'] for cls in true_classes])
predicted_classes = np.array([class_mapping['melanoma'] if cls == 0 else class_mapping['no_melanoma'] for cls in predicted_classes])

# Create confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Print confusion matrix values
tn, fp, fn, tp = cm.ravel()
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
print(f'True Positives (TP): {tp}')

# Calculate metrics
accuracy = accuracy_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)
specificity = tn / (tn + fp)

# Print metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Precision: {precision:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'Specificity: {specificity:.2f}')

# Visualize the confusion matrix with labels

# Visualize the confusion matrix


plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Melanoma', 'Melanoma'], rotation=45)
plt.yticks(tick_marks, ['No Melanoma', 'Melanoma'])

fmt = 'd'
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(true_classes, predicted_classes)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(true_classes, predicted_classes)
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
pr_display.plot()
plt.title('Precision-Recall curve')
plt.show()
