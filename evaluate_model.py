import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

print("Loading model and test data...")
model = load_model('signal_classifier.h5')
with np.load("prepared_data.npz", allow_pickle=True) as data:
    X_test = data['X_test']
    Y_test = data['Y_test']
    modulation_types = data['modulation_types']

X_test = X_test.reshape(X_test.shape[0], 2, 128, 1)

print("Making predictions on the test set...")
pred_probabilities = model.predict(X_test)
pred_classes = np.argmax(pred_probabilities, axis=1)
true_classes = np.argmax(Y_test, axis=1)

print("\n--- Classification Report ---")

report = classification_report(true_classes, pred_classes, target_names=modulation_types)
print(report)
print("-----------------------------\n")

print("Generating and saving the confusion matrix...")
conf_matrix = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(10, 8))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis',
            xticklabels=modulation_types,
            yticklabels=modulation_types)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')

print("Confusion matrix saved as 'confusion_matrix.png'.")
print("Project finished!")
