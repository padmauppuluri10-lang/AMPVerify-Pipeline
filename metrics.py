import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv("peptides.csv")

# Extract columns
y_true = df["True_Label"]
y_pred = df["Pred_Label"]
y_score = df["AMP_Probability"]

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
auroc = roc_auc_score(y_true, y_score)
auprc = average_precision_score(y_true, y_score)

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("MCC:", mcc)
print("AUROC:", auroc)
print("AUPRC:", auprc)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-AMP", "AMP"],
            yticklabels=["Non-AMP", "AMP"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("Confusion matrix saved as confusion_matrix.png")
