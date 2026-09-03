import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("peptides.csv")

# Extract columns
y_true = df["True_Label"]
y_score = df["AMP_Probability"]

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
avg_precision = average_precision_score(y_true, y_score)

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2,
         label=f"PR curve (AP = {avg_precision:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="lower left")

# Save figure
plt.savefig("pr_curve.png", dpi=300)
plt.close()

print("PR curve saved as pr_curve.png")
