import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.models import load_model
from FDF_env import norm_cols
from sklearn.metrics import confusion_matrix
import csv

# Function to calculate TPR and FPR for each group
def compute_tpr_fpr(group, y_true, y_pred):
    # Confusion Matrix: [TN, FP], [FN, TP]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print(tn,fp,fn,tp)
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # Avoid division by zero
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # Avoid division by zero

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Avoid division by zero
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    return tpr, fpr, tnr, fnr

DATA_PATH = "adults_income_preprocessed.csv"
# ------------------------
# File paths
fairness_file = "adults - fairness_metrics_agg.csv"
performance_file = "adults - performance_metrics_agg.csv"

# Write fairness metrics
with open(fairness_file, mode='w', newline='') as file1, open(performance_file, mode='w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    header = ["Model", "Group", "EO_gap", "Diff. TPR", "Diff. TNR", "Diff. FPR", "Diff. FNR", "SP", "DI"]
    writer1.writerow(header)
    header = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "TN", "FP", "FN", "TP"]
    writer2.writerow(header)

    for ep in range (50):
        # -------- CONFIG --------
        MODEL_PATH = f"FL Models Agg/ep{ep}/Server.keras"
        # Load dataset
        df = pd.read_csv(DATA_PATH)
        X, y = df.iloc[:,:-2], df.iloc[:,-2:]
        X = np.array(X).astype("float32")  # or float64 if needed

        # Load the trained model
        model = load_model(MODEL_PATH,compile=False)

        groups = [1,2,3,4,5,6] #for testing
        metrics = {}

        Xx = pd.DataFrame(X, columns=norm_cols[:-2])

        predicted = model(X)
        # y = np.argmax(y, axis=1)
        # predicted = np.argmax(y_pred, axis=1).round()

        attribute_val_dict = {
        "1" : ["sex_Female", True],
        "2" : ["age_>45", True],
        "3" : ["race_Black", True],
        "4" : ["race_White", True],
        "5" : ["race_Asian-Pac-Islander", True],
        "6" : ["race_Amer-Indian-Eskimo", True],
        }

        for group in groups:
            key, val = attribute_val_dict[f"{group}"]
            
            if not isinstance(predicted, np.ndarray):
                predicted = predicted.numpy()
                
            if not isinstance(y, np.ndarray):
                y = y.to_numpy()

            group1indices = np.where(Xx[key].astype(int) == val)[0]
            group2indices = np.where(Xx[key].astype(int) != val)[0]

            group1_true = y[group1indices]
            group1_pred = predicted[group1indices]

            group2_true = y[group2indices]
            group2_pred = predicted[group2indices]
            
            y_true_labels_1 = np.argmax(group1_true, axis=1)
            y_pred_labels_1 = np.argmax(group1_pred, axis=1).round()

            y_true_labels_2 = np.argmax(group2_true, axis=1)
            y_pred_labels_2 = np.argmax(group2_pred, axis=1).round()

            tpr1, fpr1, tnr1, fnr1 = compute_tpr_fpr(group1indices, y_true_labels_1, y_pred_labels_1)
            tpr2, fpr2, tnr2, fnr2 = compute_tpr_fpr(group2indices, y_true_labels_2, y_pred_labels_2)

            P_Y1_A1 = np.mean(y_pred_labels_1)  # P(Y=1 | A=1)
            P_Y1_A0 = np.mean(y_pred_labels_2)  # P(Y=1 | A=0)

            f1g1 = f1_score(y_true_labels_1, y_pred_labels_1)
            f1g2 = f1_score(y_true_labels_2, y_pred_labels_2)

            # TPR = tpr1 / (tpr1 + fnr1) if (tpr1 + fnr1) > 0 else 0
            # FPR = fpr1 / (fpr1 + tnr1) if (fpr1 + tnr1) > 0 else 0
            eo_gap = max(abs(tpr1 - tpr2), abs(fpr1 - fpr2))

            key,_ = attribute_val_dict[f"{group}"]
            metrics[key] = {'Diff. TPR': abs(tpr2-tpr1), 'Diff. FPR': abs(fpr2-fpr1),
                            'Diff. TNR': abs(tnr2-tnr1), 'Diff. FNR': abs(fnr2-fnr1),
                            "SP": (P_Y1_A1-P_Y1_A0), "DI": f1g2-f1g1, 'EO-gap': eo_gap}

        # Print TPR and FPR for each group
        print("Metrics by group:")
        for group_name, metric in metrics.items():
            print(f"{group_name}: EO-gap = {metric['EO-gap']}, TPR = {metric['Diff. TPR']}, TNR = {metric['Diff. TNR']}, FPR = {metric['Diff. FPR']}, FNR = {metric['Diff. FNR']}, SP = {metric['SP']}, DI = {metric['DI']}")
            writer1.writerow([ep, group_name, metric['EO-gap'], metric["Diff. TPR"], metric["Diff. TNR"], metric["Diff. FPR"], metric["Diff. FNR"], metric["SP"], metric["DI"]])
        
        predicted = np.argmax(predicted, axis=1).round()
        y = np.argmax(y, axis=1).round()
        
        # --- Evaluation Metrics ---
        acc = accuracy_score(y, predicted)
        prec = precision_score(y, predicted)
        rec = recall_score(y, predicted)
        f1 = f1_score(y, predicted)
        cm = confusion_matrix(y, predicted)

        # --- Print Results ---
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        writer2.writerow([ep,acc, prec, rec, f1,cm[0][0], cm[0][1], cm[1][0], cm[1][1]])