import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import torch.nn.functional as F
import csv
import torch.multiprocessing  # Used for set_start_method
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CascadeMultiTaskModel 
from data import CascadeImageDataset   
from utils import set_seed, FocalLoss, CascadeLoss 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set seed (imported from utils.py)
set_seed(42)

# Test/inference function (restructured from original code, for test set only, no training part)
def test_cascade_model(model, test_loader, criterion, device='cpu', output_dir='fig', model_path=None):
    # Load pretrained model weights (if path provided)
    if model_path:
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()  # Set to evaluation mode

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Add CSV writing, including vulnerability metric columns
    csv_file = os.path.join(output_dir, "metrics_log_test.csv")  # Separate file for test
    is_new = not os.path.exists(csv_file)
    csv_fields = [
        "epoch", "phase",
        "plaque_acc", "plaque_auc", "plaque_precision", "plaque_recall", "plaque_f1",
        "sensitivity", "specificity"
    ]
    for t in ['form', 'surface', 'echo', 'calcification', 'stenosis']:
        csv_fields += [f"{t}_auc", f"{t}_precision", f"{t}_recall", f"{t}_f1"]
    # vulnerability
    csv_fields += ["vul_auc", "vul_precision", "vul_recall", "vul_f1"]

    if is_new:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    # Fixed epoch=0, run once
    epoch = 0
    phase = 'test'
    print(f'Test Phase (Epoch {epoch})')

    running_loss = 0.0
    plaque_correct = 0

    # Collect metrics for each task and vulnerability
    all_labels_plaque = []
    all_preds_plaque = []
    all_labels_multi = {t: [] for t in ['form', 'surface', 'echo', 'calcification', 'stenosis']}
    all_preds_multi = {t: [] for t in all_labels_multi}
    all_probs_multi = {t: [] for t in all_labels_multi}
    all_labels_vul = []
    all_preds_vul = []

    desc   = f"{phase} Epoch {epoch}"
    for inputs, labels in tqdm(test_loader, desc=desc):
        inputs = inputs.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        with torch.no_grad():  # No gradient computation
            gt_mask = (labels['plaque'] > 0.5).squeeze()      # shape [B]
            outputs = model(inputs, gt_plaque_mask=gt_mask)

            loss = criterion(outputs, labels)  # Compute loss (for reporting)

        # Plaque accuracy
        plaque_pred = (outputs['plaque'] > 0.5).float()
        plaque_correct += (plaque_pred == labels['plaque']).sum().item()

        running_loss += loss.item() * inputs.size(0)

        # Collect plaque metrics
        all_labels_plaque.extend(labels['plaque'].cpu().numpy())
        all_preds_plaque.extend(outputs['plaque'].detach().cpu().numpy())

        # Collect multi-task metrics (only for plaque-positive samples)
        valid_mask = (labels['plaque'] > 0.5).squeeze().cpu().numpy().astype(bool)
        for idx, task in enumerate(all_labels_multi.keys()):
            y = labels['multi_task'][:, idx].cpu().numpy()
            logits = outputs['task_logits'][task]
            probs  = F.softmax(logits, dim=1).detach().cpu().numpy()
            preds  = torch.argmax(logits, dim=1).detach().cpu().numpy()

            y_valid    = y[valid_mask]
            probs_valid= probs[valid_mask]
            preds_valid= preds[valid_mask]

            all_labels_multi[task].extend(y_valid.tolist())
            all_preds_multi[task].extend(preds_valid.tolist())
            all_probs_multi[task].extend(probs_valid.tolist())

        # Collect vulnerability metrics (only for plaque==1)
        plaque_mask = (labels['plaque'] > 0.5).squeeze()
        if plaque_mask.any():
            all_labels_vul.extend(labels['vulnerability'][plaque_mask]
                                .cpu().numpy().tolist())
            vul_scores = outputs['vul_prob'][plaque_mask] \
                        .detach().cpu().numpy()                    
            all_preds_vul.extend(vul_scores.tolist())

    # ====== Calculate metrics ======
    dataset_size = len(test_loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc  = plaque_correct / dataset_size

    lbls = np.array(all_labels_plaque)
    preds = np.array(all_preds_plaque)
    print("labels NaN:", np.isnan(lbls).sum(), "  preds NaN:", np.isnan(preds).sum())

    # --- Plaque task ---
    epoch_auc       = roc_auc_score(all_labels_plaque, all_preds_plaque)
    epoch_precision = precision_score(all_labels_plaque, (np.array(all_preds_plaque)>0.5).astype(int))
    epoch_recall    = recall_score(all_labels_plaque,  (np.array(all_preds_plaque)>0.5).astype(int))
    epoch_f1        = f1_score(all_labels_plaque,    (np.array(all_preds_plaque)>0.5).astype(int))
    tn, fp, fn, tp  = confusion_matrix(all_labels_plaque, (np.array(all_preds_plaque)>0.5).astype(int)).ravel()
    sensitivity    = tp/(tp+fn) if tp+fn>0 else 0
    specificity    = tn/(tn+fp) if tn+fp>0 else 0

    # --- Multi-task ---
    multi_task_metrics = {}
    task_classes = {'form':2,'surface':2,'echo':3,'calcification':3,'stenosis':2}
    for task in all_labels_multi:
        y_true_full = np.array(all_labels_multi[task])
        y_pred_full = np.array(all_preds_multi[task])
        y_prob_full = np.array(all_probs_multi[task])
        mask = y_true_full >= 0
        y_true = y_true_full[mask]
        y_pred = y_pred_full[mask]
        y_prob = y_prob_full[mask]
        if y_true.size == 0:
            metrics = dict(AUC=np.nan, Precision=np.nan, Recall=np.nan, F1=np.nan)
        else:
            # AUC
            if len(np.unique(y_true))<2:
                auc = np.nan
            elif task_classes[task]==2:
                auc = roc_auc_score(y_true, y_prob[:,1])
            else:
                auc = roc_auc_score(y_true, y_prob,
                                    multi_class='ovr', average='weighted',
                                    labels=np.arange(task_classes[task]))
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            # calculate_and_plot_confusion_matrix(all_labels_multi[task], all_preds_multi[task], task, task_classes[task], epoch, output_dir=output_dir)  # Comment out to run
            metrics = dict(AUC=auc, Precision=prec, Recall=rec, F1=f1)
        multi_task_metrics[task] = metrics

    # --- Vulnerability task ---
    # AUC + binary metrics at threshold 0.5
    vul_auc       = roc_auc_score(all_labels_vul, all_preds_vul)
    vul_pred_bin  = (np.array(all_preds_vul) > 0.5).astype(int)
    vul_precision = precision_score(all_labels_vul, vul_pred_bin, zero_division=0)
    vul_recall    = recall_score(all_labels_vul, vul_pred_bin, zero_division=0)
    vul_f1        = f1_score(all_labels_vul, vul_pred_bin, zero_division=0)

    # ====== Print ======
    print(f'{phase} Loss: {epoch_loss:.4f} | Plaque Acc: {epoch_acc:.4f} | '
          f'Plaque AUC: {epoch_auc:.4f} | P Prec: {epoch_precision:.4f} | '
          f'P Rec: {epoch_recall:.4f} | P F1: {epoch_f1:.4f} | '
          f'Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | '
          f'Vul AUC: {vul_auc:.4f} | V Prec: {vul_precision:.4f} | '
          f'V Rec: {vul_recall:.4f} | V F1: {vul_f1:.4f}')

    for task, m in multi_task_metrics.items():
        print(f'  {task} AUC: {m["AUC"]:.4f} | Prec: {m["Precision"]:.4f} | '
              f'Rec: {m["Recall"]:.4f} | F1: {m["F1"]:.4f}')

    # ====== Write to CSV ======
    row = {
        "epoch": epoch, "phase": phase,
        "plaque_acc": epoch_acc, "plaque_auc": epoch_auc,
        "plaque_precision": epoch_precision, "plaque_recall": epoch_recall,
        "plaque_f1": epoch_f1, "sensitivity": sensitivity,
        "specificity": specificity,
        "vul_auc": vul_auc, "vul_precision": vul_precision,
        "vul_recall": vul_recall, "vul_f1": vul_f1
    }
    for t, metrics in multi_task_metrics.items():
        row.update({
            f"{t}_auc": metrics["AUC"],
            f"{t}_precision": metrics["Precision"],
            f"{t}_recall": metrics["Recall"],
            f"{t}_f1": metrics["F1"],
        })
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writerow(row)

    print("Test completed. Metrics and figures saved.")

# Main function (for running the test)
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    excel_file = ''  # Adjust to your path
    model_path = "model.pth"  # Adjust to your path

    # Assume test set in sheet_name=1 (adjust if different)
    test_dataset = CascadeImageDataset(excel_file, sheet_name=1, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Model (imported from model.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name = 'convnext_base'
    model = CascadeMultiTaskModel(backbone_name)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Loss function (imported from utils.py)
    criterion = CascadeLoss(lambda_mt=1.0)

    # Print sample data (optional, similar to original code)
    for imgs, labels in test_loader:
        print("plaque label sample:", labels['plaque'][:10])
        print("multi_task sample:", labels['multi_task'][:2])
        print("vul label sample:", labels['vulnerability'][:10])
        print("imgs min/max:", imgs.min().item(), imgs.max().item())
        break

    # Test
    test_cascade_model(
        model, test_loader,
        criterion,
        device=device,
        model_path=model_path
    )

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
