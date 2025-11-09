import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from plot import plot_confusion_matrix, calculate_and_plot_confusion_matrix, plot_plaque_confusion_matrix
import csv
import itertools
import matplotlib
matplotlib.use('Agg') 

# Import other modules
from data import CascadeImageDataset
from model import CascadeMultiTaskModel
from loss import CascadeLoss
from utils import set_seed

def train_cascade_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        num_epochs=100, device='cpu'):
    # 1) Freeze before training
    warmup_epochs = 3
    ctan_params  = (model.module.ctan if isinstance(model, nn.DataParallel) else model.ctan).parameters()
    embed_params = (model.module.task_embed if isinstance(model, nn.DataParallel) else model.task_embed).parameters()
    for p in itertools.chain(ctan_params, embed_params):
        p.requires_grad = False

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    output_dir = 'fig'
    os.makedirs(output_dir, exist_ok=True)

    # Add CSV writing, including vulnerability metric columns
    csv_file = os.path.join(output_dir, "metrics_log.csv")
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

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
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

            loader = train_loader if phase == 'train' else val_loader
            desc   = f"{phase} Epoch {epoch + 1}"
            for inputs, labels in tqdm(loader, desc=desc):
                inputs = inputs.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs = model(inputs)
                    gt_mask = (labels['plaque'] > 0.5).squeeze()      # shape [B]
                    outputs = model(inputs, gt_plaque_mask=gt_mask)

                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

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
                    # logits = outputs[task]
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
                    # vul_scores = torch.sigmoid(outputs['vulnerability'])[plaque_mask] \
                    #             .detach().cpu().numpy()
                    vul_scores = outputs['vul_prob'][plaque_mask] \
                                .detach().cpu().numpy()                    
                    all_preds_vul.extend(vul_scores.tolist())

            # ====== Calculate metrics ======
            dataset_size = len(train_loader.dataset) if phase=='train' else len(val_loader.dataset)
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
            plot_plaque_confusion_matrix(
                all_labels_plaque,
                (np.array(all_preds_plaque)>0.5).astype(int),
                epoch, output_dir=output_dir
            )

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
                    calculate_and_plot_confusion_matrix(
                        all_labels_multi[task], all_preds_multi[task],
                        task, task_classes[task], epoch, output_dir=output_dir
                    )
                    metrics = dict(AUC=auc, Precision=prec, Recall=rec, F1=f1)
                multi_task_metrics[task] = metrics

            # --- Vulnerability task ---
            # AUC + binary metrics at threshold 0.5
            vul_auc       = roc_auc_score(all_labels_vul, all_preds_vul)
            vul_pred_bin  = (np.array(all_preds_vul) > 0.5).astype(int)
            vul_precision = precision_score(all_labels_vul, vul_pred_bin, zero_division=0)
            vul_recall    = recall_score(all_labels_vul, vul_pred_bin, zero_division=0)
            vul_f1        = f1_score(all_labels_vul, vul_pred_bin, zero_division=0)
            plot_confusion_matrix(
                all_labels_vul,
                vul_pred_bin,
                task_name='vulnerability',
                num_classes=2,
                epoch=epoch,
                output_dir=output_dir
            )


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
                "epoch": epoch+1, "phase": phase,
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
                # csv.writer(f)  # ensure writer exists
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(row)

            # Save the best model (based on plaque accuracy or can be changed to vulnerability AUC)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"{output_dir}/best_model.pth")

        # 2) Unfreeze after warmup_epochs
        if epoch + 1 == warmup_epochs:
            for p in itertools.chain(ctan_params, embed_params):
                p.requires_grad = True
            print(">>> CTAN & task-embed unfrozen <<<")

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

    # Load the best weights and return
    model.load_state_dict(best_model_wts)
    return model

def main():
    set_seed(42)  # Imported from utils

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    excel_file = ''

    # 1) Dataset also outputs vulnerability
    train_dataset = CascadeImageDataset(excel_file, sheet_name=0, transform=transform)
    val_dataset   = CascadeImageDataset(excel_file, sheet_name=1, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0)

    # 2) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name = 'convnext_base'
    model = CascadeMultiTaskModel(backbone_name)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 3) Optimizer: Add vulnerability_head
    encoder = model.module.encoder   if isinstance(model, nn.DataParallel) else model.encoder
    task_embed = model.module.task_embed   if isinstance(model, nn.DataParallel) else model.task_embed
    ctan       = model.module.ctan         if isinstance(model, nn.DataParallel) else model.ctan
    plaque_hd  = model.module.plaque_head  if isinstance(model, nn.DataParallel) else model.plaque_head
    multi_heads= model.module.multi_task_heads if isinstance(model, nn.DataParallel) else model.multi_task_heads
    # vul_head   = model.module.vulnerability_head if isinstance(model, nn.DataParallel) else model.vulnerability_head
    alpha_fc   = model.module.alpha_fc      if isinstance(model, nn.DataParallel) else model.alpha_fc
    vul_mlp    = model.module.vul_mlp       if isinstance(model, nn.DataParallel) else model.vul_mlp

    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': 1e-4},
        {'params': task_embed.parameters(), 'lr': 5e-4},
        {'params': ctan.parameters(),       'lr': 5e-4},
        {'params': plaque_hd.parameters(),  'lr': 1e-3},
        {'params': multi_heads.parameters(),'lr': 1e-3},
        # {'params': vul_head.parameters(),   'lr': 1e-3},
        {'params': alpha_fc.parameters(), 'lr': 1e-3},
        {'params': vul_mlp.parameters(), 'lr': 1e-3},
    ])


    # 4) Loss: Optionally pass λ
    criterion = CascadeLoss(lambda_mt=1.0)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Place before calling train_cascade_model in main()
    for imgs, labels in train_loader:
        print("plaque label sample:", labels['plaque'][:10])
        print("multi_task sample:", labels['multi_task'][:2])
        print("vul label sample:", labels['vulnerability'][:10])
        print("imgs min/max:", imgs.min().item(), imgs.max().item())
        break

    # 5) Training: Add mode parameter
    trained_model = train_cascade_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        num_epochs=50,
        # mode='T-Tune',      # Or 'F-Freeze' / 'FT-Full'
        device=device
    )

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
