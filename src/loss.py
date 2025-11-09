import torch
import torch.nn as nn
from focalloss import FocalLoss

class CascadeLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3, reduction='mean', lambda_mt=1.0):
        super().__init__()
        self.plaque_criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.multi_task_criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)  # Added
        self.task_weights = {
            'form': 1.0,
            'surface': 1.0,
            'echo': 1.0,
            'calcification': 1.0,
            'stenosis': 1.0
        }

        # Added: Vulnerability task
        self.vul_criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        self.lambda_mt = lambda_mt
        self.task_weights = {t: 1.0 for t in ['form','surface','echo','calcification','stenosis']}

    def forward(self, outputs, labels):
        # 1) Plaque loss
        plaque_loss = self.plaque_criterion(outputs['plaque'], labels['plaque'])

        # 2) Sub-task losses - Take logits_dict[t]
        multi_loss = 0.0
        for i, task in enumerate(self.task_weights):
            true = labels['multi_task'][:, i]
            logits = outputs['task_logits'][task]
            valid = (true >= 0) & (true < logits.size(1))
            if valid.any():
                multi_loss += self.task_weights[task] * \
                              self.multi_task_criterion(
                                  logits[valid], true[valid]
                              )

        # 3) Vulnerability loss - Take vul_logit
        plaque_mask = (labels['plaque'] > 0.5)
        if plaque_mask.any():
            vul_loss = self.vul_criterion(
                outputs['vul_logit'][plaque_mask],
                labels['vulnerability'][plaque_mask]
            )
        else:
            vul_loss = torch.tensor(0.0, device=outputs['vul_logit'].device)

        # 4) Total loss
        total = vul_loss + self.lambda_mt * (plaque_loss + 0.5 * multi_loss)
        return total
