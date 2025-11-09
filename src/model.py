import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models.layers import DropPath 

class CTANLayer(nn.Module):
    """
    Cross-Task Attention (T tasks × H heads)
    Input: dict{task: [B,C]}
    Output: same shape, fused with cross-task information
    """
    def __init__(self, tasks, dim, heads=4, attn_drop=0.1):
        super().__init__()
        self.tasks  = tasks
        self.heads  = heads
        self.scale  = (dim // heads) ** -0.5
        # One set of QKV & output projection per task
        self.qkv   = nn.ModuleDict({t: nn.Linear(dim, dim * 3) for t in tasks})
        self.proj  = nn.ModuleDict({t: nn.Linear(dim, dim)     for t in tasks})
        # For stable training
        self.dropout = nn.Dropout(attn_drop)
        self.norm    = nn.LayerNorm(dim)

    def forward(self, feats):
        """
        feats: dict{task: [B,C]}
        """
        B = next(iter(feats.values())).size(0)
        orig = feats                                # Backup for residual

        q_dict, all_k, all_v = {}, [], []
        for t in self.tasks:
            q, k, v = self.qkv[t](orig[t]).chunk(3, -1)
            q_dict[t] = q                           # [B,C]
            all_k.append(k)
            all_v.append(v)

        K = torch.stack(all_k, dim=1)               # [B,T,C]
        V = torch.stack(all_v, dim=1)               # [B,T,C]

        # Multi-head reshape: (B,T,H,C//H)
        K = K.view(B, len(self.tasks), self.heads, -1)
        V = V.view(B, len(self.tasks), self.heads, -1)

        out = {}
        for t in self.tasks:
            Q = q_dict[t].view(B, self.heads, -1)   # [B,H,C//H]
            attn = (Q.unsqueeze(1) * K).sum(-1) * self.scale    # [B,T,H]
            attn = attn.softmax(1)
            ctx  = (attn.unsqueeze(-1) * V).sum(1)              # [B,H,C//H]
            ctx  = ctx.flatten(1)                               # [B,C]
            ctx  = self.dropout(self.proj[t](ctx))
            out[t] = self.norm(ctx + orig[t])                   # Correct residual
        return out
    
class DictDropPath(nn.Module):
    """
    Apply DropPath to dict{task: tensor} key by key.
    """
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.dp = DropPath(drop_prob)

    def forward(self, feats: dict):
        return {k: self.dp(v) for k, v in feats.items()}

class DictLayerNorm(nn.Module):
    """
    Apply LayerNorm to each tensor in dict{task: tensor} separately.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, feats: dict):
        return {k: self.norm(v) for k, v in feats.items()}

class CascadeMultiTaskModel(nn.Module):
    def __init__(self, backbone_name: str = 'convnext_base'):
        super().__init__()
        self.tasks = ['form', 'surface', 'echo', 'calcification', 'stenosis']

        # ---------- 1) Shared feature extractor ----------
        if backbone_name.startswith('convnext'):
            weights_map = {
                'convnext_tiny':  models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                'convnext_small': models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
                'convnext_base':  models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            }
            backbone = getattr(models, backbone_name)(weights=weights_map[backbone_name])
            self.encoder = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1)
            )
            num_feat = backbone.classifier[-1].in_features
        else:                                     # resnet50
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(1))
            num_feat = 2048

        # ---------- 2) Lightweight projection per task ----------
        self.task_embed = nn.ModuleDict({
            t: nn.Linear(num_feat, num_feat) for t in self.tasks
        })

        # ---------- 3) Two layers of CTAN + DropPath + LayerNorm ----------
        self.ctan = nn.Sequential(
            CTANLayer(self.tasks, dim=num_feat, heads=4),
            DictDropPath(drop_prob=0.1),            # Key addition
            CTANLayer(self.tasks, dim=num_feat, heads=4),
            DictLayerNorm(num_feat)                 # Unified normalization
        )

        # ---------- 4) Task output heads ----------
        self.plaque_head = nn.Sequential(
            nn.Linear(num_feat, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1), nn.Sigmoid()
        )
        self.multi_task_heads = nn.ModuleDict({
            'form':          nn.Linear(num_feat, 2),
            'surface':       nn.Linear(num_feat, 2),
            'echo':          nn.Linear(num_feat, 3),
            'calcification': nn.Linear(num_feat, 3),
            'stenosis':      nn.Linear(num_feat, 2)
        })

        # Added: SE-style gating α branch
        self.alpha_fc = nn.Sequential(
            nn.Linear(num_feat, num_feat // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_feat // 16, num_feat, bias=False),
            nn.Sigmoid()
        )

        # Added: MLP for integrating β
        total_beta = sum([2,2,3,3,2])
        self.vul_mlp = nn.Sequential(
            nn.Linear(num_feat + total_beta, num_feat // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_feat // 2, 1)
        )

    # ---------------- forward ----------------
    def forward(self, x, gt_plaque_mask: torch.Tensor = None):
        # 1) shared feature
        feat_shared = self.encoder(x)                           # [B, C]
        # 2) Main task: plaque prediction
        plaque_prob = self.plaque_head(feat_shared).squeeze(-1) # [B]

        # ---------- Select gating mask ----------
        if gt_plaque_mask is not None:                # Passed during training
            # Ensure dtype=bool, shape=[B], and on the same device
            plaque_mask = gt_plaque_mask.to(plaque_prob.device).bool()
        else:                                          # Use prediction during inference
            plaque_mask = plaque_prob > 0.5            # [B] boolean

        out = {'plaque': plaque_prob}

        # 3) Multi-task embedding → CTAN → multi-task heads
        task_in   = {t: self.task_embed[t](feat_shared) for t in self.tasks}
        task_out  = self.ctan(task_in)                          # dict of [B, C]
        beta_list = []
        for t, head in self.multi_task_heads.items():
            logit      = head(task_out[t])                      # [B, num_classes_t]
            out[t]     = logit
            beta_list.append(F.softmax(logit, dim=1))           # [B, num_classes_t]
        # Concatenate β vector
        beta = torch.cat(beta_list, dim=1)                      # [B, total_beta]

        # 4) Initialize vulnerability output to all 0
        vul_logit = torch.full_like(plaque_prob, -1e4, requires_grad=True)        

        # 6) Perform SE-gate + β-fusion only for positive samples
        if plaque_mask.any():
            # SE-gate
            alpha  = self.alpha_fc(feat_shared)                 # [B, C]
            gated  = alpha * feat_shared                        # [B, C]
            # Concatenate gated + β
            fusion = torch.cat([gated, beta], dim=1)            # [B, C + total_beta]
            # Output only for positive indices
            logits = self.vul_mlp(fusion).squeeze(-1)           # [B]
            vul_logit = torch.where(plaque_mask, logits, vul_logit)
            
        # 7) Calculate probability and organize output dict
        vul_prob = torch.sigmoid(vul_logit)
        out['vulnerability_logit'] = vul_logit
        out['vulnerability']      = vul_prob

        # 8) (Optional) Merge logits for subsequent loss/metric use
        out['task_logits'] = {t: out[t] for t in self.tasks}
        out['vul_logit']   = out.pop('vulnerability_logit')
        out['vul_prob']    = out.pop('vulnerability')

        return out
