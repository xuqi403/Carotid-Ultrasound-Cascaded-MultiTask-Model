from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import label_binarize

# ======== 任务类别描述 & 正类定义 ========
CLASS_DESCS = {
    "plaque":        {0: "None", 1: "Plaque"},
    "form":          {0: "Regular", 1: "Irregular"},
    "surface":       {0: "Smooth", 1: "Rough/Ulcer"},
    "echo":          {0: "Low", 1: "Hetero", 2: "High"},
    "calcification": {0: "None", 1: "Diffuse", 2: "Punctate"},
    "stenosis":      {0: "None", 1: "Present"},
    "vulnerability": {0: "Stable", 1: "Vulnerable"},
}

# 若需要指定“正类”是谁（默认取 max(label)）
POS_LABEL = {           # 只有二分类任务需手动写
    "plaque":        1,  # 例：把“斑/厚”合并成正类可写 1;2->1 在 Dataset 里完成
    "form":          1,
    "surface":       1,
    "stenosis":      1,
    "vulnerability": 1,
}

def _adjust_tick_fontsize_by_label_length(ax, axis='x', 
                                          max_size=20, min_size=8, 
                                          scale_factor=10):
    """
    根据标签字符长度自动调整刻度字体大小。
    - ax: matplotlib Axes 对象
    - axis: 'x' 或 'y'
    - max_size, min_size: 允许的字体大小上下限
    - scale_factor: 控制缩放灵敏度，越大收缩越明显
    """
    if axis == 'x':
        labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    else:
        labels = [lbl.get_text() for lbl in ax.get_yticklabels()]
    if not labels:
        return
    max_len = max(len(t) for t in labels)
    # 根据最长 label 长度反比缩放
    size = int(max(min_size, min(max_size, (max_size * scale_factor) / max_len)))
    ax.tick_params(axis=axis, labelsize=size)

def plot_plaque_confusion_matrix(y_true, y_pred, epoch, output_dir='./fig'):
    """
    绘制并保存斑块分类任务的混淆矩阵（二分类）。
    使用 CLASS_DESCS['plaque'] 作为标签文字，并自动调整字体大小。
    """
    os.makedirs(output_dir, exist_ok=True)
    # 计算混淆矩阵（2类：None vs Plaque）
    labels = list(CLASS_DESCS['plaque'].keys())  # [0,1]
    names  = [CLASS_DESCS['plaque'][i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=names, yticklabels=names,
        annot_kws={"size": 36}
    )
    plt.xlabel('Predicted', fontsize=34)
    plt.ylabel('Actual', fontsize=34)
    ax = plt.gca()
    _adjust_tick_fontsize_by_label_length(ax, 'x', max_size=26, min_size=16, scale_factor=12)
    _adjust_tick_fontsize_by_label_length(ax, 'y', max_size=26, min_size=16, scale_factor=12)


    # 新增：设置 colorbar 的数值标尺字体大小
    cbar = ax.collections[0].colorbar  # 获取 colorbar 对象
    cbar.ax.tick_params(labelsize=24)  # 设置字体大小（可调整为其他值，如 20 或 30）

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plaque_confusion_matrix_epoch_{epoch+1}.png"), dpi=600)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, task_name, num_classes, epoch, output_dir="./fig"):
    """绘制并保存混淆矩阵"""
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算混淆矩阵
    # cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    labels = sorted(CLASS_DESCS[task_name].keys())      # 0,1,(2)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    names = [CLASS_DESCS[task_name][i] for i in labels]
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=names, yticklabels=names, annot_kws={"size": 36})

    plt.xlabel('Predicted', fontsize=34)
    plt.ylabel('Actual', fontsize=34)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    ax = plt.gca()
    _adjust_tick_fontsize_by_label_length(ax, 'x', max_size=26, min_size=16, scale_factor=12)
    _adjust_tick_fontsize_by_label_length(ax, 'y', max_size=26, min_size=16, scale_factor=12)

    # 新增：设置 colorbar 的数值标尺字体大小
    cbar = ax.collections[0].colorbar  # 获取 colorbar 对象
    cbar.ax.tick_params(labelsize=24)  # 设置字体大小（可调整为其他值，如 20 或 30）

    # 保存图片到指定文件夹
    plt.savefig(f"{output_dir}/{task_name}_confusion_matrix_epoch_{epoch+1}.png")
    plt.close()  # 关闭以释放内存

def calculate_and_plot_confusion_matrix(all_labels, all_preds, task, num_classes, epoch, output_dir='./fig'):
    """计算并保存每个任务的混淆矩阵"""
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    plot_confusion_matrix(y_true, y_pred, task, num_classes, epoch, output_dir)


def _prepare_dir(output_dir: str):
    """确保输出目录存在。"""
    os.makedirs(output_dir, exist_ok=True)

def _binary_prob(y_score):
    """
    统一二分类概率为 shape (N,).
    支持:
      - (N,)  : 已经是正类概率
      - (N,2) : 取第二列为正类概率
      - (N,1) : 只有一列时，视为正类概率
    """
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        return y_score
    if y_score.ndim == 2:
        if y_score.shape[1] == 2:
            return y_score[:, 1]
        if y_score.shape[1] == 1:
            return y_score[:, 0]
    raise ValueError(
        f"Expect prob vector shape (N,), (N,2) or (N,1) for binary task, "
        f"got {y_score.shape}"
    )


def plot_roc_curve(
    y_true,
    y_score,
    epoch: int,
    task_name: str,
    output_dir: str = "./fig",
    label_names: dict = CLASS_DESCS,
    label_colors: list = ['#1E88E5', '#D81B60', '#FFC107', '#004D40', '#8E24AA'],
    n_bootstrap: int = 1000,
    ci: int = 95,
    random_state: int = 42,
):
    """
    画 ROC 曲线并把 mean AUC 和 bootstrap 置信区间写进 legend。
    对二分类 / 多分类均适用。
    """
    os.makedirs(output_dir, exist_ok=True)
    rng_state = np.random.get_state()                # 保存外部随机状态
    np.random.seed(random_state)

    # ---------- 与旧版相同的数据整理 ----------
    y_true = np.asarray(y_true)
    y_obj  = np.asarray(y_score, dtype=object)
    if y_obj.dtype == object and y_obj.ndim == 1:
        try:
            y_score = np.vstack(y_obj).astype(float)
        except ValueError:
            y_score = y_obj
    else:
        y_score = y_obj.astype(float)

    if y_true.size == 0:
        print("[plot_roc_curve_ci] Empty input – skip")
        np.random.set_state(rng_state)
        return np.nan

# --- 在 plot_roc_curve 顶部“数据整理”之后，替换 is_binary 判定 & 二/多分类分支 ---
    # ---------- 判断二 / 多分类 ----------
    def _is_scalar_array(a):
        return a.ndim == 1 and (
            a.dtype != object or all(np.isscalar(x) for x in a)
        )

    # 若是 object 一维数组，尽量把每个元素变成至少一维，再 vstack 成规则二维
    if isinstance(y_score, np.ndarray) and y_score.dtype == object and y_score.ndim == 1:
        try:
            y_score = np.vstack([np.atleast_1d(np.asarray(x, dtype=float)) for x in y_score])
        except Exception:
            # 如果仍然不能堆叠，保持原样，后续分支会做更稳健的判断
            pass

    # 允许 (N,), (N,1), (N,2) 都走二分类分支（且 y_true 只有 0/1）
    is_binary = (
        (np.nanmax(y_true) <= 1) and (
            _is_scalar_array(y_score) or
            (isinstance(y_score, np.ndarray) and y_score.ndim == 2 and y_score.shape[1] in (1, 2))
        )
    )

    # ============ 二分类 ============
    if is_binary:
        pos = POS_LABEL.get(task_name, 1)
        prob = _binary_prob(y_score)

        fpr, tpr, _ = roc_curve(y_true, prob, pos_label=pos)
        auc_main = auc(fpr, tpr)

        # Bootstrap AUC
        auc_boot = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(y_true), len(y_true))
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], prob[idx], pos_label=pos)
            auc_boot.append(auc(fpr_b, tpr_b))
        lower = np.percentile(auc_boot, (100-ci)/2)
        upper = np.percentile(auc_boot, 100-(100-ci)/2)

        # 绘图（保持你原来的风格）
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, lw=2, color=label_colors[0],
                 label=(f"{label_names[task_name][pos]} "
                        f"(AUC = {auc_main:.2f} [{lower:.2f}-{upper:.2f}])"))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.xlabel("False Positive Rate", fontsize=34); plt.ylabel("True Positive Rate", fontsize=34)
        plt.tick_params(labelsize=24)
        plt.legend(frameon=False, loc="lower right", fontsize=24)
        plt.tight_layout()
        fname = f"{task_name}_roc_epoch_{epoch+1}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches="tight")
        plt.close()
        np.random.set_state(rng_state)
        return auc_main

    # ============ 多分类 ============
    classes = list(label_names[task_name].keys())
    labels_bin = label_binarize(y_true, classes=classes)


    # 强化健壮性：确保 y_score 是二维
    if not (isinstance(y_score, np.ndarray) and y_score.ndim == 2):
        raise ValueError(f"[plot_roc_curve] {task_name}: expected y_score shape (N,K) for multi-class, got {np.shape(y_score)}")

    n_cols = y_score.shape[1]

    plt.figure(figsize=(10, 8))
    auc_dict = {}
    lower_pct = (100-ci)/2; upper_pct = 100-lower_pct

    for i, cls in enumerate(classes):
        # 若概率列数 < 类别数，跳过缺失列，避免越界
        if i >= n_cols:
            # 可选：打印一次提示（不会中断训练）
            print(f"[plot_roc_curve] {task_name}: y_score has only {n_cols} columns; skip class index {i}.")
            continue
        if not np.any(y_true == cls):
            continue

        fpr, tpr, _ = roc_curve(labels_bin[:, i], y_score[:, i])
        auc_main = auc(fpr, tpr)

        # Bootstrap
        boot = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(y_true), len(y_true))
            fpr_b, tpr_b, _ = roc_curve(labels_bin[idx, i], y_score[idx, i])
            boot.append(auc(fpr_b, tpr_b))
        lower = np.percentile(boot, lower_pct)
        upper = np.percentile(boot, upper_pct)

        color = label_colors[i % len(label_colors)]
        plt.plot(
            fpr, tpr, lw=2, color=color,
            label=(f"{label_names[task_name][cls]} "
                   f"(AUC = {auc_main:.2f} [{lower:.2f}-{upper:.2f}])")
        )
        auc_dict[cls] = auc_main

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel("False Positive Rate", fontsize=34); plt.ylabel("True Positive Rate", fontsize=34)
    plt.tick_params(labelsize=24)
    plt.legend(frameon=False, loc="lower right", fontsize=24)
    plt.tight_layout()
    fname = f"{task_name}_roc_epoch_{epoch+1}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches="tight")
    plt.close()
    np.random.set_state(rng_state)
    return auc_dict


# ---------------- PR 曲线 + CI ----------------
def plot_pr_curve(
    y_true,
    y_score,
    epoch: int,
    task_name: str,
    output_dir: str = "./fig",
    label_names: dict = CLASS_DESCS,
    label_colors: list = ['#1E88E5', '#D81B60', '#FFC107', '#004D40', '#8E24AA'],
    n_bootstrap: int = 1000,
    ci: int = 95,
    random_state: int = 42,
):
    """
    画 PR 曲线并把平均精度 (AP) 及 bootstrap 置信区间写进 legend。
    二 / 多分类通用，稳健支持 (N,), (N,1), (N,2) 的二分类输入。
    """
    os.makedirs(output_dir, exist_ok=True)
    rng_state = np.random.get_state()
    np.random.seed(random_state)

    # -------- 数据整理（同 ROC） --------
    y_true = np.asarray(y_true)
    y_obj  = np.asarray(y_score, dtype=object)
    if y_obj.dtype == object and y_obj.ndim == 1:
        try:
            y_score = np.vstack([np.atleast_1d(np.asarray(x, dtype=float)) for x in y_obj])
        except Exception:
            y_score = y_obj
    else:
        y_score = y_obj.astype(float)

    if y_true.size == 0:
        np.random.set_state(rng_state)
        print("[plot_pr_curve] Empty input – skip")
        return np.nan

    def _is_scalar(a):
        return a.ndim == 1 and (a.dtype != object or all(np.isscalar(x) for x in a))

    # 允许 (N,), (N,1), (N,2) 都走二分类分支
    is_binary = (
        (np.nanmax(y_true) <= 1) and (
            _is_scalar(y_score) or
            (isinstance(y_score, np.ndarray) and y_score.ndim == 2 and y_score.shape[1] in (1, 2))
        )
    )

    # ==================== 二分类 ====================
    if is_binary:
        pos = POS_LABEL.get(task_name, 1)
        prob = _binary_prob(y_score)  # (N,), 自动兼容 (N,1)/(N,2)

        prec, rec, _ = precision_recall_curve(y_true, prob, pos_label=pos)
        ap_main = average_precision_score(y_true, prob, pos_label=pos)

        # Bootstrap
        ap_boot = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(y_true), len(y_true))
            ap_boot.append(average_precision_score(y_true[idx], prob[idx], pos_label=pos))
        low = np.percentile(ap_boot, (100-ci)/2)
        up  = np.percentile(ap_boot, 100-(100-ci)/2)

        # 绘图
        plt.figure(figsize=(10, 8))
        plt.plot(rec, prec, lw=2, color=label_colors[0],
                 label=(f"{label_names[task_name][pos]} (AP = {ap_main:.2f} [{low:.2f}-{up:.2f}])"))
        ax = plt.gca()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.xlabel("Recall", fontsize=34); plt.ylabel("Precision", fontsize=34)
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.legend(frameon=False, loc="lower left", fontsize=24)
        plt.tick_params(labelsize=24)
        plt.tight_layout()
        fname = f"{task_name}_pr_epoch_{epoch+1}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches="tight")
        plt.close()
        np.random.set_state(rng_state)
        return ap_main

    # ==================== 多分类 ====================
    classes = list(label_names[task_name].keys())
    labels_bin = label_binarize(y_true, classes=classes)

    # 确保概率是二维
    if not (isinstance(y_score, np.ndarray) and y_score.ndim == 2):
        raise ValueError(f"[plot_pr_curve] {task_name}: expected y_score shape (N,K) for multi-class, got {np.shape(y_score)}")
    n_cols = y_score.shape[1]

    plt.figure(figsize=(10, 8))
    ap_dict = {}
    low_pct = (100-ci)/2; up_pct = 100-low_pct

    for i, cls in enumerate(classes):
        if i >= n_cols:
            print(f"[plot_pr_curve] {task_name}: y_score has only {n_cols} columns; skip class index {i}.")
            continue
        if not np.any(y_true == cls):
            continue

        prec, rec, _ = precision_recall_curve(labels_bin[:, i], y_score[:, i])
        ap_main = average_precision_score(labels_bin[:, i], y_score[:, i])

        # Bootstrap
        boots = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(y_true), len(y_true))
            boots.append(average_precision_score(labels_bin[idx, i], y_score[idx, i]))
        low = np.percentile(boots, low_pct)
        up  = np.percentile(boots, up_pct)

        color = label_colors[i % len(label_colors)]
        plt.plot(rec, prec, lw=2, color=color,
                 label=(f"{label_names[task_name][cls]} (AP = {ap_main:.2f} [{low:.2f}-{up:.2f}])"))
        ap_dict[cls] = ap_main

    ax = plt.gca()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.xlabel("Recall", fontsize=34); plt.ylabel("Precision", fontsize=34)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.legend(frameon=False, loc="lower left", fontsize=24)
    plt.tick_params(labelsize=24)
    plt.tight_layout()
    fname = f"{task_name}_pr_epoch_{epoch+1}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches="tight")
    plt.close()
    np.random.set_state(rng_state)
    return ap_dict


def _sanitize_labels_scores(y_true, y_score):
    """去掉 label==-1 或 NaN 的样本；保持 y_true, y_score 对齐"""
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=object)   # 兼容 ragged list
    valid   = (~np.isnan(y_true)) & (y_true >= 0)
    return y_true[valid], y_score[valid]

def calculate_and_plot_roc_pr(
    y_true, y_score, task_name, epoch, output_dir="./fig"
):
    # ① 清洗无效标签
    y_true, y_score = _sanitize_labels_scores(y_true, y_score)

    # ② 样本全被过滤或只有 1 个类别 → 直接返回 NaN，避免 roc_curve 报错
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return np.nan, np.nan

    # ③ 正常绘图
    roc_res = plot_roc_curve(y_true, y_score, epoch, task_name, output_dir)
    pr_res  = plot_pr_curve (y_true, y_score, epoch, task_name, output_dir)
    return roc_res, pr_res

# ------------------------------------------------------------------
#  预测明细导出函数
# ------------------------------------------------------------------
def export_prediction_details(
        records: list,
        output_dir: str = "./fig",
        filename: str = "prediction_details.xlsx"):
    """
    将逐样本预测结果保存为 Excel & CSV。
    ----------
    records : list[dict]
        每个元素示例：
        {
            "ImagePath": "...",
            "plaque_true": 0, "plaque_pred": 1, "plaque_prob": 0.87,
            "form_true": -1,  "form_pred": -1,  "form_prob": np.nan,
            ...
            "vulnerability_true": -1, "vulnerability_pred": -1, "vulnerability_prob": np.nan
        }
    """
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(records)
    # Excel
    x_path = os.path.join(output_dir, filename)
    df.to_excel(x_path, index=False)
    # 兼顾纯文本
    c_path = os.path.splitext(x_path)[0] + ".csv"
    df.to_csv(c_path, index=False, encoding="utf-8-sig")
    print(f"[export_prediction_details] saved → {x_path} / {c_path}")


def visualize_tsne(feat_records,
                   save_path='fig/tsne_epoch0.png',
                   perplexity=30,
                   pca_dim=50,
                   random_state=42):
    """
    feat_records  : list[dict]  # 见上表
    save_path     : str         # *.png / *.svg
    """
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # ------------- 1. 整理矩阵 -------------
    vecs   = np.vstack([r['vec']      for r in feat_records])   # (M, C)
    tasks  = [r['task']  for r in feat_records]
    trues  = [r['true']  for r in feat_records]
    preds  = [r['pred']  for r in feat_records]

    # ------------- 2. 降维 -------------
    X = StandardScaler().fit_transform(vecs)
    if pca_dim and vecs.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
    tsne = TSNE(n_components=2,
                metric='cosine',
                perplexity=min(perplexity, len(X)//3),
                init='pca',
                learning_rate='auto',
                random_state=random_state)
    coords = tsne.fit_transform(X)        # (M, 2)

    # ------------- 3. 绘图映射 -------------
    shapes = dict(form='^', surface='s', echo='D',
                calcification='o', stenosis='v',
                vulnerability='*')

    # 颜色字典拆成「二分类」「三分类」
    color_bin = {0:'#A9CCE3', 1:'#1F77B4'}
    color_tri = {0:'#BBDEF0', 1:'#5DADE2', 2:'#154360'}

    def _color(task, cls):
        if task == 'vulnerability':
            return '#D62728' if cls==1 else '#1F77B4'
        elif task in ['echo','calcification']:
            return color_tri[int(cls)]
        else:
            return color_bin[int(cls)]
        
    # ------------- 4. 逐点 scatter -------------
    plt.figure(figsize=(12,10))
    ax = plt.gca()
    for i,(x,y) in enumerate(coords):
        task = tasks[i]; cls = trues[i]; pred = preds[i]
        marker = shapes[task]
        face   = _color(task, cls)
        edge   = 'red' if (task=='vulnerability' and pred!=cls) else 'black'
        lw     = 1.2 if (task=='vulnerability' and pred!=cls) else 0.2
        ax.scatter(x, y, s=60 if task=='vulnerability' else 25,
                   marker=marker, c=face,
                   edgecolors=edge, linewidths=lw, alpha=0.85)

    # ------------- 5. 图例 & 标注 -------------
    from matplotlib.lines import Line2D

    # --- Task legend (形状) ---
    task_handles = [Line2D([0],[0], marker=shp, color='k', linestyle='',
                        markerfacecolor='none', markersize=10,
                        label=tsk)         # tsk 为字符串
                    for tsk, shp in shapes.items()]

    # --- Class legend (颜色) ---
    class_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color_bin[0],
            label='binary‑0', markersize=8),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color_bin[1],
            label='binary‑1', markersize=8),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color_tri[0],
            label='tri‑0', markersize=8),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color_tri[1],
            label='tri‑1', markersize=8),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color_tri[2],
            label='tri‑2', markersize=8)
    ]

    first = ax.legend(handles=task_handles, title='Task (shape)',
                    loc='upper right', frameon=False, fontsize=14, title_fontsize=16)
    ax.add_artist(first)      # 固定第一段
    ax.legend(handles=class_handles, title='Class (color)',
            loc='lower right', frameon=False, fontsize=14, title_fontsize=16)

    ax.set_xticks([]); ax.set_yticks([])
    # ax.set_title('t‑SNE of task_out & vulnerability hidden')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

    # ------------- 6. 距离矩阵（最近邻平均）-------------
    # 可选：计算星形 → 5 任务各自最近邻距离，写 csv / print
    vul_idx = [i for i,t in enumerate(tasks) if t=='vulnerability']
    task_idx= {t:[i for i,tk in enumerate(tasks) if tk==t] for t in ['form','surface','echo','calcification','stenosis']}
    from scipy.spatial.distance import cdist
    ddict = {}
    for t,idxs in task_idx.items():
        d = cdist(coords[vul_idx], coords[idxs]).min(1).mean()  # mean min‑dist
        ddict[t] = d
    print('Avg nearest distance (易损★→各细节):', ddict)
    return coords, tasks, trues, preds
