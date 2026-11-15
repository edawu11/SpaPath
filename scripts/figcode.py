import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.cm as cm
def plot_cropped_celltypes(adata,slice_name, dir_raw,dir_fig,img_path,celltype_key,
                            celltype_map=None, label_to_color=None,alpha=1.0,
                            savename=None, show=True, figsize=(5, 5),s=1, dpi=300):
    adata.obs['CellType_major'] = adata.obs[celltype_key].map(celltype_map)

    img = cv2.imread(img_path)
    coords = adata.obsm['spatial']
    labels = adata.obs['CellType_major'].values

    x_min, x_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    y_min, y_max = int(coords[:, 1].min()), int(coords[:, 1].max())
    img_h, img_w = img.shape[:2]
    x_min = max(x_min, 0)
    x_max = min(x_max, img_w)
    y_min = max(y_min, 0)
    y_max = min(y_max, img_h)
    img_cropped = img[y_min:y_max, x_min:x_max]
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    coords_cropped = coords - np.array([x_min, y_min])

    colors = [label_to_color.get(label, "#999999") for label in labels]

    plt.figure(figsize=figsize)
    plt.imshow(img_rgb, alpha=0.8)
    plt.scatter(coords_cropped[:, 0], coords_cropped[:, 1], c=colors, s=s, alpha=alpha)
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().set_position([0, 0, 1, 1])
    out_path = f"{dir_fig}{slice_name}_{savename}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=False)

    if show:
        plt.show()
        plt.close()
    else:
        plt.close()

def plot_cropped_celltypes_no_HE(adata,slice_name, dir_raw,dir_fig,img_path,celltype_key,
                            celltype_map=None, label_to_color=None,
                            savename=None, show=True, figsize=(5, 5),s=1, dpi=300):
    """Plot cropped spatial labels without an H&E background."""

    if celltype_map is None:
        celltype_map = {
            "Stromal": "Stromal",
            "Immune": "Immune",
            "RBC": "Others",
            "Tumor": "Tumor"
        }

    if label_to_color is None:
        label_to_color = {
            'Stromal': "#ffd460",
            'Immune': "#878ecd",
            'Others': "#a8e6cf",
            'Tumor': "#e84a5f"
        }

    adata.obs['CellType_major'] = adata.obs[celltype_key].map(celltype_map)

    coords = adata.obsm['spatial']
    labels = adata.obs['CellType_major'].values

    x_min, x_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    y_min, y_max = int(coords[:, 1].min()), int(coords[:, 1].max())
    coords_cropped = coords - np.array([x_min, y_min])

    colors = [label_to_color.get(label, "#999999") for label in labels]

    plt.figure(figsize=figsize)
    plt.scatter(coords_cropped[:, 0], coords_cropped[:, 1], c=colors, s=s, alpha=1.0)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().set_position([0, 0, 1, 1])
    out_path = f"{dir_fig}{slice_name}_{savename}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=False)

    if show:
        plt.show()
        plt.close()
    else:
        plt.close()

def plot_cropped_celltypes_continuous(adata, slice_name, dir_raw, dir_fig, img_path,
                                      celltype_key, celltype_map=None,
                                      label_to_color=None, tumor_score_key='tumor_score',
                                      cancer_cmap='Reds', savename=None, show=True,
                                      figsize=(5,5), s=1, dpi=300):
    """Plot normal spots in one color and cancer spots by a continuous score."""

    adata.obs['CellType_major'] = adata.obs[celltype_key].map(celltype_map)

    img = cv2.imread(img_path)
    coords = adata.obsm['spatial']
    labels = adata.obs['CellType_major'].values
    tumor_scores = adata.obs[tumor_score_key].values

    x_min, x_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    y_min, y_max = int(coords[:, 1].min()), int(coords[:, 1].max())
    img_h, img_w = img.shape[:2]
    x_min = max(x_min, 0)
    x_max = min(x_max, img_w)
    y_min = max(y_min, 0)
    y_max = min(y_max, img_h)
    img_cropped = img[y_min:y_max, x_min:x_max]
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    coords_cropped = coords - np.array([x_min, y_min])

    cancer_scores = np.array([score for lab, score in zip(labels, tumor_scores) if lab != 'Normal'], dtype=float)
    vmin, vmax = np.min(cancer_scores), np.max(cancer_scores)

    colors = []
    cmap = plt.get_cmap(cancer_cmap)
    for lab, score in zip(labels, tumor_scores):
        if lab == 'Normal' or score is None:
            colors.append(label_to_color.get('Normal', "#BDBDBD"))
        else:
            norm_score = (score - vmin) / (vmax - vmin + 1e-8)
            colors.append(cmap(norm_score))

    plt.figure(figsize=figsize)
    plt.imshow(img_rgb, alpha=0.8)
    plt.scatter(coords_cropped[:, 0], coords_cropped[:, 1], c=colors, s=s, alpha=1.0)
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().set_position([0, 0, 1, 1])

    out_path = f"{dir_fig}{slice_name}_{savename}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=False)

    if show:
        plt.show()
        plt.close()
    else:
        plt.close()

def plot_cropped_continuous(adata, slice_name, dir_raw, dir_fig, img_path,
                            value_key, cmap="viridis", vmin=None, vmax=None,
                            savename=None, show=True, figsize=(5, 5), s=1, dpi=300):

    values = adata.obs[value_key].astype(float).values

    img = cv2.imread(img_path)
    coords = adata.obsm['spatial']

    x_min, x_max = int(coords[:, 0].min()), int(coords[:, 0].max())
    y_min, y_max = int(coords[:, 1].min()), int(coords[:, 1].max())
    img_h, img_w = img.shape[:2]
    x_min = max(x_min, 0)
    x_max = min(x_max, img_w)
    y_min = max(y_min, 0)
    y_max = min(y_max, img_h)
    img_cropped = img[y_min:y_max, x_min:x_max]
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    coords_cropped = coords - np.array([x_min, y_min])

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb, alpha=0.8)
    sc = ax.scatter(coords_cropped[:, 0], coords_cropped[:, 1], c=values, cmap=cmap,
                    s=s, alpha=1.0, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.margins(0, 0)
    ax.set_position([0, 0.1, 1, 0.9])

    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08)
    cbar.set_ticks([])
    cbar.set_label("")
    out_path = f"{dir_fig}{slice_name}_{savename}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=False)

    if show:
        plt.show()
        plt.close()
    else:
        plt.close()
def plot_method_scores(df, metric='ACC', dir_fig='./', name_map=None):

    custom_palette = {
        'ours': '#D55E00',
        'scmap': '#009E73',
        'CAESAR': '#E69F00',
        'scvi': '#CC79A7',
        'stands': '#0072B2',
        'STAMapper': '#56B4E9',
        'harmony': '#F0E442'
    }

    metric = metric.upper()
    if metric not in df.columns.str.upper().tolist():
        raise ValueError(f"Metric column '{metric}' was not found in the DataFrame.")
    metric_col = [col for col in df.columns if col.upper() == metric][0]

    df = df.copy()
    if name_map is not None:
        df['method_display'] = df['method'].map(name_map).fillna(df['method'])
    else:
        df['method_display'] = df['method']

    summary = df.groupby(['method', 'method_display'])[metric_col].agg(['mean', 'min', 'max']).reset_index()
    summary['yerr_lower'] = summary['mean'] - summary['min']
    summary['yerr_upper'] = summary['max'] - summary['mean']
    summary = summary.sort_values(by='mean', ascending=False)
    method_order = summary['method_display'].tolist()
    method_key_order = summary['method'].tolist()
    colors = [custom_palette.get(m, '#888888') for m in method_key_order]

    sns.set(style="white", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x=method_order,
        height=summary['mean'],
        color=colors,
        edgecolor='black',
        linewidth=1.2,
        yerr=[summary['yerr_lower'], summary['yerr_upper']],
        capsize=5,
        error_kw={'elinewidth': 1.2},
        width=0.5,
        alpha=0.6
    )

    for method_key, method_display in zip(method_key_order, method_order):
        y_values = df[df['method'] == method_key][metric_col].values
        x_pos = method_order.index(method_display)
        jitter = (np.random.rand(len(y_values)) - 0.5) * 0.2
        ax.scatter(
            [x_pos + j for j in jitter],
            y_values,
            marker='^',
            color='black',
            s=50,
            alpha=0.8,
            zorder=5
        )

    ax.set_ylabel(metric_col, fontsize=14, weight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, fontweight="bold", rotation=15)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.4)
    ax.spines['bottom'].set_linewidth(1.4)
    ax.tick_params(left=True, bottom=True)
    sns.despine(trim=True)
    plt.tight_layout()

    metric_safe = metric_col.lower()
    save_path = f"{dir_fig}method_{metric_safe}_score_triangles.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
