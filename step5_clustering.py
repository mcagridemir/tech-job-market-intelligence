# ============================================================
# STEP 5: Unsupervised Learning — Role Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3d4a', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0b0', 'ytick.color': '#a0a0b0',
    'text.color': '#e0e0e0', 'grid.color': '#2a2d3a', 'grid.alpha': 0.5,
})
PALETTE = ['#00d4aa', '#ff6b9d', '#ffa94d', '#74c7ec', '#cba6f7', '#89dceb']

df = pd.read_csv('jobs_nlp.csv')

# ── A. Feature prep ──────────────────────────────────────────
cluster_features = []
le_tmp = LabelEncoder()

if 'experience_level' in df.columns:
    df['exp_enc'] = le_tmp.fit_transform(df['experience_level'].astype(str))
    cluster_features.append('exp_enc')
if 'company_size' in df.columns:
    df['size_enc'] = le_tmp.fit_transform(df['company_size'].astype(str))
    cluster_features.append('size_enc')
if 'remote_ratio' in df.columns:
    cluster_features.append('remote_ratio')
if 'skill_count' in df.columns:
    cluster_features.append('skill_count')

cluster_features.append('salary_in_usd')
X_clust = df[cluster_features].fillna(0)

scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_clust)

# ── B. Find optimal K ────────────────────────────────────────
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, lbl))

best_k = K_range[np.argmax(sil_scores)]
print(f"Optimal k = {best_k}  (silhouette = {max(sil_scores):.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

# ── C. PCA for 2D vis ────────────────────────────────────────
pca       = PCA(n_components=2, random_state=42)
X_pca     = pca.fit_transform(X_scaled)
df['pc1'] = X_pca[:, 0]
df['pc2'] = X_pca[:, 1]

# ── D. Cluster profiling ─────────────────────────────────────
profile_cols = ['salary_in_usd', 'skill_count']
if 'remote_ratio' in df.columns:
    profile_cols.append('remote_ratio')
if 'exp_enc' in df.columns:
    profile_cols.append('exp_enc')

cluster_profiles = df.groupby('cluster')[profile_cols].mean()
cluster_labels   = []
for i, row in cluster_profiles.iterrows():
    sal = row['salary_in_usd']
    if sal > 130_000:
        label = f'Cluster {i}: 🚀 Top Earners'
    elif sal > 90_000:
        label = f'Cluster {i}: 💼 Senior Pros'
    elif sal > 60_000:
        label = f'Cluster {i}: 📈 Growing Roles'
    else:
        label = f'Cluster {i}: 🌱 Entry Level'
    cluster_labels.append(label)
    print(f"  {label}  |  Avg salary ${sal:,.0f}  |  {len(df[df.cluster==i])} jobs")

# ── E. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Job Role Clustering (Unsupervised ML)',
             fontsize=18, fontweight='bold', color='white')
fig.patch.set_facecolor('#0f1117')

# Elbow + Silhouette
ax = axes[0]
ax2 = ax.twinx()
ax.plot(list(K_range), inertias, color='#00d4aa', lw=2.5, marker='o', ms=6, label='Inertia')
ax2.plot(list(K_range), sil_scores, color='#ff6b9d', lw=2.5, marker='s', ms=6, ls='--', label='Silhouette')
ax.axvline(best_k, color='#ffa94d', ls=':', lw=2, alpha=0.8)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia', color='#00d4aa')
ax2.set_ylabel('Silhouette Score', color='#ff6b9d')
ax.set_title('Optimal k Selection\n(Elbow + Silhouette)', fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0)
ax.grid(axis='y')

# PCA scatter
ax = axes[1]
for i in range(best_k):
    mask = df['cluster'] == i
    ax.scatter(df.loc[mask, 'pc1'], df.loc[mask, 'pc2'],
               color=PALETTE[i % len(PALETTE)], s=12, alpha=0.5,
               label=cluster_labels[i])
ax.set_title('PCA Cluster Visualization\n(2D projection)', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.legend(fontsize=7, framealpha=0.15, loc='best')
ax.grid(True)

# Salary by cluster
ax = axes[2]
cluster_sal = [df[df['cluster'] == i]['salary_in_usd'] for i in range(best_k)]
bp = ax.boxplot(cluster_sal,
                labels=[f'C{i}' for i in range(best_k)],
                patch_artist=True,
                medianprops=dict(color='white', lw=2),
                whiskerprops=dict(color='#a0a0b0'),
                capprops=dict(color='#a0a0b0'),
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_title('Salary Distribution\nby Cluster', fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_ylabel('Salary (USD)')
import matplotlib.ticker as mticker
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.grid(axis='y')

plt.tight_layout()
plt.savefig('clustering.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("✅ Saved → clustering.png")

df.to_csv('jobs_final.csv', index=False)
print("✅ Saved → jobs_final.csv")
