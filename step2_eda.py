# ============================================================
# STEP 2: Exploratory Data Analysis (EDA)
# Project: Tech Job Market Intelligence
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4a',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3a',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
ACCENT   = '#00d4aa'   # teal-green
ACCENT2  = '#ff6b9d'   # pink
ACCENT3  = '#ffa94d'   # orange
PALETTE  = [ACCENT, ACCENT2, ACCENT3, '#74c7ec', '#cba6f7', '#89dceb']

df = pd.read_csv('jobs_cleaned.csv')

# ── FIGURE 1: Market Overview (2×3 grid) ────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Tech Job Market Intelligence — Global Overview',
             fontsize=20, fontweight='bold', color='white', y=1.01)
fig.patch.set_facecolor('#0f1117')

# 1-A  Salary distribution
ax = axes[0, 0]
ax.hist(df['salary_in_usd'], bins=50, color=ACCENT, edgecolor='none', alpha=0.85)
ax.axvline(df['salary_in_usd'].median(), color=ACCENT2, lw=2, ls='--',
           label=f"Median: ${df['salary_in_usd'].median():,.0f}")
ax.set_title('Salary Distribution (USD)', fontweight='bold')
ax.set_xlabel('Annual Salary (USD)')
ax.set_ylabel('Count')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.legend(framealpha=0)
ax.grid(axis='y')

# 1-B  Salary by experience
ax = axes[0, 1]
order = ['Entry', 'Mid', 'Senior', 'Executive']
order = [o for o in order if o in df['experience_level'].unique()]
sns.boxplot(data=df, x='experience_level', y='salary_in_usd',
            order=order, palette=PALETTE, ax=ax, linewidth=1.2,
            flierprops=dict(marker='o', markersize=2, alpha=0.3))
ax.set_title('Salary by Experience Level', fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Salary (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.grid(axis='y')

# 1-C  Jobs over time
ax = axes[0, 2]
year_counts = df['work_year'].value_counts().sort_index()
ax.fill_between(year_counts.index, year_counts.values, color=ACCENT, alpha=0.4)
ax.plot(year_counts.index, year_counts.values, color=ACCENT, lw=2.5, marker='o', ms=6)
ax.set_title('Job Postings Over Time', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Jobs')
ax.grid(axis='y')
ax.set_xticks(year_counts.index)

# 1-D  Top 10 job categories
ax = axes[1, 0]
cat_col = 'job_category' if 'job_category' in df.columns else 'job_title'
top_cats = df[cat_col].value_counts().head(10)
bars = ax.barh(top_cats.index[::-1], top_cats.values[::-1], color=ACCENT, edgecolor='none')
for bar, val in zip(bars, top_cats.values[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=8, color='#a0a0b0')
ax.set_title('Top Job Categories', fontweight='bold')
ax.set_xlabel('Count')
ax.grid(axis='x')

# 1-E  Remote ratio
ax = axes[1, 1]
if 'remote_ratio' in df.columns:
    remote_labels = {0: 'On-site', 50: 'Hybrid', 100: 'Remote'}
    remote_counts = df['remote_ratio'].map(remote_labels).value_counts()
    wedges, texts, autotexts = ax.pie(
        remote_counts.values,
        labels=remote_counts.index,
        autopct='%1.1f%%',
        colors=PALETTE[:len(remote_counts)],
        startangle=90,
        wedgeprops=dict(edgecolor='#0f1117', linewidth=2)
    )
    for at in autotexts:
        at.set_color('#0f1117')
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title('Work Arrangement', fontweight='bold')
else:
    emp_counts = df['employment_type'].value_counts()
    ax.pie(emp_counts.values, labels=emp_counts.index,
           autopct='%1.1f%%', colors=PALETTE[:len(emp_counts)],
           wedgeprops=dict(edgecolor='#0f1117', linewidth=2))
    ax.set_title('Employment Type', fontweight='bold')

# 1-F  Company size
ax = axes[1, 2]
if 'company_size' in df.columns:
    size_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    size_counts = df['company_size'].map(size_map).value_counts()
    bars = ax.bar(size_counts.index, size_counts.values,
                  color=[ACCENT, ACCENT2, ACCENT3], edgecolor='none', width=0.5)
    for bar, val in zip(bars, size_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:,}', ha='center', fontsize=10)
    ax.set_title('Company Size Distribution', fontweight='bold')
    ax.set_xlabel('Company Size')
    ax.set_ylabel('Count')
    ax.grid(axis='y')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
print("✅ Saved → eda_overview.png")

# ── FIGURE 2: Germany Deep-Dive ──────────────────────────────
if 'is_germany' in df.columns and df['is_germany'].sum() > 0:
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle('🇩🇪 Germany Tech Jobs — Deep Dive',
                  fontsize=18, fontweight='bold', color='white')
    fig2.patch.set_facecolor('#0f1117')

    de = df[df['is_germany']]
    global_ = df[~df['is_germany']]

    # Salary comparison DE vs World
    ax = axes2[0]
    data_comp = [global_['salary_in_usd'].dropna(), de['salary_in_usd'].dropna()]
    bp = ax.boxplot(data_comp, labels=['Global', 'Germany'],
                    patch_artist=True,
                    boxprops=dict(facecolor=ACCENT, alpha=0.7),
                    medianprops=dict(color=ACCENT2, lw=2),
                    whiskerprops=dict(color='#a0a0b0'),
                    capprops=dict(color='#a0a0b0'),
                    flierprops=dict(marker='o', markersize=2, alpha=0.3, color=ACCENT))
    bp['boxes'][1].set_facecolor(ACCENT2)
    ax.set_title('Salary: Germany vs Global', fontweight='bold')
    ax.set_ylabel('Salary (USD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    ax.grid(axis='y')

    # Top categories in Germany
    ax = axes2[1]
    de_cats = de[cat_col].value_counts().head(7)
    ax.barh(de_cats.index[::-1], de_cats.values[::-1], color=ACCENT3)
    ax.set_title('Top Roles in Germany', fontweight='bold')
    ax.set_xlabel('Count')
    ax.grid(axis='x')

    # Experience distribution in Germany
    ax = axes2[2]
    de_exp = de['experience_level'].value_counts()
    ax.pie(de_exp.values, labels=de_exp.index, autopct='%1.1f%%',
           colors=PALETTE[:len(de_exp)],
           wedgeprops=dict(edgecolor='#0f1117', linewidth=2))
    ax.set_title('Experience Levels in Germany', fontweight='bold')

    plt.tight_layout()
    plt.savefig('eda_germany.png', dpi=150, bbox_inches='tight',
                facecolor='#0f1117')
    print("✅ Saved → eda_germany.png")

print("EDA complete!")
