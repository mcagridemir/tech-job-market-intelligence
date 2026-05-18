# ============================================================
# STEP 4: Salary Prediction Model + SHAP Explainability
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3d4a', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0b0', 'ytick.color': '#a0a0b0',
    'text.color': '#e0e0e0', 'grid.color': '#2a2d3a', 'grid.alpha': 0.5,
})
ACCENT, ACCENT2, ACCENT3 = '#00d4aa', '#ff6b9d', '#ffa94d'

df = pd.read_csv('jobs_nlp.csv')

# ── A. Feature Engineering ───────────────────────────────────
cat_features = ['experience_level', 'employment_type', 'company_size',
                'job_category', 'employee_residence', 'company_location']
cat_features  = [c for c in cat_features if c in df.columns]

num_features  = ['work_year', 'skill_count']
if 'remote_ratio' in df.columns:
    num_features.append('remote_ratio')
if 'is_germany' in df.columns:
    num_features.append('is_germany')

# Encode categoricals
le_dict = {}
df_enc  = df.copy()
for col in cat_features:
    le = LabelEncoder()
    df_enc[col + '_enc'] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

feature_cols = [c + '_enc' for c in cat_features] + num_features
feature_cols = [c for c in feature_cols if c in df_enc.columns]

X = df_enc[feature_cols].fillna(0)
y = df_enc['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── B. Model Comparison ──────────────────────────────────────
models = {
    'Ridge Regression':       Ridge(alpha=10),
    'Random Forest':          RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting':      GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds      = model.predict(X_test)
    mae        = mean_absolute_error(y_test, preds)
    r2         = r2_score(y_test, preds)
    results[name] = {'model': model, 'mae': mae, 'r2': r2, 'preds': preds}
    print(f"  {name:<25}  MAE=${mae:>8,.0f}   R²={r2:.3f}")

# Best model = Gradient Boosting
best_name  = max(results, key=lambda k: results[k]['r2'])
best_model = results[best_name]['model']
best_preds = results[best_name]['preds']
print(f"\n  🏆 Best model: {best_name}")

# ── C. SHAP Explainability ───────────────────────────────────
print("\nComputing SHAP values (this takes ~30 sec)...")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Friendly feature names
friendly = {c + '_enc': c.replace('_', ' ').title() for c in cat_features}
friendly.update({'work_year': 'Year', 'skill_count': 'Skills Detected',
                 'remote_ratio': 'Remote %', 'is_germany': 'Germany-based'})
X_test_display         = X_test.copy()
X_test_display.columns = [friendly.get(c, c) for c in X_test.columns]

# ── D. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Salary Prediction Model + SHAP Explainability',
             fontsize=18, fontweight='bold', color='white')
fig.patch.set_facecolor('#0f1117')

# D1 – Model comparison
ax = axes[0, 0]
names = list(results.keys())
r2s   = [results[n]['r2'] for n in names]
maes  = [results[n]['mae'] for n in names]
x     = np.arange(len(names))
bars  = ax.bar(x, r2s, color=[ACCENT, ACCENT2, '#ffa94d'], width=0.5, edgecolor='none')
for bar, r2, mae in zip(bars, r2s, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'R²={r2:.3f}\nMAE=${mae/1000:.0f}k', ha='center', fontsize=9,
            color='white', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
ax.set_ylabel('R² Score'); ax.set_title('Model Comparison', fontweight='bold')
ax.set_ylim(0, 1.1); ax.grid(axis='y')

# D2 – Actual vs Predicted
ax = axes[0, 1]
ax.scatter(y_test, best_preds, alpha=0.35, color=ACCENT, s=8)
line_min = min(y_test.min(), best_preds.min())
line_max = max(y_test.max(), best_preds.max())
ax.plot([line_min, line_max], [line_min, line_max], color=ACCENT2, lw=2, ls='--')
ax.set_xlabel('Actual Salary (USD)'); ax.set_ylabel('Predicted Salary (USD)')
ax.set_title(f'Actual vs Predicted\n({best_name})', fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.grid(True)

# D3 – SHAP summary (mean |SHAP|)
ax = axes[1, 0]
mean_shap = np.abs(shap_values).mean(axis=0)
feat_names_display = X_test_display.columns.tolist()
idx_sorted = np.argsort(mean_shap)
colors = [ACCENT if v > 0 else ACCENT2 for v in mean_shap[idx_sorted]]
ax.barh([feat_names_display[i] for i in idx_sorted],
        mean_shap[idx_sorted], color=ACCENT)
ax.set_title('SHAP Feature Importance\n(mean |SHAP value|)', fontweight='bold')
ax.set_xlabel('Mean |SHAP Value|  →  Impact on salary prediction')
ax.grid(axis='x')

# D4 – Residuals
ax = axes[1, 1]
residuals = y_test.values - best_preds
ax.scatter(best_preds, residuals, alpha=0.3, color=ACCENT3, s=8)
ax.axhline(0, color=ACCENT2, lw=2, ls='--')
ax.set_xlabel('Predicted Salary')
ax.set_ylabel('Residual (Actual − Predicted)')
ax.set_title('Residual Plot\n(good model = random scatter around 0)', fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax.grid(True)

plt.tight_layout()
plt.savefig('salary_model.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("✅ Saved → salary_model.png")

# Save model artifacts for Streamlit app
import joblib, json
joblib.dump(best_model, 'salary_model.pkl')
joblib.dump(le_dict,    'label_encoders.pkl')
with open('feature_cols.json', 'w') as f:
    json.dump({'features': feature_cols, 'cat_features': cat_features,
               'num_features': num_features}, f)
print("✅ Saved → salary_model.pkl, label_encoders.pkl, feature_cols.json")
