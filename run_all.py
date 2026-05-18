"""
Master runner — executes all steps in order.
Usage: python run_all.py
Make sure jobs_in_data.csv is in the same directory!
"""
import subprocess, sys, os

steps = [
    ("Step 1 — Data Loading & Cleaning",  "step1_data_loading.py"),
    ("Step 2 — EDA & Visualizations",     "step2_eda.py"),
    ("Step 3 — NLP Analysis",             "step3_nlp.py"),
    ("Step 4 — Salary Model + SHAP",      "step4_salary_model.py"),
    ("Step 5 — Clustering",               "step5_clustering.py"),
]

for label, script in steps:
    print(f"\n{'='*55}")
    print(f"  ▶  {label}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ {script} failed. Fix the error above and re-run.")
        sys.exit(1)

print("\n" + "="*55)
print("  ✅  ALL STEPS COMPLETE!")
print("="*55)
print("\nGenerated files:")
for f in sorted(os.listdir('.')):
    if f.endswith(('.csv', '.pkl', '.json', '.png')):
        size = os.path.getsize(f)
        print(f"  {f:<35} {size/1024:>6.1f} KB")

print("\n🚀 Launch the app:")
print("   streamlit run app.py")
