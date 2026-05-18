# ============================================================
# STEP 1: Data Loading, Cleaning & First Look
# Project: Tech Job Market Intelligence — Germany & Beyond
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ────────────────────────────────────────────
# Dataset: "Jobs in Data" from Kaggle
# Download URL: https://www.kaggle.com/datasets/hummaamqaasim/jobs-in-data
# File: jobs_in_data.csv  (~9,000 rows)
# Place the CSV in the same folder as this script.

df = pd.read_csv('jobs_in_data.csv')

print("=" * 55)
print("  RAW DATA SNAPSHOT")
print("=" * 55)
print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns      : {list(df.columns)}")
print(f"  Memory usage : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ── 2. INITIAL QUALITY CHECK ────────────────────────────────
print("\n── Missing Values ──────────────────────────────────────")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  ✅ No missing values!")

print("\n── Duplicate Rows ──────────────────────────────────────")
dupes = df.duplicated().sum()
print(f"  Found {dupes} duplicate rows")

print("\n── Column Data Types ───────────────────────────────────")
print(df.dtypes.to_string())

# ── 3. CLEANING STEPS ───────────────────────────────────────
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Rename columns to snake_case for convenience
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(' ', '_'))

# Salary: keep only USD rows OR convert (dataset already has salary_in_usd)
if 'salary_in_usd' in df.columns:
    df = df[df['salary_in_usd'] > 0]
    # Remove extreme outliers (below 5th and above 99th percentile)
    low, high = df['salary_in_usd'].quantile([0.05, 0.99])
    df = df[(df['salary_in_usd'] >= low) & (df['salary_in_usd'] <= high)]

# Standardize experience levels
exp_map = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
if 'experience_level' in df.columns:
    df['experience_level'] = df['experience_level'].map(exp_map).fillna(df['experience_level'])

# Standardize employment type
emp_map = {'FT': 'Full-Time', 'PT': 'Part-Time', 'CT': 'Contract', 'FL': 'Freelance'}
if 'employment_type' in df.columns:
    df['employment_type'] = df['employment_type'].map(emp_map).fillna(df['employment_type'])

# Add year as integer (it may already be int)
if 'work_year' in df.columns:
    df['work_year'] = df['work_year'].astype(int)

# ── 4. FEATURE ENGINEERING (basic, more in Step 3) ──────────
# Salary buckets
bins   = [0, 60_000, 100_000, 150_000, 999_999]
labels = ['Low (<60k)', 'Mid (60–100k)', 'High (100–150k)', 'Very High (>150k)']
df['salary_bucket'] = pd.cut(df['salary_in_usd'], bins=bins, labels=labels)

# Is company in Germany?
if 'company_location' in df.columns:
    df['is_germany'] = df['company_location'].str.upper() == 'DE'

# ── 5. CLEAN DATA SUMMARY ───────────────────────────────────
print("\n" + "=" * 55)
print("  CLEANED DATA SUMMARY")
print("=" * 55)
print(f"  Rows after cleaning : {len(df):,}")
print(f"  Columns             : {len(df.columns)}")
print(f"\n  Salary (USD) stats:")
print(df['salary_in_usd'].describe().apply(lambda x: f"    {x:,.0f}").to_string())

if 'experience_level' in df.columns:
    print(f"\n  Experience levels   : {df['experience_level'].value_counts().to_dict()}")
if 'is_germany' in df.columns:
    print(f"  Germany companies   : {df['is_germany'].sum()} rows")
if 'job_category' in df.columns:
    print(f"\n  Job categories (top 5):")
    print(df['job_category'].value_counts().head(5).to_string())

# ── 6. SAVE CLEANED DATA ─────────────────────────────────────
df.to_csv('jobs_cleaned.csv', index=False)
print("\n  ✅ Saved → jobs_cleaned.csv")
print("=" * 55)
