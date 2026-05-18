# ============================================================
# STEP 6: Streamlit App — Tech Job Market Intelligence
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib, json, warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Job Market Intelligence 🇩🇪",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container { background: #0f1117; }
    .main .block-container { padding-top: 1rem; }
    h1 { color: #00d4aa; font-size: 2.2rem !important; }
    h2 { color: #e0e0e0; border-bottom: 1px solid #3a3d4a; padding-bottom: 0.3rem; }
    h3 { color: #00d4aa; }
    .metric-card {
        background: #1a1d27; border-radius: 12px; padding: 1rem 1.5rem;
        border: 1px solid #3a3d4a; margin-bottom: 0.5rem;
    }
    .stMetric label { color: #a0a0b0 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #00d4aa !important; font-size: 1.6rem !important; }
    .prediction-box {
        background: linear-gradient(135deg, #1a1d27, #0f1117);
        border: 2px solid #00d4aa; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin-top: 1rem;
    }
    .prediction-box h2 { color: #00d4aa; font-size: 2.5rem !important; border: none; }
    .tag {
        display: inline-block; background: #1a1d27;
        border: 1px solid #00d4aa; color: #00d4aa;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.8rem; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('jobs_final.csv')

@st.cache_resource
def load_model():
    try:
        model = joblib.load('salary_model.pkl')
        le_dict = joblib.load('label_encoders.pkl')
        with open('feature_cols.json') as f:
            fc = json.load(f)
        return model, le_dict, fc
    except:
        return None, None, None

df = load_data()
model, le_dict, fc = load_model()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔭 Filters")
    years = sorted(df['work_year'].unique())
    sel_year = st.multiselect("Year", years, default=years[-2:])

    exp_levels = df['experience_level'].unique().tolist() if 'experience_level' in df.columns else []
    sel_exp    = st.multiselect("Experience Level", exp_levels, default=exp_levels)

    germany_only = st.checkbox("🇩🇪 Germany only", value=False)

    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown("""
    Built to support a Master's application
    in Informatik/AI to German universities.

    **Stack:** Python · sklearn · SHAP
    · Streamlit · pandas · seaborn

    **Dataset:** Jobs in Data (Kaggle)
    ~9,000 real job postings
    """)

# ── Filter data ───────────────────────────────────────────────
mask = df['work_year'].isin(sel_year)
if sel_exp and 'experience_level' in df.columns:
    mask &= df['experience_level'].isin(sel_exp)
if germany_only and 'is_germany' in df.columns:
    mask &= df['is_germany'] == True
filtered = df[mask]

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🤖 Tech Job Market Intelligence")
st.markdown("**Analysing the global tech job market · Germany focus · AI-powered salary predictor**")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Jobs",        f"{len(filtered):,}")
k2.metric("Median Salary",     f"${filtered['salary_in_usd'].median():,.0f}")
k3.metric("Top Category",
          filtered['job_category'].mode()[0] if 'job_category' in filtered.columns else "—")
k4.metric("Avg Skills/Role",
          f"{filtered['skill_count'].mean():.1f}" if 'skill_count' in filtered.columns else "—")
k5.metric("Germany Jobs",
          str(int(filtered['is_germany'].sum())) if 'is_germany' in filtered.columns else "—")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "💰 Salary Predictor", "🌍 Germany Deep Dive"])

# ────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    # Salary by experience
    with col1:
        st.subheader("Salary by Experience Level")
        if 'experience_level' in filtered.columns:
            order = ['Entry', 'Mid', 'Senior', 'Executive']
            order = [o for o in order if o in filtered['experience_level'].unique()]
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
            ax.set_facecolor('#1a1d27')
            data_list = [filtered[filtered['experience_level'] == o]['salary_in_usd'] for o in order]
            bp = ax.boxplot(data_list, labels=order, patch_artist=True,
                            medianprops=dict(color='white', lw=2),
                            whiskerprops=dict(color='#a0a0b0'),
                            capprops=dict(color='#a0a0b0'),
                            flierprops=dict(marker='o', markersize=2, alpha=0.2))
            colors = ['#00d4aa', '#74c7ec', '#ff6b9d', '#ffa94d']
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c); patch.set_alpha(0.75)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
            ax.tick_params(colors='#a0a0b0'); ax.grid(axis='y', color='#2a2d3a', alpha=0.5)
            ax.set_ylabel('USD', color='#a0a0b0')
            st.pyplot(fig)

    # Top categories
    with col2:
        st.subheader("Top Job Categories")
        if 'job_category' in filtered.columns:
            top = filtered['job_category'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
            ax.set_facecolor('#1a1d27')
            ax.barh(top.index[::-1], top.values[::-1], color='#00d4aa', edgecolor='none')
            ax.tick_params(colors='#a0a0b0'); ax.grid(axis='x', color='#2a2d3a', alpha=0.5)
            ax.set_xlabel('Count', color='#a0a0b0')
            st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Salary histogram
    with col3:
        st.subheader("Salary Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        ax.hist(filtered['salary_in_usd'], bins=40, color='#00d4aa', edgecolor='none', alpha=0.85)
        ax.axvline(filtered['salary_in_usd'].median(), color='#ff6b9d', lw=2, ls='--',
                   label=f"Median ${filtered['salary_in_usd'].median():,.0f}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
        ax.tick_params(colors='#a0a0b0'); ax.legend(framealpha=0, labelcolor='#e0e0e0')
        ax.grid(axis='y', color='#2a2d3a', alpha=0.5)
        st.pyplot(fig)

    # Year trend
    with col4:
        st.subheader("Jobs Over Time")
        yc = filtered['work_year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1d27')
        ax.set_facecolor('#1a1d27')
        ax.fill_between(yc.index, yc.values, color='#00d4aa', alpha=0.3)
        ax.plot(yc.index, yc.values, color='#00d4aa', lw=2.5, marker='o', ms=7)
        ax.tick_params(colors='#a0a0b0'); ax.set_xticks(yc.index)
        ax.grid(axis='y', color='#2a2d3a', alpha=0.5)
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("💰 Salary Predictor")
    st.markdown("Fill in your profile and get an AI-powered salary estimate with explainability.")

    if model is None:
        st.warning("⚠️ Model not found. Run `step4_salary_model.py` first to generate `salary_model.pkl`.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            exp_input  = st.selectbox("Experience Level", ['Entry', 'Mid', 'Senior', 'Executive'])
            emp_input  = st.selectbox("Employment Type",  ['Full-Time', 'Part-Time', 'Contract', 'Freelance'])
            size_input = st.selectbox("Company Size",      ['Small', 'Medium', 'Large'])

        with col_b:
            cats = df['job_category'].dropna().unique().tolist() if 'job_category' in df.columns else ['Data Science']
            cat_input    = st.selectbox("Job Category", sorted(cats))
            remote_input = st.slider("Remote %", 0, 100, 50, step=50)
            year_input   = st.selectbox("Year", [2023, 2024, 2025])

        if st.button("🔮 Predict Salary", use_container_width=True):
            row = {
                'experience_level': exp_input,
                'employment_type':  emp_input,
                'company_size':     size_input,
                'job_category':     cat_input,
                'work_year':        year_input,
                'remote_ratio':     remote_input,
                'skill_count':      3,
                'is_germany':       0,
                'employee_residence': 'US',
                'company_location': 'US',
            }
            input_df = pd.DataFrame([row])

            for col in fc['cat_features']:
                if col in input_df.columns and col in le_dict:
                    le = le_dict[col]
                    val = input_df[col].iloc[0]
                    if val in le.classes_:
                        input_df[col + '_enc'] = le.transform([val])
                    else:
                        input_df[col + '_enc'] = 0
                else:
                    col_enc = col + '_enc'
                    input_df[col_enc] = 0

            X_input = input_df[[c for c in fc['features'] if c in input_df.columns]].fillna(0)
            for missing_col in fc['features']:
                if missing_col not in X_input.columns:
                    X_input[missing_col] = 0
            X_input = X_input[fc['features']]

            pred = model.predict(X_input)[0]
            low  = pred * 0.88
            high = pred * 1.12

            st.markdown(f"""
            <div class="prediction-box">
                <p style="color:#a0a0b0; margin:0">Estimated Annual Salary</p>
                <h2>${pred:,.0f}</h2>
                <p style="color:#a0a0b0; margin:0">Confidence range: ${low:,.0f} – ${high:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            # SHAP for single prediction
            try:
                import shap
                explainer   = shap.TreeExplainer(model)
                shap_vals   = explainer.shap_values(X_input)[0]
                feat_names  = fc['features']
                sorted_idx  = np.argsort(np.abs(shap_vals))[::-1][:8]

                st.markdown("#### 🔍 What drives this prediction?")
                for idx in sorted_idx:
                    val   = shap_vals[idx]
                    fname = feat_names[idx].replace('_enc', '').replace('_', ' ').title()
                    arrow = "🔺" if val > 0 else "🔻"
                    color = "#00d4aa" if val > 0 else "#ff6b9d"
                    st.markdown(
                        f'<span style="color:{color}">{arrow} **{fname}** '
                        f'{"+" if val>0 else ""}{val:,.0f} USD impact</span>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.info("Run the full pipeline to enable SHAP explanations.")

# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🇩🇪 Germany Tech Job Market")

    if 'is_germany' not in df.columns or df['is_germany'].sum() == 0:
        st.info("No Germany-specific rows found. Check `company_location == 'DE'` in your dataset.")
    else:
        de = df[df['is_germany'] == True]
        gl = df[df['is_germany'] == False]

        m1, m2, m3 = st.columns(3)
        m1.metric("Germany Jobs",    f"{len(de):,}")
        m2.metric("Germany Median",  f"${de['salary_in_usd'].median():,.0f}")
        m3.metric("Global Median",   f"${gl['salary_in_usd'].median():,.0f}")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### Salary: Germany vs Global")
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
            ax.set_facecolor('#1a1d27')
            bp = ax.boxplot([gl['salary_in_usd'], de['salary_in_usd']],
                            labels=['🌍 Global', '🇩🇪 Germany'],
                            patch_artist=True,
                            medianprops=dict(color='white', lw=2),
                            whiskerprops=dict(color='#a0a0b0'),
                            capprops=dict(color='#a0a0b0'))
            bp['boxes'][0].set_facecolor('#00d4aa'); bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('#ffa94d'); bp['boxes'][1].set_alpha(0.7)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
            ax.tick_params(colors='#a0a0b0'); ax.grid(axis='y', color='#2a2d3a', alpha=0.5)
            st.pyplot(fig)

        with col_g2:
            st.markdown("#### Top Roles in Germany")
            if 'job_category' in de.columns:
                de_cats = de['job_category'].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1d27')
                ax.set_facecolor('#1a1d27')
                ax.barh(de_cats.index[::-1], de_cats.values[::-1], color='#ffa94d', edgecolor='none')
                ax.tick_params(colors='#a0a0b0'); ax.grid(axis='x', color='#2a2d3a', alpha=0.5)
                st.pyplot(fig)

        st.markdown("#### 📝 Germany Entry Tips")
        tips = [
            "🎓 **Master's required** for most Data/AI roles at top German companies (Siemens, SAP, BMW, Bosch)",
            "🗣️ **German language** (B2+) significantly increases opportunities, especially for on-site roles",
            "🔍 **Top cities** for tech: Munich, Berlin, Hamburg, Stuttgart, Frankfurt",
            "📄 **Bewerbungsmappe**: German CVs are structured differently — no photo required since 2006 but still common",
            "🏥 **Benefits**: Public healthcare, 30 days vacation, and strong worker protections make even mid salaries very comfortable",
            "🌐 **Job portals**: StepStone.de, XING, LinkedIn.de, Make-it-in-Germany (official government portal)",
        ]
        for tip in tips:
            st.markdown(tip)
