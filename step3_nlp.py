# ============================================================
# STEP 3: NLP — Skill Extraction from Job Titles & Categories
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3d4a', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0b0', 'ytick.color': '#a0a0b0',
    'text.color': '#e0e0e0', 'grid.color': '#2a2d3a', 'grid.alpha': 0.5,
})
ACCENT, ACCENT2, ACCENT3 = '#00d4aa', '#ff6b9d', '#ffa94d'

df = pd.read_csv('jobs_cleaned.csv')

# ── A. Define Tech Skill Keywords ────────────────────────────
SKILL_KEYWORDS = {
    'Python':       ['python'],
    'SQL':          ['sql', 'database', 'postgres', 'mysql'],
    'Machine Learning': ['machine learning', 'ml', 'classification', 'regression'],
    'Deep Learning':['deep learning', 'neural', 'cnn', 'rnn', 'transformer'],
    'NLP':          ['nlp', 'natural language', 'text', 'bert', 'llm'],
    'Cloud (AWS/GCP/Azure)': ['aws', 'gcp', 'azure', 'cloud'],
    'Data Engineering': ['data engineer', 'pipeline', 'spark', 'kafka', 'airflow'],
    'Statistics':   ['statistic', 'probability', 'bayesian', 'hypothesis'],
    'Visualization':['visualization', 'tableau', 'power bi', 'dashboard'],
    'Computer Vision': ['computer vision', 'image', 'opencv', 'detection'],
    'MLOps':        ['mlops', 'deployment', 'docker', 'kubernetes', 'ci/cd'],
    'Research':     ['research', 'phd', 'publication', 'academic'],
}

def extract_skills(text):
    """Return list of matched skill names from text."""
    text = str(text).lower()
    found = []
    for skill, kws in SKILL_KEYWORDS.items():
        if any(kw in text for kw in kws):
            found.append(skill)
    return found

# Combine title + category for richer signal
text_col = (
    df.get('job_title', pd.Series([''] * len(df))).fillna('').str.lower() + ' ' +
    df.get('job_category', pd.Series([''] * len(df))).fillna('').str.lower()
)
df['skills_found'] = text_col.apply(extract_skills)
df['skill_count']  = df['skills_found'].apply(len)

# Flatten all skills
all_skills = [s for skills in df['skills_found'] for s in skills]
skill_counts = Counter(all_skills)

print("Top skills found:")
for skill, count in skill_counts.most_common(10):
    print(f"  {skill:<30} {count:>5} mentions")

# ── B. TF-IDF on job titles ──────────────────────────────────
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                        stop_words='english', min_df=3)
tfidf_matrix = tfidf.fit_transform(
    df.get('job_title', pd.Series(['data scientist'] * len(df))).fillna('')
)
feature_names = tfidf.get_feature_names_out()
mean_tfidf    = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top_idx       = mean_tfidf.argsort()[-20:][::-1]
top_terms     = [(feature_names[i], mean_tfidf[i]) for i in top_idx]

# ── C. LDA Topic Modelling ───────────────────────────────────
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda.fit(tfidf_matrix)

topic_labels = ['Data Science', 'Engineering', 'Analytics', 'AI/Research', 'Management']
topic_words  = {}
for i, topic in enumerate(lda.components_):
    top_w = [feature_names[j] for j in topic.argsort()[:-8:-1]]
    topic_words[topic_labels[i]] = top_w
    print(f"\nTopic {i+1} [{topic_labels[i]}]: {', '.join(top_w)}")

# Save topic assignments
doc_topics   = lda.transform(tfidf_matrix)
df['topic']  = np.argmax(doc_topics, axis=1)
df['topic_label'] = df['topic'].map(dict(enumerate(topic_labels)))

# ── D. Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('NLP Analysis — Tech Job Titles', fontsize=18, fontweight='bold', color='white')
fig.patch.set_facecolor('#0f1117')
PALETTE = [ACCENT, ACCENT2, ACCENT3, '#74c7ec', '#cba6f7', '#89dceb',
           '#f38ba8', '#a6e3a1', '#fab387', '#94e2d5', '#b4befe', '#eba0ac']

# Skill frequency bar
ax = axes[0]
skills_sorted = skill_counts.most_common(12)
names, vals   = zip(*skills_sorted)
colors = PALETTE[:len(names)]
ax.barh(list(names)[::-1], list(vals)[::-1], color=colors[::-1])
ax.set_title('Most In-Demand Skills\n(keyword match)', fontweight='bold')
ax.set_xlabel('Mentions')
ax.grid(axis='x')

# Top TF-IDF terms
ax = axes[1]
terms, scores = zip(*top_terms[:12])
ax.barh(list(terms)[::-1], list(scores)[::-1], color=ACCENT3)
ax.set_title('Top TF-IDF Terms\n(job titles)', fontweight='bold')
ax.set_xlabel('Mean TF-IDF Score')
ax.grid(axis='x')

# LDA Topic distribution
ax = axes[2]
topic_dist = df['topic_label'].value_counts()
ax.pie(topic_dist.values, labels=topic_dist.index,
       autopct='%1.1f%%', colors=PALETTE[:len(topic_dist)],
       wedgeprops=dict(edgecolor='#0f1117', linewidth=2),
       startangle=120)
ax.set_title('Job Topic Distribution\n(LDA)', fontweight='bold')

plt.tight_layout()
plt.savefig('nlp_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("\n✅ Saved → nlp_analysis.png")

df.to_csv('jobs_nlp.csv', index=False)
print("✅ Saved → jobs_nlp.csv")
