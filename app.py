import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Identifying Bias in AI",
    page_icon="⚖️",
    layout="wide"
)

# ==========================================
# 2. PLOTLY THEME
# ==========================================

px.defaults.template = "plotly_dark"

px.defaults.color_discrete_sequence = [
    "#A970FF",
    "#8B5CF6",
    "#C084FC",
    "#E9D5FF"
]

px.defaults.color_continuous_scale = [
    "#211832",
    "#6336A0",
    "#985BEB",
    "#C084FC",
    "#F0E5FF"
]


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #080711;
        color: #F8F6FF;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem 3rem 3rem;
        max-width: 1500px;
    }
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    p, label, li {
        color: #D5CDE8;
    }
    [data-testid="stSidebar"] {
        background-color: #110D1D;
        border-right: 1px solid #38254F;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] label {
        color: #D9C9F5 !important;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #211832, #151020);
        border: 1px solid #59347E;
        border-radius: 15px;
        padding: 22px;
        box-shadow: 0 0 18px rgba(155,92,255,0.08);
    }
    [data-testid="stMetricLabel"] {
        color: #C7B5E7 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 750 !important;
    }
    .stButton > button {
        background-color: #8E4DDA;
        color: #FFFFFF !important;
        border: 1px solid #B17AFF;
        border-radius: 9px;
    }
    .stButton > button:hover {
        background-color: #A970FF;
        border-color: #D0ADFF;
    }
    [data-testid="stPlotlyChart"] {
        background-color: #151020;
        border: 1px solid #38254F;
        border-radius: 14px;
        padding: 10px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #59347E;
        border-radius: 12px;
    }
    hr {
        border-color: #38254F !important;
    }
    [data-testid="stHeader"] {
        background-color: #080711;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 4. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    data = pd.read_csv("data/bias_dataset.csv")
    return data

df = load_data()

required_columns = ["target", "rating"]
missing_required = [col for col in required_columns if col not in df.columns]

if missing_required:
    st.error(f"Required dataset columns are missing: {missing_required}")
    st.stop()

numeric_columns = [
    "target", "severe_toxicity", "obscene", "identity_attack", 
    "insult", "threat", "likes", "funny", "wow", "sad", 
    "disagree", "male", "female", "transgender", "muslim", 
    "christian", "jewish", "hindu", "black", "white", "asian"
]

for column in numeric_columns:
    if column not in df.columns:
        df[column] = 0
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)


# ==========================================
# 5. HERO HEADER (Fixed using native layout)
# ==========================================

with st.container():
    st.markdown("##### **ARTIFICIAL INTELLIGENCE / DATA ANALYTICS**")
    st.title("Identifying Bias in AI")
    st.markdown("Exploring Toxicity Patterns and Identity-Related Annotations")
    st.caption("JIGSAW TOXICITY DATASET · EXPLORATORY DATA ANALYSIS")
    st.markdown("---")


# ==========================================
# 6. SIDEBAR FILTERS
# ==========================================

st.sidebar.title("⚙️ Dashboard Filters")
st.sidebar.markdown("Customize your dataset analysis below.")

min_toxicity = st.sidebar.slider(
    "Minimum Toxicity", min_value=0.0, max_value=1.0, value=0.0, step=0.05
)

rating_options = ["All"] + sorted(df["rating"].dropna().unique().tolist(), key=str)
selected_rating = st.sidebar.selectbox("Select Moderation Rating", rating_options)


# ==========================================
# 7. APPLY FILTERS
# ==========================================

filtered_df = df[df["target"] >= min_toxicity].copy()

if selected_rating != "All":
    filtered_df = filtered_df[filtered_df["rating"] == selected_rating].copy()

if filtered_df.empty:
    st.warning("No records match your selected filters. Please adjust the sidebar filters.")
    st.stop()


# ==========================================
# 8. KEY PERFORMANCE INDICATORS
# ==========================================

st.subheader("Key Performance Indicators")

total_comments = len(filtered_df)
average_toxicity = filtered_df["target"].mean()
toxic_comments = (filtered_df["target"] >= 0.5).sum()
average_likes = filtered_df["likes"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Comments", f"{total_comments:,}")
col2.metric("Average Toxicity", f"{average_toxicity:.3f}")
col3.metric("Toxic Comments", f"{toxic_comments:,}")
col4.metric("Average Likes", f"{average_likes:.2f}")


# ==========================================
# 9. GRAPH STYLING HELPER
# ==========================================

def style_chart(fig, height=400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151020",
        plot_bgcolor="#151020",
        font=dict(color="#F0E9FF", family="Segoe UI"),
        title_font=dict(color="#FFFFFF", size=18),
        height=height,
        margin=dict(l=35, r=25, t=65, b=45),
        coloraxis_colorbar=dict(tickfont=dict(color="#F0E9FF"))
    )
    fig.update_xaxes(showgrid=True, gridcolor="#30233F", zerolinecolor="#49315F")
    fig.update_yaxes(showgrid=True, gridcolor="#30233F", zerolinecolor="#49315F")
    return fig


# ==========================================
# 10. UNIVARIATE ANALYSIS
# ==========================================

st.divider()
st.header("01 | Univariate Analysis")
st.markdown("Understanding the distribution of toxicity scores and category-level patterns.")

chart1, chart2 = st.columns(2)

with chart1:
    fig1 = px.histogram(
        filtered_df, x="target", nbins=30,
        title="Toxicity Score Distribution",
        color_discrete_sequence=["#A970FF"]
    )
    fig1.update_layout(xaxis_title="Toxicity Score", yaxis_title="Number of Comments")
    st.plotly_chart(style_chart(fig1), use_container_width=True)

with chart2:
    toxicity_columns = ["target", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]
    toxicity_means = filtered_df[toxicity_columns].mean().reset_index()
    toxicity_means.columns = ["Category", "Average Score"]

    fig2 = px.bar(
        toxicity_means, x="Category", y="Average Score", color="Average Score",
        title="Average Toxicity by Category",
        color_continuous_scale=["#211832", "#6336A0", "#985BEB", "#C084FC"]
    )
    fig2.update_layout(xaxis_title="Toxicity Category", yaxis_title="Average Score")
    st.plotly_chart(style_chart(fig2), use_container_width=True)


# ==========================================
# 11. IDENTITY-BASED ANALYSIS
# ==========================================

st.divider()
st.header("02 | Identity-Based Analysis")

identity_columns = [
    "male", "female", "transgender", "muslim", "christian", 
    "jewish", "hindu", "black", "white", "asian"
]

identity_results = []
for column in identity_columns:
    selected = filtered_df[filtered_df[column] > 0]
    identity_results.append({
        "Identity": column,
        "Comment Count": len(selected),
        "Average Toxicity": selected["target"].mean() if len(selected) > 0 else 0
    })

identity_df = pd.DataFrame(identity_results)
identity_chart1, identity_chart2 = st.columns(2)

with identity_chart1:
    fig3 = px.bar(
        identity_df, x="Identity", y="Average Toxicity", color="Average Toxicity",
        hover_data=["Comment Count"], title="Average Toxicity Across Identity Categories",
        color_continuous_scale=["#211832", "#6336A0", "#985BEB", "#C084FC"]
    )
    fig3.update_layout(xaxis_title="Identity Category", yaxis_title="Average Toxicity Score")
    st.plotly_chart(style_chart(fig3), use_container_width=True)

with identity_chart2:
    identity_counts = (filtered_df[identity_columns] > 0).sum().sort_values(ascending=False)
    identity_frequency = identity_counts.reset_index()
    identity_frequency.columns = ["Identity", "Comment Count"]

    fig4 = px.bar(
        identity_frequency, x="Identity", y="Comment Count", color="Comment Count",
        title="Identity Annotation Frequency",
        color_continuous_scale=["#211832", "#6336A0", "#985BEB", "#C084FC"]
    )
    fig4.update_layout(xaxis_title="Identity Category", yaxis_title="Number of Comments")
    st.plotly_chart(style_chart(fig4), use_container_width=True)


# ==========================================
# 12. CORRELATION ANALYSIS
# ==========================================

st.divider()
st.header("03 | Correlation Analysis")

correlation_columns = [
    "target", "severe_toxicity", "obscene", "identity_attack", 
    "insult", "threat", "likes", "funny", "wow", "sad", "disagree"
]

correlation_matrix = filtered_df[correlation_columns].corr()

fig5 = px.imshow(
    correlation_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
    aspect="auto", title="Correlation Between Dataset Variables", zmin=-1, zmax=1
)
st.plotly_chart(style_chart(fig5, height=550), use_container_width=True)


# ==========================================
# 13. SCATTER PLOT
# ==========================================

st.subheader("Likes vs Toxicity Analysis")
scatter_data = filtered_df.sample(n=min(5000, len(filtered_df)), random_state=42)

fig6 = px.scatter(
    scatter_data, x="likes", y="target", opacity=0.5,
    title="Relationship Between Likes and Toxicity",
    color_discrete_sequence=["#A970FF"]
)
fig6.update_layout(xaxis_title="Number of Likes", yaxis_title="Toxicity Score")
st.plotly_chart(style_chart(fig6), use_container_width=True)


# ==========================================
# 14. RAW DATA VIEWER
# ==========================================

st.divider()
st.header("04 | Explore Dataset")

if st.checkbox("Show Raw Dataset"):
    st.dataframe(filtered_df, use_container_width=True)


# ==========================================
# 15. DATA-DRIVEN KEY FINDINGS
# ==========================================

st.divider()
st.header("05 | Key Findings & Insights")
st.markdown("Insights calculated from the currently filtered dataset.")

top_category = toxicity_means.sort_values("Average Score", ascending=False).iloc[0]
top_identity = identity_frequency.sort_values("Comment Count", ascending=False).iloc[0]
moderation_means = filtered_df.groupby("rating")["target"].mean().sort_values(ascending=False)
likes_correlation = filtered_df[["likes", "target"]].corr().iloc[0, 1]

finding1, finding2 = st.columns(2)

with finding1:
    with st.container(border=True):
        st.caption("FINDING 01")
        st.subheader("Toxicity Category")
        st.markdown(f"### **{top_category['Category']}**")
        st.write(f"Highest average score: {top_category['Average Score']:.3f}")

with finding2:
    with st.container(border=True):
        st.caption("FINDING 02")
        st.subheader("Identity Annotation")
        st.markdown(f"### **{top_identity['Identity'].title()}**")
        st.write(f"Annotated comments: {int(top_identity['Comment Count']):,}")

finding3, finding4 = st.columns(2)

with finding3:
    with st.container(border=True):
        st.caption("FINDING 03")
        st.subheader("Moderation Analysis")
        st.markdown(f"### **{moderation_means.index[0]}**")
        st.write(f"Highest average toxicity: {moderation_means.iloc[0]:.3f}")

with finding4:
    with st.container(border=True):
        st.caption("FINDING 04")
        st.subheader("Engagement Analysis")
        st.markdown(f"### **{likes_correlation:.3f}**")
        st.write("Correlation between likes and toxicity.")

st.info("These are descriptive patterns in the dataset. They do not independently prove discrimination or AI bias.")


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()
st.markdown("<p align='center' style='color:#A970FF; letter-spacing:2px;'>IDENTIFYING BIAS IN AI</p>", unsafe_allow_html=True)
st.markdown("<p align='center' style='color:#8F83A8; font-size:12px;'>Exploratory Data Analysis | AI & Data Science Project</p>", unsafe_allow_html=True)