import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import chi2_contingency

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Big Data Beauty Dashboard",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

/* Root variables */
:root {
    --watsons: #00a8a0;
    --sociolla: #e8849a;
    --accent: #7c5cbf;
    --bg: #f8f7fc;
    --card: #ffffff;
    --text: #1a1a2e;
    --muted: #6b7280;
    --border: #e5e7eb;
}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit default header */
#MainMenu, footer {visibility: hidden;}
header {visibility: visible;}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08) !important;
}

/* Main content background */
.main .block-container {
    background: #f8f7fc;
    padding-top: 2rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1.2rem;
    border-left: 4px solid var(--accent);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #1a1a2e;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 0.3rem;
}

/* Tampilkan tombol toggle sidebar */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: block !important;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #f0f4ff 0%, #fdf2f8 100%);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #e0d7f5;
    margin-top: 1.2rem;
}
.insight-box h3 {
    font-size: 1rem;
    font-weight: 700;
    color: #4c1d95;
    margin-bottom: 0.8rem;
}
.insight-box li {
    color: #374151;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

/* Page title */
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1a1a2e;
    margin-bottom: 0.3rem;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 1.8rem;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 2px solid #e5e7eb;
    margin: 1.5rem 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
}

/* Dataframe */
.dataframe { font-size: 0.85rem; }

/* Platform badge */
.badge-watsons {
    background: #e6f7f6; color: #00a8a0;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    display: inline-block; margin-bottom: 0.5rem;
}
.badge-sociolla {
    background: #fce7ef; color: #e8849a;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    display: inline-block; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────
C_W   = '#00a8a0'   # Watsons teal
C_S   = '#e8849a'   # Sociolla pink
C_ACC = '#7c5cbf'   # purple accent
C_DRK = '#1a1a2e'   # dark navy
C_MUT = '#9ca3af'   # muted gray

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    url_w = "https://raw.githubusercontent.com/naiarch/UTS-Big-Data-Web-Scraping-Lazada/main/product_watsons_clean.csv"
    url_s = "https://raw.githubusercontent.com/naiarch/UTS-Big-Data-Web-Scraping-Lazada/main/products_sociolla_clean.csv"
    df_w = pd.read_csv(url_w)
    df_s = pd.read_csv(url_s)
    df_r = pd.read_csv("Responden Terabyte - Form Responses 1.csv")
    df_r.columns = [
        'timestamp', 'nama', 'gender', 'usia', 'spending', 'toko_digunakan',
        'w_kelengkapan', 'w_miss', 'w_review', 'w_diskon', 'w_rating', 'col11',
        'w_terjual', 'w_kategori',
        's_kelengkapan', 's_miss', 's_review', 's_diskon', 's_rating', 's_terjual',
        's_kategori', 'rekomendasi', 'score', 'email'
    ]
    df_merged = pd.concat([df_w, df_s], ignore_index=True)
    return df_w, df_s, df_r, df_merged

df_w, df_s, df_r, df_merged = load_data()

# Pre-compute shared stats
avg_harga_w   = df_w['harga'].mean()
avg_harga_s   = df_s['harga_original'].mean()
avg_rating_w  = df_w['rating'].mean()
avg_rating_s  = df_s['rating'].mean()
avg_review_w  = df_w['jumlah_review'].mean()
avg_review_s  = df_s['jumlah_review'].mean()
avg_terjual_s = df_s['jumlah_terjual'].mean()
std_w = df_w['harga'].std()
std_s = df_s['harga_original'].std()
cv_w  = std_w / avg_harga_w * 100
cv_s  = std_s / avg_harga_s * 100

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.2rem;'>💄</div>
        <div style='font-family: DM Serif Display, serif; font-size:1.15rem; font-weight:700; margin-top:0.4rem;'>Beauty Analytics</div>
        <div style='font-size:0.75rem; color:#94a3b8; margin-top:0.2rem;'>Big Data Dashboard · UAS</div>
    </div>
    <hr style='border-color: rgba(255,255,255,0.1); margin-bottom:1rem;'>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Home", "🟢  Watsons", "🩷  Sociolla", "📊  Insight 8V", "📋  Analisis Kuisioner"],
        label_visibility="collapsed"
    )
    menu = menu.split("  ", 1)[-1]   # strip emoji prefix

    st.markdown("""
    <hr style='border-color: rgba(255,255,255,0.1); margin-top:2rem;'>
    <div style='font-size:0.72rem; color:#64748b; text-align:center; padding-top:0.5rem;'>
        Sumber data: Watsons · Sociolla<br>Survey Responden Terabyte
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def insight_box(content_md: str):
    st.markdown(f'<div class="insight-box">{content_md}</div>', unsafe_allow_html=True)

def metric_card(label, value, sub="", border_color=C_ACC):
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{border_color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def styled_fig(figsize=(12,5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    return fig, ax

def bar_labels(ax, bars, fmt='{:,.0f}', color=C_DRK, offset=0.02):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                fmt.format(h), ha='center', va='bottom', fontsize=10, fontweight='600', color=color)

# ─────────────────────────────────────────
# HOME
# ─────────────────────────────────────────
if menu == "Home":
    st.markdown('<div class="page-title">Dashboard Big Data Beauty Products</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analisis komparatif Watsons & Sociolla · Konsep 8V Big Data · Insight Kuisioner Konsumen</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Produk Watsons", f"{len(df_w):,}", "dari web scraping Lazada", C_W)
    with c2: metric_card("Produk Sociolla", f"{len(df_s):,}", "dari web scraping Sociolla", C_S)
    with c3: metric_card("Total Produk", f"{len(df_merged):,}", "dataset gabungan", C_ACC)
    with c4: metric_card("Responden Survey", f"{len(df_r):,}", "konsumen beauty", "#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### 🎯 Tujuan Analisis")
        goals = [
        "Analisis mendalam dataset Watsons & Sociolla",
        "Insight berdasarkan kerangka 8V Big Data",
        "Analisis perilaku konsumen dari kuesioner",
        "Perbandingan komparatif antar platform",
        "Rekomendasi strategis berbasis data",
        ]
        for text in goals:
            st.markdown(f"""
            <div style='padding:0.6rem 0; border-bottom:1px solid #f0f0f0;'>
                <span style='font-size:0.9rem; color:#374151;'>- {text}</span>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("#### 📈 Ringkasan Platform")
        comparisons = [
            ("Rata-rata Harga", f"Rp{avg_harga_w:,.0f}", f"Rp{avg_harga_s:,.0f}"),
            ("Rata-rata Rating", f"⭐ {avg_rating_w:.2f}", f"⭐ {avg_rating_s:.2f}"),
            ("Avg Review/Produk", f"{avg_review_w:.0f}", f"{avg_review_s:.0f}"),
        ]
        header = """
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; font-size:0.78rem;
                    font-weight:700; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em;
                    padding:0.5rem 0; border-bottom:2px solid #e5e7eb; margin-bottom:0.5rem;'>
            <span>Metrik</span>
            <span style='color:#00a8a0; text-align:center;'>Watsons</span>
            <span style='color:#e8849a; text-align:center;'>Sociolla</span>
        </div>"""
        rows = ""
        for m, w, s in comparisons:
            rows += f"""
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem;
                        padding:0.6rem 0; border-bottom:1px solid #f3f4f6; font-size:0.88rem;'>
                <span style='color:#374151; font-weight:500;'>{m}</span>
                <span style='color:#00a8a0; font-weight:700; text-align:center;'>{w}</span>
                <span style='color:#e8849a; font-weight:700; text-align:center;'>{s}</span>
            </div>"""
        st.markdown(f'<div style="background:white; border-radius:14px; padding:1.2rem 1.4rem; box-shadow:0 2px 12px rgba(0,0,0,0.06);">{header}{rows}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# WATSONS
# ─────────────────────────────────────────
elif menu == "Watsons":
    st.markdown('<span class="badge-watsons">🟢 WATSONS</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analisis Dataset Watsons</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset", "Distribusi", "Top Produk", "Korelasi", "Insight"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Jumlah Produk", f"{len(df_w):,}", "", C_W)
        with c2: metric_card("Jumlah Kolom", str(df_w.shape[1]), "", C_W)
        with c3: metric_card("Missing Values", f"{df_w.isnull().sum().sum():,}", "total sel kosong", C_W)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Preview Data (5 baris pertama)**")
        st.dataframe(df_w.head(), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Tipe Data**")
            st.dataframe(df_w.dtypes.reset_index().rename(columns={'index':'Kolom', 0:'Tipe'}), use_container_width=True)
        with col_r:
            st.markdown("**Statistik Deskriptif**")
            st.dataframe(df_w.describe().round(2), use_container_width=True)

        with tab2:
            red_pop = '#FF4D4D'
            blue_pop = '#3399FF'
            plt.style.use('default')

            # Row 1: Harga & Rating
            col_l, col_r = st.columns(2)

            with col_l:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df_w['harga'].dropna(), bins=50, color=C_W, kde=True,
                             edgecolor='black', ax=ax)
                if ax.lines:
                    ax.lines[0].set_color(red_pop)
                    ax.lines[0].set_linewidth(3)
                ax.set_xlim(0, df_w['harga'].quantile(0.99))
                ax.set_title('Distribusi Harga Produk', fontweight='bold', fontsize=13)
                ax.set_xlabel('Harga (Rp)', fontsize=11)
                ax.set_ylabel('Frekuensi', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)

            with col_r:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df_w['rating'].dropna(), bins=20, color=C_W, kde=True,
                             edgecolor='black', ax=ax)
                if ax.lines:
                    ax.lines[0].set_color(red_pop)
                    ax.lines[0].set_linewidth(3)
                ax.set_title('Distribusi Rating Produk', fontweight='bold', fontsize=13)
                ax.set_xlabel('Rating', fontsize=11)
                ax.set_ylabel('Frekuensi', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)

            # Row 2: Diskon (centered)
            col_l2, col_r2 = st.columns(2)

            with col_l2:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df_w['diskon'].dropna(), bins=30, color=C_W, kde=True,
                             edgecolor='black', ax=ax)
                if ax.lines:
                    ax.lines[0].set_color(red_pop)
                    ax.lines[0].set_linewidth(3)
                ax.set_title('Distribusi Diskon Produk', fontweight='bold', fontsize=13)
                ax.set_xlabel('Diskon (%)', fontsize=11)
                ax.set_ylabel('Frekuensi', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)

            with col_r2:
                st.markdown("### Statistik Ringkas")
                st.metric("Rata-rata Rating", f"{df_w['rating'].mean():.2f}")
                st.metric("Median Rating", f"{df_w['rating'].median():.2f}")
                st.metric("Rata-rata Diskon", f"{df_w['diskon'].mean():.1f}%")
                st.metric("Produk Tanpa Diskon", f"{df_w['diskon'].isna().sum():,}")

        with tab3:
            st.markdown("### 10 Produk dengan Review Terbanyak")
            top10 = (df_w.groupby('nama_produk')['jumlah_review']
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index())

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top10['nama_produk'], top10['jumlah_review'],
                        color=C_W, edgecolor='black', alpha=0.85)
            ax.bar_label(bars, fmt='{:,.0f}', padding=4, fontsize=9)
            ax.set_xlabel('Jumlah Review', fontsize=11)
            ax.set_ylabel('')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            st.markdown("### Korelasi Antar Variabel")

            # Row 1: Heatmap korelasi
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("#### Korelasi Antar Variabel Numerik")
                num_cols = df_w[['harga', 'rating', 'diskon', 'jumlah_review']].dropna()
                corr_matrix = num_cols.corr()

                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                            center=0, linewidths=0.5, ax=ax,
                            annot_kws={'fontsize': 11, 'fontweight': 'bold'})
                ax.set_title('Heatmap Korelasi', fontweight='bold', fontsize=13)
                plt.tight_layout()
                st.pyplot(fig)

            with col_r:
                st.markdown("#### Korelasi Diskon vs Harga")
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=df_w.dropna(subset=['diskon', 'harga']),
                                x='diskon', y='harga',
                                color=C_W, alpha=0.5, edgecolor='black', linewidth=0.3, ax=ax)
                sns.regplot(data=df_w.dropna(subset=['diskon', 'harga']),
                            x='diskon', y='harga',
                            scatter=False, color=red_pop, line_kws={'linewidth': 2}, ax=ax)
                ax.set_title('Diskon vs Harga', fontweight='bold', fontsize=13)
                ax.set_xlabel('Diskon (%)', fontsize=11)
                ax.set_ylabel('Harga (Rp)', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)

            # Row 2: Harga vs Rating (center)
            col_l2, col_r2 = st.columns(2)

            with col_l2:
                st.markdown("#### Korelasi Harga vs Rating")
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=df_w.dropna(subset=['harga', 'rating']),
                                x='harga', y='rating',
                                color=C_W, alpha=0.5, edgecolor='black', linewidth=0.3, ax=ax)
                sns.regplot(data=df_w.dropna(subset=['harga', 'rating']),
                            x='harga', y='rating',
                            scatter=False, color=red_pop, line_kws={'linewidth': 2}, ax=ax)
                ax.set_xlim(0, df_w['harga'].quantile(0.99))
                ax.set_title('Harga vs Rating', fontweight='bold', fontsize=13)
                ax.set_xlabel('Harga (Rp)', fontsize=11)
                ax.set_ylabel('Rating', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)

            with col_r2:
                st.markdown("#### Ringkasan Korelasi")
                corr_harga_rating = df_w[['harga', 'rating']].corr().iloc[0, 1]
                corr_diskon_harga = df_w[['diskon', 'harga']].corr().iloc[0, 1]
                corr_diskon_rating = df_w[['diskon', 'rating']].corr().iloc[0, 1]

                st.metric("Korelasi Harga & Rating",   f"{corr_harga_rating:.3f}")
                st.metric("Korelasi Diskon & Harga",   f"{corr_diskon_harga:.3f}")
                st.metric("Korelasi Diskon & Rating",  f"{corr_diskon_rating:.3f}")
                    
                    
        with tab5:
            insight_box("""
            <h3>💡 Insight Watsons</h3>
            <ul>
            <li>Sebagian besar produk memiliki <strong>rating tinggi</strong> (terkonsentrasi di atas 4.0).</li>
            <li>Distribusi harga bersifat <strong>right-skewed</strong> — mayoritas produk berada di segmen harga menengah ke bawah dengan sedikit produk premium.</li>
            <li>Produk dengan <strong>jumlah review tinggi</strong> umumnya memiliki rating yang baik, menunjukkan korelasi positif antara engagement dan kualitas produk.</li>
            </ul>
            """)

# ─────────────────────────────────────────
# SOCIOLLA
# ─────────────────────────────────────────
elif menu == "Sociolla":
    st.markdown('<span class="badge-sociolla">🩷 SOCIOLLA</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analisis Dataset Sociolla</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset", "Distribusi", "Top Produk", "Korelasi", "Insight"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Jumlah Produk", f"{len(df_s):,}", "", C_S)
        with c2: metric_card("Jumlah Kolom", str(df_s.shape[1]), "", C_S)
        with c3: metric_card("Missing Values", f"{df_s.isnull().sum().sum():,}", "total sel kosong", C_S)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Preview Data (5 baris pertama)**")
        st.dataframe(df_s.head(), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Tipe Data**")
            st.dataframe(df_s.dtypes.reset_index().rename(columns={'index':'Kolom', 0:'Tipe'}), use_container_width=True)
        with col_r:
            st.markdown("**Statistik Deskriptif**")
            st.dataframe(df_s.describe().round(2), use_container_width=True)

    with tab2:
        pink_pop = '#FF3385'
        green_pop = '#00FF66'
        red_pop = '#FF4D4D'
        plt.style.use('default')

        col_l, col_r = st.columns(2)

        with col_l:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(df_s['harga_original'], bins=50, color=C_S,
                        kde=True, edgecolor='black', ax=ax)
            if ax.lines:
                ax.lines[0].set_color(red_pop)
                ax.lines[0].set_linewidth(3)
            ax.set_xlim(0, 1000000)
            ax.set_title('Distribusi Harga Produk', fontweight='bold', fontsize=13)
            ax.set_xlabel('Harga Original (Rp)', fontsize=11)
            ax.set_ylabel('Frekuensi', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)

        with col_r:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(df_s['rating'].dropna(), bins=30, color=C_S,
                        kde=True, edgecolor='black', ax=ax)
            if ax.lines:
                ax.lines[0].set_color(red_pop)
                ax.lines[0].set_linewidth(3)
            ax.set_title('Distribusi Rating Produk', fontweight='bold', fontsize=13)
            ax.set_xlabel('Rating', fontsize=11)
            ax.set_ylabel('Frekuensi', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)

    with tab3:
        st.markdown("### 10 Produk Terlaris di Sociolla")
        top10_s = df_s.nlargest(10, 'jumlah_terjual').sort_values('jumlah_terjual', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top10_s['nama_produk'], top10_s['jumlah_terjual'],
                    color=C_S, edgecolor=C_S, linewidth=1.5)
        ax.bar_label(bars, fmt='{:,.0f}', padding=4, fontsize=9)
        ax.set_xlabel('Jumlah Terjual', fontsize=11)
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.markdown("### Korelasi Antar Variabel")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Korelasi Review vs Terjual")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(x='jumlah_review', y='jumlah_terjual', data=df_s,
                        scatter_kws={'color': C_S, 'alpha': 0.5, 'edgecolor': 'black', 's': 40},
                        line_kws={'color': red_pop, 'linewidth': 2}, ax=ax)
            ax.set_xlabel('Jumlah Review', fontsize=11)
            ax.set_ylabel('Jumlah Terjual', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

        with col_r:
            st.markdown("#### Korelasi Harga vs Terjual")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.scatterplot(x='harga_original', y='jumlah_terjual', data=df_s,
                            color=C_S, alpha=0.6, edgecolor='black', s=50, ax=ax)
            ax.set_xlim(0, 1500000)
            ax.set_ylim(0, 5000)
            ax.set_xlabel('Harga Original (Rp)', fontsize=11)
            ax.set_ylabel('Jumlah Terjual', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown("#### Ringkasan Korelasi")
            corr_review_terjual = df_s[['jumlah_review', 'jumlah_terjual']].corr().iloc[0, 1]
            corr_harga_terjual  = df_s[['harga_original', 'jumlah_terjual']].corr().iloc[0, 1]
            st.metric("Korelasi Review & Terjual", f"{corr_review_terjual:.3f}")
            st.metric("Korelasi Harga & Terjual",  f"{corr_harga_terjual:.3f}")

    with tab5:
        insight_box("""
        <h3>💡 Insight Sociolla</h3>
        <ul>
        <li>Sebagian besar produk memiliki <strong>rating tinggi</strong>, mencerminkan kualitas produk yang konsisten.</li>
        <li>Distribusi harga menunjukkan pola <strong>right-skewed</strong> dengan beberapa produk premium di ujung atas.</li>
        <li>Produk dengan jumlah review tinggi umumnya memiliki rating yang baik — menandakan <strong>engagement pengguna yang aktif</strong>.</li>
        </ul>
        """)

# ─────────────────────────────────────────
# INSIGHT 8V
# ─────────────────────────────────────────
elif menu == "Insight 8V":
    st.markdown('<div class="page-title">📊 Insight 8V Big Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analisis dataset beauty menggunakan kerangka Volume · Velocity · Variety · Veracity · Value · Variability · Visualization · Vulnerability</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Volume", "Velocity", "Variety", "Veracity",
                    "Value", "Variability", "Visualization", "Vulnerability"])

    # ── V1 VOLUME ──
    with tabs[0]:
        st.markdown("#### V1 — Volume: Jumlah Data per Platform")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Watsons", f"{len(df_w):,}", "produk", C_W)
        with c2: metric_card("Sociolla", f"{len(df_s):,}", "produk", C_S)
        with c3: metric_card("Gabungan", f"{len(df_merged):,}", "total produk", C_ACC)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(['Watsons', 'Sociolla', 'Gabungan'],
                      [len(df_w), len(df_s), len(df_merged)],
                      color=[C_W, C_S, C_ACC], width=0.45, edgecolor='white', linewidth=1.5)
        bar_labels(ax, bars, fmt='{:,.0f}', offset=15)
        ax.set_title('Jumlah Produk per Platform', fontsize=13, fontweight='700')
        ax.set_ylabel('Jumlah Produk'); ax.set_ylim(0, len(df_merged) * 1.25)
        plt.tight_layout(); st.pyplot(fig)

        porsi_w = len(df_w)/len(df_merged)*100
        porsi_s = len(df_s)/len(df_merged)*100
        insight_box(f"""
        <h3>📌 Temuan Volume</h3>
        <ul>
        <li>Total data yang dianalisis: <strong>{len(df_merged):,} produk</strong>.</li>
        <li>Watsons menyumbang <strong>{porsi_w:.1f}%</strong>, Sociolla <strong>{porsi_s:.1f}%</strong> dari keseluruhan data.</li>
        <li><strong>{'Watsons' if len(df_w)>len(df_s) else 'Sociolla'} memiliki katalog yang lebih besar</strong>, menawarkan pilihan produk yang lebih beragam.</li>
        </ul>""")

    # ── V2 VELOCITY ──
    with tabs[1]:
        st.markdown("#### V2 — Velocity: Dinamika Interaksi Pengguna")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Avg Review Watsons", f"{avg_review_w:.1f}", "per produk", C_W)
        with c2: metric_card("Avg Review Sociolla", f"{avg_review_s:.1f}", "per produk", C_S)
        with c3: metric_card("Avg Terjual Sociolla", f"{avg_terjual_s:,.0f}", "per produk", C_ACC)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(['Watsons', 'Sociolla'], [avg_review_w, avg_review_s],
                    color=[C_W, C_S], width=0.4, edgecolor='white')
        for i, v in enumerate([avg_review_w, avg_review_s]):
            axes[0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontweight='700', fontsize=12)
        axes[0].set_title('Rata-rata Review per Produk', fontsize=12, fontweight='700')
        axes[0].set_ylabel('Rata-rata Review')

        totals = [df_w['jumlah_review'].sum(), df_s['jumlah_review'].sum(), df_s['jumlah_terjual'].sum()]
        axes[1].bar(['Total Review\nWatsons', 'Total Review\nSociolla', 'Total Terjual\nSociolla'],
                    totals, color=[C_W, C_S, C_ACC], width=0.4, edgecolor='white')
        for i, v in enumerate(totals):
            axes[1].text(i, v + 500, f'{v:,.0f}', ha='center', fontweight='700', fontsize=9)
        axes[1].set_title('Total Aktivitas Platform', fontsize=12, fontweight='700')
        axes[1].set_ylabel('Total')
        plt.tight_layout(); st.pyplot(fig)

        winner = 'Sociolla' if avg_review_s > avg_review_w else 'Watsons'
        insight_box(f"""
        <h3>📌 Temuan Velocity</h3>
        <ul>
        <li><strong>{winner}</strong> menunjukkan engagement pengguna yang lebih tinggi berdasarkan aktivitas review.</li>
        <li>Informasi jumlah terjual hanya tersedia di Sociolla (rata-rata <strong>{avg_terjual_s:.0f}</strong> per produk).</li>
        <li>Platform dengan review rendah perlu program insentif untuk meningkatkan partisipasi pengguna.</li>
        </ul>""")

    # ── V3 VARIETY ──
    with tabs[2]:
        st.markdown("#### V3 — Variety: Kelengkapan Atribut Data")
        kelengkapan = pd.DataFrame({
            'Kolom': ['harga', 'harga_original', 'diskon', 'rating', 'jumlah_review', 'jumlah_terjual'],
            'Watsons': [df_w['harga'].notna().mean()*100, df_w['harga_original'].notna().mean()*100,
                        df_w['diskon'].notna().mean()*100, df_w['rating'].notna().mean()*100,
                        df_w['jumlah_review'].notna().mean()*100, 0],
            'Sociolla': [df_s['harga'].notna().mean()*100, df_s['harga_original'].notna().mean()*100,
                         0, df_s['rating'].notna().mean()*100, df_s['jumlah_review'].notna().mean()*100,
                         df_s['jumlah_terjual'].notna().mean()*100]
        })
        x = np.arange(len(kelengkapan))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x-0.2, kelengkapan['Watsons'], 0.35, label='Watsons', color=C_W, alpha=0.85, edgecolor='white')
        ax.bar(x+0.2, kelengkapan['Sociolla'], 0.35, label='Sociolla', color=C_S, alpha=0.85, edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(kelengkapan['Kolom'], rotation=15)
        ax.set_title('Kelengkapan Kolom per Platform (%)', fontsize=13, fontweight='700')
        ax.set_ylabel('% Data Terisi'); ax.set_ylim(0, 120)
        ax.axhline(100, color='gray', linestyle='--', alpha=0.4)
        ax.legend()
        plt.tight_layout(); st.pyplot(fig)

        insight_box("""
        <h3>📌 Temuan Variety</h3>
        <ul>
        <li><strong>Watsons</strong> unggul pada kolom <em>diskon</em> yang tidak tersedia di Sociolla.</li>
        <li><strong>Sociolla</strong> unggul pada kolom <em>jumlah terjual</em> yang tidak dimiliki Watsons.</li>
        <li>Perbedaan skema data menjadi tantangan dalam analisis komparatif — standardisasi atribut sangat diperlukan.</li>
        </ul>""")

    # ── V4 VERACITY ──
    with tabs[3]:
        st.markdown("#### V4 — Veracity: Analisis Outlier Harga")
        sub_w, sub_s = st.tabs(["🟢 Watsons", "🩷 Sociolla"])

        for sub, df_plot, harga_col, color, label in [
            (sub_w, df_w, 'harga', C_W, 'Watsons'),
            (sub_s, df_s, 'harga_original', C_S, 'Sociolla')
        ]:
            with sub:
                harga = df_plot[harga_col].dropna()
                q1, q3 = harga.quantile(0.25), harga.quantile(0.75)
                iqr = q3 - q1
                batas_atas = q3 + 1.5*iqr
                outlier_atas = df_plot[df_plot[harga_col] > batas_atas]
                normal = df_plot[(df_plot[harga_col] >= q1-1.5*iqr) & (df_plot[harga_col] <= batas_atas)]

                fig, axes = plt.subplots(1, 3, figsize=(17, 5))
                fig.suptitle(f'Analisis Outlier Harga — {label}', fontsize=13, fontweight='700')
                axes[0].boxplot(harga, patch_artist=True,
                                boxprops=dict(facecolor=color, alpha=0.6),
                                medianprops=dict(color=C_DRK, linewidth=2))
                axes[0].axhline(batas_atas, color='red', linestyle='--', linewidth=1.5, label='Batas atas')
                axes[0].set_title('Boxplot'); axes[0].legend(fontsize=8)

                axes[1].scatter(range(len(normal)), normal[harga_col], color=color, alpha=0.25, s=20, label='Normal')
                axes[1].scatter(range(len(outlier_atas)), outlier_atas[harga_col], color='#ef4444', s=30, label='Outlier')
                axes[1].set_title('Sebaran Harga'); axes[1].legend(fontsize=8)

                top10 = outlier_atas.nlargest(10, harga_col)
                axes[2].barh(top10['nama_produk'].str[:28], top10[harga_col], color=color, alpha=0.85)
                axes[2].set_title('Top 10 Produk Termahal')

                plt.tight_layout(); st.pyplot(fig)

    # ── V5 VALUE ──
    with tabs[4]:
        st.markdown("#### V5 — Value: Harga, Rating & Diskon")
        sub1, sub2, sub3 = st.tabs(["Harga & Rating", "Value for Money", "Pengaruh Diskon"])

        with sub1:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for ax, vals, title, ylabel in [
                (axes[0], [avg_harga_w, avg_harga_s], 'Rata-rata Harga Produk', 'Harga (Rp)'),
                (axes[1], [avg_rating_w, avg_rating_s], 'Rata-rata Rating', 'Rating')
            ]:
                bars = ax.bar(['Watsons','Sociolla'], vals, color=[C_W, C_S], width=0.4, edgecolor='white')
                ax.set_title(title, fontsize=12, fontweight='700'); ax.set_ylabel(ylabel)
                if 'Rating' in title: ax.set_ylim(0, 5.5)
                for b in bars:
                    h = b.get_height()
                    ax.text(b.get_x()+b.get_width()/2, h+0.01*h, f'{h:,.2f}', ha='center', fontsize=10, fontweight='600')
            plt.tight_layout(); st.pyplot(fig)

        with sub2:
            median_harga_w = df_w['harga'].median()
            median_harga_s = df_s['harga_original'].median()
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for ax, df_plot, harga_col, color, label, median_h in [
                (axes[0], df_w, 'harga', C_W, 'Watsons', median_harga_w),
                (axes[1], df_s, 'harga_original', C_S, 'Sociolla', median_harga_s)
            ]:
                df_all = df_plot.dropna(subset=[harga_col, 'rating'])
                zona = df_all[(df_all[harga_col] <= median_h) & (df_all['rating'] >= 4.0)]
                ax.scatter(df_all[harga_col], df_all['rating'], color=color, alpha=0.2, s=20)
                ax.scatter(zona[harga_col], zona['rating'], color='#f59e0b', edgecolors=C_DRK, s=55, zorder=5, label='Value for Money')
                ax.axvline(median_h, color='gray', linestyle='--', linewidth=1.2)
                ax.axhline(4.0, color='#22c55e', linestyle='--', linewidth=1.2)
                ax.axvspan(0, median_h, ymin=0.6, alpha=0.06, color='#f59e0b')
                ax.set_title(f'{label} — {len(zona)} produk VfM', fontsize=12, fontweight='700')
                ax.set_xlabel('Harga (Rp)'); ax.set_ylabel('Rating'); ax.set_ylim(0, 5.5)
                ax.set_xlim(0, df_all[harga_col].quantile(0.97)); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig)

        with sub3:
            df_diskon_ada   = df_w[df_w['diskon'].notna() & df_w['rating'].notna()]
            df_diskon_tidak = df_w[df_w['diskon'].isna()  & df_w['rating'].notna()]
            avg_r_ada   = df_diskon_ada['rating'].mean()
            avg_r_tidak = df_diskon_tidak['rating'].mean()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(df_diskon_ada['diskon'], df_diskon_ada['rating'], color=C_W, alpha=0.4, s=30)
            z = np.polyfit(df_diskon_ada['diskon'].dropna(), df_diskon_ada.loc[df_diskon_ada['diskon'].notna(),'rating'], 1)
            x_line = np.linspace(df_diskon_ada['diskon'].min(), df_diskon_ada['diskon'].max(), 100)
            axes[0].plot(x_line, np.poly1d(z)(x_line), color='#ef4444', linewidth=2, label=f'Tren (slope={z[0]:+.3f})')
            axes[0].set_title('Besar Diskon vs Rating', fontsize=12, fontweight='700')
            axes[0].set_xlabel('Diskon (%)'); axes[0].set_ylabel('Rating'); axes[0].legend()

            bars = axes[1].bar(['Ada Diskon','Tanpa Diskon'], [avg_r_ada, avg_r_tidak],
                               color=[C_ACC, C_MUT], width=0.4, edgecolor='white')
            for i, v in enumerate([avg_r_ada, avg_r_tidak]):
                axes[1].text(i, v+0.05, f'{v:.2f}', ha='center', fontsize=12, fontweight='700')
            axes[1].set_title('Rata-rata Rating: Diskon vs Tidak', fontsize=12, fontweight='700')
            axes[1].set_ylabel('Rata-rata Rating'); axes[1].set_ylim(0, 5.5)
            plt.tight_layout(); st.pyplot(fig)

            insight_box(f"""
            <h3>💡 Insight Value</h3>
            <ul>
            <li>Platform lebih terjangkau: <strong>Watsons</strong> (Rp{avg_harga_w:,.0f} vs Rp{avg_harga_s:,.0f})</li>
            <li>Platform dengan rating lebih tinggi: <strong>{'Watsons' if avg_rating_w > avg_rating_s else 'Sociolla'}</strong></li>
            <li>Diskon terbukti <strong>tidak meningkatkan</strong> kepuasan pembeli secara signifikan.</li>
            </ul>""")

    # ── V6 VARIABILITY ──
    with tabs[5]:
        st.markdown("#### V6 — Variability: Sebaran Harga Produk")
        c1, c2 = st.columns(2)
        with c1: metric_card("CV Watsons", f"{cv_w:.1f}%", "Coefficient of Variation harga", C_W)
        with c2: metric_card("CV Sociolla", f"{cv_s:.1f}%", "Coefficient of Variation harga", C_S)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, data, color, label, cv in zip(
            axes,
            [df_w['harga'].dropna(), df_s['harga_original'].dropna()],
            [C_W, C_S], ['Watsons', 'Sociolla'], [cv_w, cv_s]
        ):
            ax.boxplot(data, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.6),
                       medianprops=dict(color=C_DRK, linewidth=2),
                       flierprops=dict(marker='o', color=color, alpha=0.3, markersize=3))
            ax.set_title(f'{label} — CV = {cv:.1f}%', fontsize=12, fontweight='700')
            ax.set_ylabel('Harga (Rp)'); ax.set_xticklabels([label])
        plt.tight_layout(); st.pyplot(fig)

        lebih_variatif = 'Watsons' if cv_w > cv_s else 'Sociolla'
        insight_box(f"""
        <h3>📌 Temuan Variability</h3>
        <ul>
        <li>Platform dengan rentang harga paling luas: <strong>{lebih_variatif}</strong> (CV lebih tinggi).</li>
        <li>CV yang besar menandakan platform melayani segmen yang lebih beragam — dari <em>budget</em> hingga <em>premium</em>.</li>
        </ul>""")

    # ── V7 VISUALIZATION ──
    with tabs[6]:
        st.markdown("#### V7 — Visualization: Distribusi Komparatif")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for label, data, color in [('Watsons', df_w['harga'].dropna(), C_W), ('Sociolla', df_s['harga_original'].dropna(), C_S)]:
            axes[0].hist(data, bins=50, alpha=0.5, color=color, density=True, label=label)
        axes[0].axvline(avg_harga_w, color=C_W, linestyle='--', linewidth=1.8)
        axes[0].axvline(avg_harga_s, color=C_S, linestyle='--', linewidth=1.8)
        axes[0].set_title('Distribusi Harga', fontsize=12, fontweight='700')
        axes[0].set_xlabel('Harga (Rp)'); axes[0].set_ylabel('Densitas'); axes[0].set_xlim(0, 500000); axes[0].legend()

        for label, data, color in [('Watsons', df_w['rating'].dropna(), C_W), ('Sociolla', df_s['rating'].dropna(), C_S)]:
            axes[1].hist(data, bins=20, alpha=0.5, color=color, density=True, label=label)
        axes[1].set_title('Distribusi Rating', fontsize=12, fontweight='700')
        axes[1].set_xlabel('Rating'); axes[1].set_ylabel('Densitas'); axes[1].legend()

        axes[2].pie([len(df_w), len(df_s)], labels=['Watsons', 'Sociolla'], colors=[C_W, C_S],
                    autopct='%1.1f%%', startangle=140,
                    wedgeprops=dict(edgecolor='white', linewidth=2),
                    textprops=dict(fontsize=12, fontweight='700'))
        axes[2].set_title('Proporsi Produk', fontsize=12, fontweight='700')
        plt.tight_layout(); st.pyplot(fig)

        insight_box("""
        <h3>📌 Temuan Visualization</h3>
        <ul>
        <li>Distribusi harga keduanya <strong>right-skewed</strong> — sebagian besar produk di rentang harga rendah-menengah.</li>
        <li>Rating terkonsentrasi pada nilai tinggi (4–5), menunjukkan <strong>bias positif</strong> pada review e-commerce.</li>
        <li>Proporsi jumlah produk <strong>relatif seimbang</strong>, sehingga keduanya layak dibandingkan secara langsung.</li>
        </ul>""")

    # ── V8 VULNERABILITY ──
    with tabs[7]:
        st.markdown("#### V8 — Vulnerability: Missing Values")
        pct_missing_w = df_w.isnull().mean() * 100
        pct_missing_s = df_s.isnull().mean() * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, pct, color, label in [
            (axes[0], pct_missing_w, C_W, 'Watsons'),
            (axes[1], pct_missing_s, C_S, 'Sociolla')
        ]:
            bar_colors = [C_W if v >= 30 else color for v in pct.values]
            ax.barh(pct.index, pct.values, color=bar_colors, alpha=0.85, edgecolor='white')
            ax.axvline(30, color='#ef4444', linestyle='--', linewidth=1.5, label='Batas kritis 30%')
            ax.set_title(f'Missing Value — {label}', fontsize=12, fontweight='700')
            ax.set_xlabel('% Missing'); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig)

        kol_w = pct_missing_w[pct_missing_w > 30].index.tolist()
        kol_s = pct_missing_s[pct_missing_s > 30].index.tolist()
        insight_box(f"""
        <h3>📌 Temuan Vulnerability</h3>
        <ul>
        <li>Kolom kritis Watsons (missing >30%): <strong>{kol_w if kol_w else 'Tidak ada'}</strong></li>
        <li>Kolom kritis Sociolla (missing >30%): <strong>{kol_s if kol_s else 'Tidak ada'}</strong></li>
        <li>Ketidakseimbangan atribut <em>diskon</em> dan <em>jumlah_terjual</em> menjadi kerentanan utama dalam analisis lintas platform.</li>
        </ul>""")

# ─────────────────────────────────────────
# ANALISIS KUISIONER
# ─────────────────────────────────────────
elif menu == "Analisis Kuisioner":
    st.markdown('<div class="page-title">📋 Analisis Form Kuisioner</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Persepsi konsumen vs data aktual · Likert · Demografi · Review Behavior</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Gap Analysis", "Likert Comparison", "Demografi", "Review Behavior"])

    # ── GAP ANALYSIS ──
    with tab1:
        sub1, sub2, sub3 = st.tabs(["Info", "Visualisasi", "Insight"])
        with sub1:
            st.markdown("""
            #### Gap Analysis
            Membandingkan persepsi responden (kuesioner) dengan kondisi aktual hasil web scraping.
            Semua indikator dinormalisasi ke skala 1–4 untuk perbandingan langsung.
            """)
            for d in ["Rating Produk", "Pengaruh Diskon", "Frekuensi Review", "Pengaruh Jumlah Terjual"]:
                st.markdown(f"- {d}")

        with sub2:
            w_rating_norm = (df_w['rating'].mean()-1)/(5-1)*3+1
            s_rating_norm = (df_s['rating'].mean()-1)/(5-1)*3+1
            w_diskon_norm = 1+(df_w['diskon'].mean()/50)*3
            s_diskon_norm = 1.0
            w_rev_log = np.log1p(avg_review_w); s_rev_log = np.log1p(avg_review_s)
            max_log = max(w_rev_log, s_rev_log)
            w_rev_norm = 1+(w_rev_log/max_log)*3; s_rev_norm = 1+(s_rev_log/max_log)*3
            s_terjual_norm = 1+(np.log1p(avg_terjual_s)/max_log)*3

            dimensi_label = ['Rating\nProduk','Pengaruh\nDiskon','Freq\nReview','Jml\nTerjual']
            persepsi_w = [df_r['w_rating'].mean(), df_r['w_diskon'].mean(), df_r['w_review'].mean(), df_r['w_terjual'].mean()]
            persepsi_s = [df_r['s_rating'].mean(), df_r['s_diskon'].mean(), df_r['s_review'].mean(), df_r['s_terjual'].mean()]
            aktual_w = [w_rating_norm, w_diskon_norm, w_rev_norm, 2.5]
            aktual_s = [s_rating_norm, s_diskon_norm, s_rev_norm, s_terjual_norm]

            x = np.arange(len(dimensi_label)); width = 0.18
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
            fig.suptitle('GAP ANALYSIS: Persepsi Responden vs Data Scraping (Skala 1–4)', fontsize=13, fontweight='700')
            for ax, (toko, p_vals, a_vals, color) in zip(axes, [
                ('Watsons', persepsi_w, aktual_w, C_W), ('Sociolla', persepsi_s, aktual_s, C_S)
            ]):
                ax.bar(x-width/2, p_vals, width, label='Persepsi Responden', color=color, alpha=0.9, edgecolor='white')
                ax.bar(x+width/2, a_vals, width, label='Data Aktual', color=color, alpha=0.35, edgecolor=color, linewidth=1.5)
                for i, (p, a) in enumerate(zip(p_vals, a_vals)):
                    gap = p-a; clr = '#ef4444' if gap > 0.3 else '#22c55e' if gap < -0.3 else '#9ca3af'
                    ax.annotate(f'Δ{gap:+.2f}', xy=(x[i], max(p,a)+0.05), ha='center', fontsize=9, color=clr, fontweight='700')
                ax.set_title(toko, fontsize=12, fontweight='700', color=color)
                ax.set_xticks(x); ax.set_xticklabels(dimensi_label, fontsize=9)
                ax.set_ylim(0, 4.8); ax.set_ylabel('Skor (1–4)'); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig)

        with sub3:
            insight_box(f"""
            <h3>📌 Temuan Gap Analysis</h3>
            <ul>
            <li>Diskon Watsons berpengaruh kuat — persepsi responden rata-rata <strong>{df_r['w_diskon'].mean():.2f}/4</strong>, selaras data scraping ({df_w['diskon'].mean():.1f}% rata-rata diskon).</li>
            <li>Kualitas produk Watsons cenderung <strong>di-underestimate</strong> oleh responden (rating aktual {df_w['rating'].mean():.2f}/5 vs persepsi {df_r['w_rating'].mean():.2f}/4).</li>
            <li>Sociolla memiliki review per produk lebih tinggi (<strong>{avg_review_s:.0f}</strong>) vs Watsons (<strong>{avg_review_w:.0f}</strong>).</li>
            </ul>""")

    # ── LIKERT ──
    with tab2:
        sub1, sub2, sub3 = st.tabs(["Info", "Visualisasi", "Insight"])
        dimensi = {
            'Kelengkapan\nProduk': ('w_kelengkapan','s_kelengkapan'),
            'Pengaruh\nDiskon':    ('w_diskon','s_diskon'),
            'Pengaruh\nRating':    ('w_rating','s_rating'),
            'Jml Terjual':         ('w_terjual','s_terjual'),
            'Freq Review':         ('w_review','s_review'),
        }
        labels   = list(dimensi.keys())
        w_scores = [df_r[v[0]].mean() for v in dimensi.values()]
        s_scores = [df_r[v[1]].mean() for v in dimensi.values()]

        with sub1:
            st.markdown("#### Likert Comparison\nPerbandingan persepsi responden Watsons vs Sociolla (skala 1–4).")

        with sub2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('KOMPARASI PERSEPSI: Watsons vs Sociolla (Skala 1–4)', fontsize=13, fontweight='700')
            x = np.arange(len(labels)); w = 0.35
            b1 = ax1.bar(x-w/2, w_scores, w, label='Watsons',  color=C_W, alpha=0.9, edgecolor='white')
            b2 = ax1.bar(x+w/2, s_scores, w, label='Sociolla', color=C_S, alpha=0.9, edgecolor='white')
            for bars in [b1,b2]:
                for bar in bars:
                    h = bar.get_height()
                    ax1.text(bar.get_x()+bar.get_width()/2, h+0.03, f'{h:.2f}', ha='center', fontsize=8, fontweight='700')
            ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
            ax1.set_ylim(0,4.5); ax1.set_ylabel('Rata-rata Skor Likert'); ax1.legend(); ax1.set_title('Bar Chart')

            N = len(labels); angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]
            w_vals = w_scores + w_scores[:1]; s_vals = s_scores + s_scores[:1]
            ax2 = plt.subplot(122, polar=True)
            ax2.plot(angles, w_vals, 'o-', linewidth=2, color=C_W, label='Watsons'); ax2.fill(angles, w_vals, alpha=0.18, color=C_W)
            ax2.plot(angles, s_vals, 'o-', linewidth=2, color=C_S, label='Sociolla'); ax2.fill(angles, s_vals, alpha=0.18, color=C_S)
            ax2.set_xticks(angles[:-1]); ax2.set_xticklabels([l.replace('\n',' ') for l in labels], fontsize=8)
            ax2.set_ylim(0,4); ax2.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=9); ax2.set_title('Radar Chart')
            plt.tight_layout(); st.pyplot(fig)

        with sub3:
            rows = "".join([f"<li><strong>{l.replace(chr(10),' ')}</strong>: {'Sociolla' if s>w else 'Watsons'} lebih unggul (W={w:.2f} vs S={s:.2f})</li>" for l, w, s in zip(labels, w_scores, s_scores)])
            insight_box(f"<h3>📌 Temuan Likert</h3><ul>{rows}</ul>")

    # ── DEMOGRAFI ──
    with tab3:
        sub1, sub2, sub3, sub4 = st.tabs(["Info", "Demografi", "Preferensi", "Insight"])
        with sub1:
            st.markdown("#### Demographic Segmentation\nSegmentasi berdasarkan gender, pengeluaran, dan preferensi platform.")

        with sub2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('SEGMENTASI DEMOGRAFI vs PREFERENSI TOKO', fontsize=13, fontweight='700')
            ct_gender = pd.crosstab(df_r['gender'], df_r['rekomendasi'])
            ct_gender_pct = ct_gender.div(ct_gender.sum(axis=1),axis=0)*100
            ct_gender_pct.plot(kind='bar', ax=axes[0], color=[C_S, C_W], edgecolor='white', width=0.55)
            chi2_g, p_g, _, _ = chi2_contingency(ct_gender)
            sig_g = 'Signifikan ✅' if p_g < 0.05 else 'Tidak Signifikan'
            axes[0].set_title(f'Gender vs Rekomendasi\nChi²={chi2_g:.2f}, p={p_g:.3f} → {sig_g}', fontsize=11, fontweight='700')
            axes[0].set_ylim(0,110); axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
            axes[0].legend(['Sociolla','Watsons'])
            for c in axes[0].containers: axes[0].bar_label(c, fmt='%.1f%%', fontsize=9, padding=3)

            spending_order = ['< Rp100.000','Rp100.000 – Rp300.000','Rp300.001 – Rp600.000','Rp600.001 – Rp1.000.000','> Rp1.000.000']
            ct_spend = pd.crosstab(df_r['spending'], df_r['rekomendasi']).reindex(spending_order, fill_value=0)
            ct_spend_pct = ct_spend.div(ct_spend.sum(axis=1),axis=0)*100
            ct_spend_pct.plot(kind='bar', ax=axes[1], color=[C_S, C_W], edgecolor='white', width=0.65)
            chi2_s, p_s, _, _ = chi2_contingency(ct_spend)
            sig_s = 'Signifikan ✅' if p_s < 0.05 else 'Tidak Signifikan'
            axes[1].set_title(f'Pengeluaran vs Rekomendasi\nChi²={chi2_s:.2f}, p={p_s:.3f} → {sig_s}', fontsize=11, fontweight='700')
            axes[1].set_ylim(0,110)
            axes[1].set_xticklabels([s.replace('Rp','Rp\n') for s in spending_order], rotation=15, fontsize=8, ha='right')
            axes[1].legend(['Sociolla','Watsons'])
            for c in axes[1].containers: axes[1].bar_label(c, fmt='%.0f%%', fontsize=8, padding=3)
            plt.tight_layout(); st.pyplot(fig)

        with sub3:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('PREFERENSI PLATFORM RESPONDEN', fontsize=14, fontweight='700')
            toko_used = df_r['toko_digunakan'].value_counts()
            colors_u = [C_S if t=='Sociolla' else C_W for t in toko_used.index]
            bars = axes[0,0].bar(toko_used.index, toku_used := toko_used.values if False else toko_used.values,
                                  color=colors_u, width=0.45, edgecolor='white')
            for bar, val in zip(bars, toko_used.values):
                axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                               f'{val}\n({val/len(df_r)*100:.0f}%)', ha='center', fontsize=10, fontweight='700')
            axes[0,0].set_title('Toko Aktif Digunakan', fontsize=12, fontweight='700'); axes[0,0].set_ylim(0,120)

            toko_rec = df_r['rekomendasi'].value_counts()
            colors_r = [C_S if t=='Sociolla' else C_W for t in toko_rec.index]
            bars = axes[0,1].bar(toko_rec.index, toko_rec.values, color=colors_r, width=0.45, edgecolor='white')
            for bar, val in zip(bars, toko_rec.values):
                axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                               f'{val}\n({val/len(df_r)*100:.0f}%)', ha='center', fontsize=10, fontweight='700')
            axes[0,1].set_title('Toko yang Direkomendasikan', fontsize=12, fontweight='700'); axes[0,1].set_ylim(0,120)

            w_cat = df_r['w_kategori'].dropna().str.split(',').explode().str.strip().value_counts()
            axes[1,0].barh(w_cat.index[::-1], w_cat.values[::-1], color=C_W, alpha=0.85, edgecolor='white')
            axes[1,0].set_title('Kategori Paling Dicari — Watsons', fontsize=12, fontweight='700', color=C_W)
            axes[1,0].set_xlabel('Jumlah Responden'); axes[1,0].set_xlim(0,70)

            s_cat = df_r['s_kategori'].dropna().str.split(',').explode().str.strip().value_counts()
            axes[1,1].barh(s_cat.index[::-1], s_cat.values[::-1], color=C_S, alpha=0.85, edgecolor='white')
            axes[1,1].set_title('Kategori Paling Dicari — Sociolla', fontsize=12, fontweight='700', color=C_S)
            axes[1,1].set_xlabel('Jumlah Responden'); axes[1,1].set_xlim(0,110)
            plt.tight_layout(); st.pyplot(fig)

        with sub4:
            insight_box(f"""
            <h3>📌 Temuan Demografi</h3>
            <ul>
            <li><strong>Perempuan</strong> cenderung memilih Sociolla; <strong>laki-laki</strong> lebih ke Watsons.</li>
            <li>Hubungan gender-preferensi {'<strong>signifikan secara statistik</strong>' if p_g<0.05 else 'tidak signifikan'} (p={p_g:.3f}).</li>
            <li>Pengeluaran tinggi (>Rp600rb) lebih memilih Sociolla → positioning <strong>lebih premium</strong>.</li>
            <li>Sociolla lebih banyak digunakan & direkomendasikan sehari-hari.</li>
            <li>Kategori dominan keduanya: <strong>skincare & makeup</strong>. Watsons relatif lebih kuat pada farmasi/kesehatan.</li>
            </ul>""")

    # ── REVIEW BEHAVIOR ──
    with tab4:
        sub1, sub2, sub3 = st.tabs(["Info", "Visualisasi", "Insight"])
        with sub1:
            st.markdown("#### Review Behavior Gap\nPerilaku pengguna dalam menulis review vs volume review aktual di marketplace.")

        with sub2:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('REVIEW BEHAVIOR GAP: Survei vs Volume Review Aktual', fontsize=13, fontweight='700')
            labels_freq = ['Tidak Pernah\n(1)','Jarang\n(2)','Kadang\n(3)','Sering\n(4)']
            w_freq = [df_r['w_review'].eq(i).sum() for i in [1,2,3,4]]
            s_freq = [df_r['s_review'].eq(i).sum() for i in [1,2,3,4]]
            x = np.arange(4); width = 0.35
            axes[0].bar(x-width/2, w_freq, width, label='Watsons',  color=C_W, alpha=0.9, edgecolor='white')
            axes[0].bar(x+width/2, s_freq, width, label='Sociolla', color=C_S, alpha=0.9, edgecolor='white')
            axes[0].set_xticks(x); axes[0].set_xticklabels(labels_freq, fontsize=9)
            axes[0].set_title('Frekuensi Nulis Review (Survei)', fontsize=11, fontweight='700')
            axes[0].set_ylabel('Jumlah Responden'); axes[0].legend()

            axes[1].hist(df_w['jumlah_review'].clip(upper=100), bins=30, alpha=0.7, color=C_W, edgecolor='white', density=True, label='Watsons')
            axes[1].hist(df_s['jumlah_review'].clip(upper=100), bins=30, alpha=0.7, color=C_S, edgecolor='white', density=True, label='Sociolla')
            axes[1].axvline(avg_review_w, color=C_W, linestyle='--', linewidth=2, label=f'Mean W={avg_review_w:.0f}')
            axes[1].axvline(avg_review_s, color=C_S, linestyle='--', linewidth=2, label=f'Mean S={avg_review_s:.0f}')
            axes[1].set_title('Distribusi Review/Produk (Scraping)', fontsize=11, fontweight='700')
            axes[1].set_xlabel('Jumlah Review'); axes[1].set_ylabel('Densitas'); axes[1].legend(fontsize=8)

            axes[2].axis('off')
            metrics_info = [
                (f'Avg Skor Review — Watsons', f'{df_r["w_review"].mean():.2f}/4', C_W),
                (f'Avg Skor Review — Sociolla', f'{df_r["s_review"].mean():.2f}/4', C_S),
                (f'Avg Review/Produk — Watsons', f'{avg_review_w:.1f}', C_W),
                (f'Avg Review/Produk — Sociolla', f'{avg_review_s:.1f}', C_S),
                (f'Rasio Sociolla vs Watsons', f'{avg_review_s/avg_review_w:.0f}x lebih banyak', C_ACC),
            ]
            axes[2].text(0.5, 0.97, 'RINGKASAN', ha='center', va='top', fontsize=12, fontweight='700')
            y = 0.82
            for label_, val, clr in metrics_info:
                axes[2].text(0.05, y, label_, ha='left', fontsize=9, color='#555')
                axes[2].text(0.95, y, val, ha='right', fontsize=12, fontweight='700', color=clr)
                y -= 0.14
            plt.tight_layout(); st.pyplot(fig)

        with sub3:
            insight_box(f"""
            <h3>📌 Temuan Review Behavior</h3>
            <ul>
            <li>Mayoritas responden <strong>tidak pernah atau jarang</strong> menulis review — engagement masih rendah.</li>
            <li>Review per produk Sociolla sekitar <strong>{avg_review_s/avg_review_w:.0f}x lebih tinggi</strong> dari Watsons.</li>
            <li>Volume review kemungkinan berasal dari sejumlah kecil <em>power reviewer</em>, bukan mayoritas pembeli.</li>
            <li>Program insentif review (poin loyalitas, voucher) dapat meningkatkan partisipasi pengguna.</li>
            </ul>""")