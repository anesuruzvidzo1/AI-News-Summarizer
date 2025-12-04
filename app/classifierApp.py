
import streamlit as st
from pymongo import MongoClient
import math
from pathlib import Path
import html as html_lib  


# DB

client = MongoClient("mongodb://localhost:27017/")
db = client["news_db"]
classified = db["classified_articles"]


# Page & Theme

st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

#dark CSS
st.markdown(
    """
    <style>
    /* page background and typography */
    .reportview-container, .main {
        background-color: #0a0a0a;
        color: #ffffff;
        font-family: Arial, Helvetica, sans-serif;
    }
    /* Headlines (BBC red) */
    .news-title {
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: #bb1919;
    }
    /* Links in red, open in new tab */
    .news-link {
        color: #bb1919;
        text-decoration: none;
        font-weight: 700;
    }
    .news-link:hover {
        text-decoration: underline;
    }
    /* summaries */
    .news-summary {
        color: #eaeaea;
        margin: 0 0 8px 0;
        font-size: 15px;
    }
    /* small metadata text */
    .meta {
        color: #bdbdbd;
        font-size: 12px;
        margin-top: 6px;
    }
    hr.st-sep {
        border: 0;
        border-top: 1px solid #222;
        margin: 12px 0;
    }
    /* sidebar pipeline message box adjustments */
    .stSidebar .stButton>button {
        background-color: #1a1a1a;
        color: #fff;
        border: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📰 Summarized & Classified News")



# Sidebar: pipeline button

st.sidebar.header("Pipeline")
if st.sidebar.button("Refresh"):
    #  simple blocking
    PROC = Path(__file__).resolve().parents[1] / "scripts" / "pipeline.py"
    if PROC.exists():
        st.sidebar.info("Running pipeline... this may take a few minutes.")
        # run with venv python (adjust if your venv path differs)
        venv_python = str(Path(__file__).resolve().parents[1] / "venv" / "bin" / "python")
        # fallback to system python if venv not present
        if not Path(venv_python).exists():
            venv_python = "python"
        import subprocess
        cmd = [venv_python, str(PROC)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Pipeline finished")
        else:
            st.sidebar.error("Pipeline failed. See logs.")
            st.sidebar.text(result.stderr[:1000])
    else:
        st.sidebar.warning("pipeline.py not found in scripts/")


# Category selector & pagination

categories = sorted(classified.distinct("predicted_category"))
if not categories:
    st.info("No classified articles yet. Run the pipeline or classify articles first.")
else:
    # Put category selector in main area for better visibility
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_category = st.selectbox("Choose a category:", categories)
    with col2:
        # page_size can be moved to slider or select box if you want
        page_size = st.selectbox("Per page", [5, 10, 20], index=1)

    total = classified.count_documents({"predicted_category": selected_category})
    pages = math.ceil(total / page_size) if total else 1

    # Use st.number_input but place it horizontally
    page = st.number_input("Page", min_value=1, max_value=max(1, pages), value=1, step=1)
    skip = (page - 1) * page_size

    cursor = (
        classified.find({"predicted_category": selected_category})
        .sort("publishedAt", -1)
        .skip(skip)
        .limit(page_size)
    )

    # If there are results, optionally show a pinned top story
    docs = list(cursor)
    if not docs:
        st.warning("No articles found for this category.")
    else:
        for doc in docs:
            title = doc.get("title", "(no title)")
            url = doc.get("url")
            # prepare short summary (use stored short or make one from summary)
            short = doc.get("summary_short", "")
            if not short:
                s = doc.get("summary", "") or ""
                short = (s[:200] + "...") if len(s) > 200 else s

            # safely escape the title and url to avoid breaking HTML
            title_html = html_lib.escape(title)
            url_html = html_lib.escape(url) if url else ""

            # Title as clickable link that opens in a new tab; styled with news-link
            if url_html:
                st.markdown(f"<p class='news-title'><a class='news-link' href='{url_html}' target='_blank' rel='noopener'>{title_html}</a></p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='news-title'>{title_html}</p>", unsafe_allow_html=True)

            # short summary
            if short:
                st.markdown(f"<p class='news-summary'>{html_lib.escape(short)}</p>", unsafe_allow_html=True)

            # Expander for full summary only if it exists and is meaningfully longer than short
            full = doc.get("summary", "") or ""
            # normalize whitespace for comparison
            if full and len(full.strip()) > len(short.strip()) + 20:
                # label is explicit to avoid redundancy
                with st.expander("Show full summary"):
                    st.write(full)

            # meta
            pub = doc.get("publishedAt")
            src = doc.get("source", "") or doc.get("source_name", "")
            meta_parts = []
            if src:
                meta_parts.append(str(src))
            if pub:
                meta_parts.append(str(pub))
            meta_parts.append(f"Category: {doc.get('predicted_category','')}")
            st.markdown(f"<div class='meta'>{' • '.join(meta_parts)}</div>", unsafe_allow_html=True)
            st.markdown("<hr class='st-sep'/>", unsafe_allow_html=True)

        st.markdown(f"Showing page {page} of {pages} — total {total} articles.")

