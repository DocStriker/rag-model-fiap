import streamlit as st

from components.chat import render_chat
from components.sidebar import render_sidebar
from services.api_client import ApiClient

st.set_page_config(
    page_title="RAG Chabot",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* hide streamlit chrome (keep header for the sidebar toggle) */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .stDeployButton { display: none !important; }

    /* keep the sidebar expand button visible */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* page container */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 6rem;
        max-width: 820px;
    }

    /* heading */
    h1 {
        font-weight: 600;
        letter-spacing: -0.025em;
        font-size: 1.75rem;
        margin: 0 0 0.25rem 0;
        color: #111827;
    }

    /* status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-radius: 999px;
        font-size: 12.5px;
        color: #047857;
        font-weight: 500;
        margin: 0.25rem 0 1.5rem 0;
    }
    .status-pill.idle {
        background: rgba(107, 114, 128, 0.06);
        border-color: rgba(107, 114, 128, 0.18);
        color: #4b5563;
    }
    .status-pill .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
    }
    .status-pill.idle .dot { background: #9ca3af; }
    .status-pill b { font-weight: 600; }

    /* sidebar */
    [data-testid="stSidebar"] {
        background: #fafaf9;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280 !important;
        margin: 1.25rem 0 0.5rem 0 !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: #e5e7eb;
    }

    /* chat input rounded */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("RAG BOT")

api_client = ApiClient()
selected_collection = render_sidebar(api_client)
render_chat(api_client, collection=selected_collection)