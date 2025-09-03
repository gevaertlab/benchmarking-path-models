
import streamlit as st

# Configure Streamlit page for wide layout
st.set_page_config(layout="wide", page_title="PathBench", page_icon="🔬")

# Custom CSS for global styling
st.markdown(
    """
    <style>
    /* Global font and background */
/* Global font and background */
body {
    font-family: 'Roboto', Arial, sans-serif !important;
    background-color: #F5F6F5 !important;
    color: #333333 !important;
}
/* Main title styling with Stanford Cardinal Red */
h1 {
    font-size: 48px !important;
    font-weight: 700 !important;
    color: #B1040E !important;
    text-align: center;
    margin-bottom: 20px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
/* Headers styling */
h2 {
    font-size: 36px !important;
    font-weight: 600 !important;
    color: #B1040E !important;
    margin-top: 20px;
}
h3 {
    font-size: 28px !important;
    font-weight: 500 !important;
    color: #B1040E !important;
    margin-top: 15px;
}
/* Container styling */
.stApp {
    background-color: #FFFFFF !important;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    max-width: 1400px;
    margin: 0 auto;
}
/* Remove black header section */
header[data-testid="stHeader"], .st-emotion-cache-1wmy9hl {
    background-color: #F5F6F5 !important;
    border: none !important;
}
/* Sidebar styling */
section[data-testid="stSidebar"], [data-testid="stSidebar"] .stSidebar, [data-testid="stSidebar"] .css-1r6slb0, [data-testid="stSidebar"] .css-17rl0q2, [data-testid="stSidebar"] .st-emotion-cache-17rl0q2 {
    background-color: #F5F6F5 !important;
    border-right: 1px solid #E0E0E0 !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 20px !important;
    box-shadow: 2px 0 4px rgba(0,0,0,0.05) !important;
    max-height: calc(100vh - 60px) !important;
    width: 280px !important;
}
/* Sidebar header and subheader */
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 34px !important;
    font-weight: 600 !important;
    color: #B1040E !important;
    margin-bottom: 15px;
}
section[data-testid="stSidebar"] h3 {
    font-size: 28px !important;
    font-weight: 500 !important;
}
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 5px !important;
    margin-bottom: 10px !important;
}
div[data-testid="stExpander"] .stMarkdown {
    font-size: 20px !important;
}
/* Enhance selectbox, multiselect, slider, and text input */
/* Enhance selectbox, multiselect, slider, and text input */
div[data-testid="stSelectbox"], div[data-testid="stMultiSelect"], div[data-testid="stSlider"], div[data-testid="stTextInput"] {
    max-width: 250px !important; /* Reduced to fit within 280px sidebar width */
    min-width: 220px !important; /* Reduced to fit within sidebar */
    margin: 12px 0 !important;
    background-color: #FFFFFF !important;
    border: 1px solid #B1040E !important;
    border-radius: 5px !important;
    padding: 6px !important;
}
div[data-testid="stSelectbox"] select, div[data-testid="stMultiSelect"] select {
    font-size: 14px !important; /* Reduced to fit within width */
    padding: 6px !important; /* Reduced padding */
    border: none !important;
    width: 100% !important;
}
[data-testid="stSidebar"] div[data-testid="stSelectbox"] .stMarkdown label, [data-testid="stSidebar"] div[data-testid="stMultiSelect"] .stMarkdown label, [data-testid="stSidebar"] div[data-testid="stSlider"] .stMarkdown label, [data-testid="stSidebar"] div[data-testid="stTextInput"] .stMarkdown label, [data-testid="stSidebar"] div[class*="emotion-cache"] label, [data-testid="stSidebar"] div[class*="emotion-cache"] p {
    color: #B1040E !important;
    margin-bottom: 6px !important;
    font-size: 16px !important;
}
div[role="listbox"], div[data-testid="stMultiSelect"] div[role="listbox"] ul li {
    font-size: 14px !important; /* Reduced to fit within sidebar */
    color: #333333 !important;
    background-color: #FFFFFF !important;
    visibility: visible !important;
    z-index: 1000 !important;
    position: relative !important;
    overflow: visible !important;
    min-width: 220px !important; /* Matches container min-width */
    max-height: 200px !important; /* Reduced to fit sidebar height */
}
div[role="listbox"] div[data-baseweb="menu"] div, div[data-testid="stMultiSelect"] div[role="listbox"] ul li span {
    font-size: 14px !important; /* Reduced to fit within sidebar */
    padding: 6px !important; /* Reduced padding */
    color: #333333 !important;
    background-color: #FFFFFF !important;
    visibility: visible !important;
    z-index: 1001 !important;
    min-width: 220px !important; /* Matches container min-width */
    white-space: normal !important;
}
div[role="listbox"] div[data-baseweb="menu"] div:hover, div[data-testid="stMultiSelect"] div[role="listbox"] ul li:hover {
    background-color: #E0E0E0 !important;
    color: #333333 !important;
}
div[data-testid="stSidebar"] .stDivider {
    background-color: #B1040E !important;
    height: 2px !important;
    margin: 15px 0 !important;
}
/* Button styling */
button {
    background-color: #B1040E !important;
    color: #FFFFFF !important;
    border-radius: 5px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    transition: background-color 0.3s ease;
}
button:hover {
    background-color: #8C1515 !important;
}
/* Footer styling */
.footer {
    font-size: 16px !important;
    color: #333333 !important;
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    border-top: 1px solid #E0E0E0;
    background-color: #F5F6F5;
    border-radius: 5px;
}
/* Chart container styling */
.vega-embed {
    width: 100% !important;
    background-color: #FFFFFF !important;
    border-radius: 5px;
    padding: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.vega-embed text {
    font-size: 20px !important;
}
.vega-embed .role-title text {
    font-size: 10px !important;
    color: red!important;
}
/* Navigation menu labels */
[data-testid="stSidebarNav"] li a, [data-testid="stSidebarNav"] li a span, [data-testid="stSidebarNav"] li p, [data-testid="stSidebarNavItems"] li a, [data-testid="stSidebarNavItems"] li a span, [data-testid="stSidebarNavItems"] li p, [data-testid="stSidebarNav"] [class*="emotion-cache"] * {    color: #FF0000 !important;
    font-size: 20px !important;
}
    </style>
    """,
    unsafe_allow_html=True
)

# Define pages
pg = st.navigation([
    st.Page("landing.py", title="Welcome", icon="🏠"),
    st.Page("dashboard.py", title="Dashboard", icon="📊")
])

# Run the selected page
pg.run()