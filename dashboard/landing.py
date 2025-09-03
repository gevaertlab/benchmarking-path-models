
import streamlit as st
import base64

# Function to encode Stanford logo to base64
def get_base64_image(image_path):
    """
    Encode an image file to base64 for embedding in HTML.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error("Stanford logo file not found. Please provide SUSig_2color_Stree_Stacked_Left.png in the same directory.")
        return ""

# Landing page content
st.title("Welcome to PathBench")
st.markdown(
    """
    <div style='text-align: center; font-size: 24px; color: #333333; margin-bottom: 20px;'>
        A state-of-the-art platform for benchmarking pathology foundation models
    </div>
    """,
    unsafe_allow_html=True
)

# Display Stanford University logo
stanford_logo = get_base64_image("SUSig_2color_Stree_Stacked_Left.png")
st.markdown(
    f'<img src="data:image/png;base64,{stanford_logo}" width="200" style="display: block; margin: 20px auto;">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; font-size: 18px; color: #333333; line-height: 1.6;'>
        PathBench is developed by the <a href='https://med.stanford.edu/gevaertlab.html' target='_blank'>Olivier Gevaert Lab</a> at Stanford University to evaluate and compare pathology foundation models across diverse datasets including TCGA, CPTAC, and External tasks. Explore performance metrics like AUROC, AUPRC, Sensitivity, Specificity, and Balanced Accuracy through interactive visualizations and tables.<br><br>
        Learn more in our <a href='https://www.medrxiv.org/content/10.1101/2025.05.08.25327250v1' target='_blank'>preprint</a>.<br><br>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation button to dashboard
if st.button("Explore the Dashboard", key="navigate_dashboard"):
    st.switch_page("dashboard.py")

st.markdown(
    """
    <div class='footer'>
        PathBench: A benchmarking platform for pathology foundation models.<br>
        Developed by the <a href='https://med.stanford.edu/gevaertlab.html' target='_blank'>Olivier Gevaert Lab</a> at Stanford University.<br>
        Preprint: <a href='https://www.medrxiv.org/content/10.1101/2025.05.08.25327250v1' target='_blank'>Benchmarking Pathology Foundation Models</a>
    </div>
    """,
    unsafe_allow_html=True
)