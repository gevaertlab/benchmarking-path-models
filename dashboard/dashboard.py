
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder
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

# Display Stanford University logo
stanford_logo = get_base64_image("SUSig_2color_Stree_Stacked_Left.png")
st.markdown(
    f'<img src="data:image/png;base64,{stanford_logo}" width="120" style="display: block; margin: 20px auto;">',
    unsafe_allow_html=True
)

# Title and lab information
st.title("PathBench: Pathology Foundation Models Benchmarking Dashboard")
st.markdown(
    """
    <div style='text-align: center; font-size: 18px; color: #333333; margin-bottom: 20px;'>
        Developed by the <a href='https://med.stanford.edu/gevaertlab.html' target='_blank'>Olivier Gevaert Lab</a> at Stanford University.<br>
        Preprint: <a href='https://www.medrxiv.org/content/10.1101/2025.05.08.25327250v1' target='_blank'>Benchmarking Pathology Foundation Models</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Load and preprocess DataFrame
df = pd.read_csv("../data/benchmarking_wci.csv")
df = df.drop(columns=['ModelName'])
df = df.rename(columns={
    'ModelName_short': 'Model Name',
    'Task_short': 'Task',
    'ModelCategory': 'Model Category',
    'ModelArch': 'Model Architecture',
    'ModelDomain': 'Model Domain'
})
df.columns = df.columns.str.replace('_', ' ')
df['Task'] = df['Task'].str.replace('_', ' ', regex=False)

# Round numeric columns to 3 decimal places
numeric_columns = ['AUROC', 'AUROC lower', 'AUROC upper', 'AUPRC', 'AUPRC lower', 'AUPRC upper',
                   'Balanced accuracy', 'Balanced accuracy lower', 'Balanced accuracy upper',
                   'Sensitivity', 'Sensitivity lower', 'Sensitivity upper',
                   'Specificity', 'Specificity lower', 'Specificity upper',
                   'Parameters in million', 'TrainingData patches in million', 'Slides in million']
df[numeric_columns] = df[numeric_columns].round(3)

# Map model categories for clarity
category_mapping = {
    'vision': 'VM',
    'vision_language': 'VLM',
    'pathology': 'Path-VM',
    'pathology_language': 'Path-VLM'
}
category_mapping2 = {
    'vision': 'General Vision',
    'vision_language': 'General Vision Language',
    'pathology': 'Pathology-Specific Vision Model',
    'pathology_language': 'Pathology-Specific Vision Language Model'
}
df['Model Category Abbreviation'] = df['Model Category'].map(category_mapping).fillna(df['Model Category'])
df['Model Category'] = df['Model Category'].map(category_mapping2).fillna(df['Model Category'])
df = df.replace({None: np.nan})

# Sidebar for filters and metric selection
# Sidebar for filters and metric selection
st.sidebar.header("Filter & Visualize")
st.sidebar.subheader("Performance Metric")
metrics = ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Balanced accuracy']
selected_metric = st.sidebar.selectbox("Metric", metrics, key="metric_select")

# Note for fusion model metric limitation
if 'fusion model' in df['Model Name'].unique():
    st.sidebar.markdown(
        """
        <div style='color: #B1040E; font-size: 14px; margin-top: 5px;'>
            <strong>Note:</strong> For the 'fusion model', <em>AUROC</em> and <em>AUPRC</em> are invalid due to 
            multi-class majority voting (defaulting to 0.50), but <em>Balanced accuracy</em>, 
            <em>Sensitivity</em>, and <em>Specificity</em> are correct metrics to use.
        </div>
        """,
        unsafe_allow_html=True
    )
    
# Debugging info in an expander
with st.sidebar.expander("Dataset Summary"):
    st.write(f"Models: {df['Model Name'].nunique()}")
    st.write(f"Tasks: {df['Task'].nunique()}")

st.sidebar.divider()

# Filter function
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a sidebar UI to filter DataFrame columns, preserving all tasks and models unless explicitly filtered.

    Args:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = df.copy()
    df[numeric_columns] = df[numeric_columns].round(3)  # Re-apply rounding
    to_filter_columns = st.sidebar.multiselect("Filter Columns", df.columns, key="filter_cols")
    for column in to_filter_columns:
        left, right = st.sidebar.columns((1, 3))  # Adjusted ratio for alignment
        if column in ['Task', 'Model Name']:  # Ensure all tasks and models included by default
            user_cat_input = right.multiselect(
                f"Values for {column}", df[column].unique(), default=list(df[column].unique()), key=f"cat_{column}"
            )
            df = df[df[column].isin(user_cat_input)]
        elif df[column].nunique() < 10:
            user_cat_input = right.multiselect(
                f"Values for {column}", df[column].unique(), default=list(df[column].unique()), key=f"cat_{column}"
            )
            df = df[df[column].isin(user_cat_input)]
        elif df[column].dtype in ['float64', 'int64']:
            _min = float(df[column].min())
            _max = float(df[column].max())
            step = (_max - _min) / 100
            user_num_input = right.slider(
                f"Values for {column}", min_value=_min, max_value=_max, value=(_min, _max), step=step, key=f"num_{column}"
            )
            df = df[df[column].between(*user_num_input)]
        else:
            user_text_input = right.text_input(f"Substring in {column}", key=f"text_{column}")
            if user_text_input:
                df = df[df[column].astype(str).str.contains(user_text_input, case=False, na=False)]
    with st.sidebar.expander("Filtered Dataset Info"):
        st.write(f"Total unique models after filtering: {df['Model Name'].nunique()}")
        st.write(f"Total unique tasks after filtering: {df['Task'].nunique()}")
    return df

# Apply filters and ensure no NA in selected metric
newdf = filter_dataframe(df)
newdf = newdf.dropna(subset=[selected_metric])

# Define top 5 models for label prioritization
top_models = ['H-optimus-0', 'UNI', 'UNI2', 'Virchow2', 'fusion model', 'Prov-GigaPath']
# Custom sort order: top 5 models first, then others
all_models = newdf['Model Name'].unique()
custom_sort = top_models + [m for m in all_models if m not in top_models]

# Model performance table (shows all models)
st.header("Model Performance Table")
gb = GridOptionsBuilder.from_dataframe(newdf)
for col in numeric_columns:
    gb.configure_column(col, type=["numericColumn"], valueFormatter="Number(x).toFixed(3)")
gb.configure_grid_options(domLayout='autoWidth', enableRangeSelection=True)
grid_options = gb.build()
AgGrid(
    newdf,
    height=600,
    gridOptions=grid_options,
    theme='streamlit',
    enable_sorting=True,
    enable_filter=True
)

# Pivot table for Models vs Tasks (shows all models)
st.header(f"Models vs Tasks: {selected_metric}")
pivot_df = pd.pivot_table(newdf, index='Model Name', columns='Task', values=selected_metric)
styled_df = pivot_df.style.format("{:.3f}").background_gradient(cmap='Blues')
numRows = len(pivot_df)
height = (numRows + 1) * 40 + 5
st.dataframe(styled_df, height=height, use_container_width=True)

# Interactive visualization section
st.header("Interactive Visualizations")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Chart Selection")
    chart_type = st.selectbox("Chart Type", 
                              ["Bar Chart", "Area Chart", "Heatmap", "Line Chart", "Dot Chart", "Box Plot"],
                              key="chart_select")
with col2:
    st.subheader("Sort By")
    sort_by = st.selectbox("Sort By", ["Metric Value", "Model Name"], key="sort_select")

# Generate charts based on selection
if chart_type == "Bar Chart":
    try:
        chart = alt.Chart(newdf).mark_bar(opacity=0.8).encode(
            x=alt.X('Model Name', sort=custom_sort if sort_by == "Model Name" else '-y',
                    title=f"Model Name ({newdf['Model Name'].nunique()})",
                    axis=alt.Axis(grid=False, labelAngle=45, labelFontSize=10, labelLimit=0)),
            y=alt.Y(selected_metric, title=selected_metric, axis=alt.Axis(grid=True, gridColor="#D3D3D3")),
            color=alt.Color("Task", title=f"Task ({newdf['Task'].nunique()})", scale=alt.Scale(scheme='category20'),
                            legend=alt.Legend(orient='right', columns=2, labelFontSize=12)),
            tooltip=['Model Name', selected_metric, 'Task']
        ).properties(
            height=500, width=max(800, 30 * newdf['Model Name'].nunique())
        ).configure_axis(
            titleFontSize=16, labelFontSize=10
        ).configure_legend(
            titleFontSize=14, labelFontSize=12
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Bar Chart: {str(e)}")
elif chart_type == "Area Chart":
    try:
        chart = alt.Chart(newdf).mark_area(opacity=0.8).encode(
            x=alt.X('Model Name', sort=custom_sort if sort_by == "Model Name" else '-y',
                    title=f"Model Name ({newdf['Model Name'].nunique()})",
                    axis=alt.Axis(grid=False, labelAngle=45, labelFontSize=10, labelLimit=0)),
            y=alt.Y(selected_metric, title=selected_metric, axis=alt.Axis(grid=True, gridColor="#D3D3D3")),
            color=alt.Color("Task", title=f"Task ({newdf['Task'].nunique()})", scale=alt.Scale(scheme='category20'),
                            legend=alt.Legend(orient='right', columns=2, labelFontSize=12)),
            tooltip=['Model Name', selected_metric, 'Task']
        ).properties(
            height=500, width=max(800, 30 * newdf['Model Name'].nunique())
        ).configure_axis(
            titleFontSize=16, labelFontSize=10
        ).configure_legend(
            titleFontSize=14, labelFontSize=12
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Area Chart: {str(e)}")
elif chart_type == "Heatmap":
    try:
        base = alt.Chart(newdf).encode(
            x=alt.X('Model Name', sort=custom_sort, title=f"Model Name ({newdf['Model Name'].nunique()})",
                    axis=alt.Axis(labelAngle=45, labelFontSize=10, labelLimit=0)),
            y=alt.Y('Task', title=f"Task ({newdf['Task'].nunique()})", axis=alt.Axis(labelFontSize=10))
        )
        heatmap = base.mark_rect(stroke='white', strokeWidth=2).encode(
            color=alt.Color(selected_metric, scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title=selected_metric, labelFontSize=12))
        )
        text = base.mark_text(baseline='middle', fontSize=10).encode(
            text=alt.Text(selected_metric, format=".3f"),
            color=alt.condition(alt.datum[selected_metric] > 0.70, alt.value("black"), alt.value("white"))
        )
        chart = (heatmap + text).properties(
            height=max(500, 20 * newdf['Task'].nunique()),
            width=max(800, 30 * newdf['Model Name'].nunique())
        ).configure_axis(
            titleFontSize=16, labelFontSize=10
        ).configure_legend(
            titleFontSize=14, labelFontSize=12
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Heatmap: {str(e)}")
elif chart_type == "Line Chart":
    try:
        chart = alt.Chart(newdf).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X('Task', sort=None, title=f"Task ({newdf['Task'].nunique()})", axis=alt.Axis(labelAngle=45, labelFontSize=10, labelLimit=0)),
            y=alt.Y(selected_metric, title=selected_metric, axis=alt.Axis(grid=True, gridColor="#D3D3D3")),
            color=alt.Color("Model Name", title=f"Model Name ({newdf['Model Name'].nunique()})", sort=custom_sort,
                            scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(orient='right', columns=2, labelFontSize=12)),
            tooltip=['Task', selected_metric, 'Model Name']
        ).properties(
            height=550, width=max(800, 30 * newdf['Task'].nunique())
        ).configure_axis(
            titleFontSize=16, labelFontSize=10
        ).configure_legend(
            titleFontSize=14, labelFontSize=12
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Line Chart: {str(e)}")
elif chart_type == "Dot Chart":
    try:
        model_performance = newdf.groupby('Model Name')[selected_metric].mean().reset_index()
        sorted_models = model_performance.sort_values(by=selected_metric, ascending=False)['Model Name'].tolist()
        custom_sort_models = [m for m in custom_sort if m in sorted_models]  # Ensure sort respects data
        chart = alt.Chart(newdf).mark_circle(size=60).encode(
            x=alt.X(selected_metric, title=selected_metric, axis=alt.Axis(grid=True, gridColor="#D3D3D3")),
            y=alt.Y('Model Name', title=f"Model Name ({newdf['Model Name'].nunique()})", sort=custom_sort_models,
                    axis=alt.Axis(labelFontSize=10, labelLimit=0)),
            color=alt.Color("Model Name", title=f"Model Name ({newdf['Model Name'].nunique()})", sort=custom_sort,
                            scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(orient='right', columns=2, labelFontSize=12)),
            tooltip=[selected_metric, 'Model Name', 'Task']
        ).properties(
            height=750, width=800
        ).configure_axis(
            titleFontSize=16, labelFontSize=10
        ).configure_legend(
            titleFontSize=14, labelFontSize=12
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Dot Chart: {str(e)}")
elif chart_type == "Box Plot":
    try:
        chart = alt.Chart(newdf).mark_boxplot(size=40).encode(
            x=alt.X('Model Name', sort=custom_sort if sort_by == "Model Name" else '-y',
                    title=f"Model Name ({newdf['Model Name'].nunique()})",
                    axis=alt.Axis(labelAngle=45, labelFontSize=10, labelLimit=0)),
            y=alt.Y(selected_metric, title=selected_metric, axis=alt.Axis(grid=True, gridColor="#D3D3D3")),
            color=alt.Color("Model Category", title="Model Category", scale=alt.Scale(scheme='tableau20')),
            tooltip=['Model Name', selected_metric, 'Task', 'Model Category']
        ).properties(
            height=500, width=max(800, 30 * newdf['Model Name'].nunique())
        ).configure_axis(
            titleFontSize=16, labelFontSize=10
        ).configure_legend(
            titleFontSize=14, labelFontSize=12
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Box Plot: {str(e)}")

# Footer with lab and preprint information
st.markdown(
    """
    <div class='footer'>
        PathBench: A benchmarking platform for pathology foundation models.<br>
        Developed by the <a href='https://med.stanford.edu/gevaertlab.html' target='_blank'>Ogevaert Lab</a> at Stanford University.<br>
        Preprint: <a href='https://www.medrxiv.org/content/10.1101/2025.05.08.25327250v1' target='_blank'>Benchmarking Pathology Foundation Models</a>
    </div>
    """,
    unsafe_allow_html=True
)

# References for code inspiration
# https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
# https://docs.streamlit.io/develop/api-reference/data/st.dataframe
# https://www.sprinklr.com/blog/data-visualization-best-practices/
# https://www.chartexpo.com/blog/data-visualization-best-practices
