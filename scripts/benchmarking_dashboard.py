import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from st_aggrid import AgGrid

# Load the DataFrame
df = pd.read_csv("~/Downloads/benchmarking/benchmarking_revised_wfusion.csv")

# Assuming df is your pandas DataFrame
# Define the mapping dictionary for ModelCategory
category_mapping = {
    'vision': 'VM',
    'vision_language': 'VLM',
    'pathology': 'Path-VM',
    'pathology_language': 'Path-VLM'
}
category_mapping2 = {
    'vision': 'general vision',
    'vision_language': 'general vision language',
    'pathology': 'pathology-specific vision model',
    'pathology_language': 'pathology-specific vision language model'
}

# Update ModelCategory column: replace specified values, keep others unchanged
df['Model Category Abbreviation'] = df['ModelCategory'].map(category_mapping).fillna(df['ModelCategory'])
df['ModelCategory'] = df['ModelCategory'].map(category_mapping2).fillna(df['ModelCategory'])

df = df.replace({None: np.nan})

# Set page configuration (wide layout, no sidebar needed)
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Add custom CSS to increase font sizes, brighten titles, style the selectbox, and increase dropdown values font size
st.markdown(
    """
    <style>
    /* Increase font size for the main title */
    h1 {
        font-size: 48px !important;
        font-weight: bold;
        color: #CD5C5C !important; /* Indian Red for better visibility */
    }
    /* Increase font size for headers */
    h2 {
        font-size: 36px !important;
        font-weight: bold;
        color: #CD5C5C !important; /* Indian Red for better visibility */
    }
    /* Increase font size for subheaders */
    h3 {
        font-size: 32px !important; /* Increased from 28px to 32px for prominence */
        font-weight: bold;
        color: #CD5C5C !important; /* Indian Red for better visibility */
    }
    /* Ensure the chart container spans the full width */
    div[data-testid="stVerticalBlock"] > div > div > div > div {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    /* Ensure the Altair chart itself uses the full width */
    .vega-embed {
        width: 100% !important;
        max-width: 100% !important;
    }
    /* Increase font sizes for Altair chart elements */
    .vega-embed text {
        font-size: 14px !important; /* Axis labels, legend labels, heatmap text */
    }
    .vega-embed .role-title text {
        font-size: 16px !important; /* Axis titles, legend titles */
    }
    /* Style the selectbox to make it larger */
    div[data-testid="stSelectbox"] {
        width: 100% !important;
        max-width: 500px !important; /* Increased width */
        margin: 10px 0 !important; /* Add vertical spacing */
    }
    div[data-testid="stSelectbox"] select {
        font-size: 18px !important; /* Larger font size for the selectbox text */
        padding: 10px !important; /* Increased padding for better appearance */
        height: 50px !important; /* Increased height */
        border-radius: 5px !important; /* Rounded corners for aesthetics */
        border: 2px solid #CD5C5C !important; /* Match the border color to the title color */
    }
    div[data-testid="stSelectbox"] label {
        font-size: 18px !important; /* Larger font size for the selectbox label */
        color: #CD5C5C !important; /* Match the label color to the title color */
    }
    /* Target the dropdown menu container and its items */
    div[role="listbox"] {
        font-size: 20px !important; /* Larger font size for dropdown options */
    }
    div[role="listbox"] div[data-baseweb="menu"] div {
        font-size: 20px !important; /* Ensure all text in the dropdown has larger font */
        padding: 10px !important; /* Add padding for better spacing */
        background-color: #f9f9f9 !important; /* Light background for debugging */
    }
    div[role="listbox"] div[data-baseweb="menu"] div:hover {
        background-color: #e0e0e0 !important; /* Hover effect for better UX */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title
st.title("PathBench: Foundation Models Benchmarking Dashboard")
st.header("Model Performance Table")

# Function to filter the DataFrame (unchanged)
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

# Display the filtered DataFrame
newdf = filter_dataframe(df)
st.dataframe(newdf)

# Create a Task vs Models DataFrame (pivot table)
pivot_df = pd.pivot_table(newdf, index='ModelName_short', columns='Task_short', values='balanced_accuracy')
styled_df = pivot_df.style\
    .format("{:.2f}")\
    .background_gradient(cmap='Blues')\
    .set_table_styles(
    [
        {'selector': 'tr', 'props': [('line-height', '4px')]},  # Adjust row height
        {'selector': 'td', 'props': [('width', '5px')]}   # Adjust column width
    ]
)

st.header("Models vs Tasks : Average Performance")

st.markdown("""
<style>
.stDataFrame > div > table > tbody > tr > td {
    height: 5px; /* Adjust the height as needed */
}
</style>
""", unsafe_allow_html=True)

# Display the pivot table
numRows = len(pivot_df)
height = (numRows + 1) * 35 + 3  # Modify the value to increase height
pd.set_option('display.max_colwidth', 50)
st.dataframe(styled_df, height=height)

# Move the plot selection to the main content area (above the "Plots" header)
# Use columns to control the width of the selectbox
col1, col2, col3 = st.columns([3, 1, 1])  # Adjusted ratios to give more space to col1
with col1:  # Place the selectbox in the first column
    st.subheader("Dashboard Plot Selection")
    chart_type = st.selectbox("Select Chart Type", ("Bar Chart", "Area Chart", "Heatmap", "Models Line Chart", "Models Dot Chart"))

# Plots section
st.header("Plots")

# Create a chart based on the selected type
if chart_type == "Bar Chart":
    # Verify required columns
    required_columns = ['ModelName_short', 'average_performance', 'Task_short']
    missing_columns = [col for col in required_columns if col not in newdf.columns]
    if missing_columns:
        st.error(f"Missing columns for Bar Chart: {missing_columns}")
    else:
        try:
            chart = alt.Chart(newdf).mark_bar(opacity=0.8).encode(
                x=alt.X('ModelName_short', sort='-y', title="Model Name",
                        axis=alt.Axis(grid=False, labelAngle=45, labelOverlap=False, labelLimit=0)),
                y=alt.Y('average_performance', title="Cumulative Average Performance",
                        axis=alt.Axis(grid=True, gridColor="#D3D3D3", gridDash=[2,2])),
                color=alt.Color("Task_short", title="Task", scale=alt.Scale(scheme='tableau20')),
                tooltip=['ModelName_short', 'average_performance', 'Task_short']
            ).properties(
                height=500
            ).configure_axis(
                titleFontSize=16,
                labelFontSize=14
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Bar Chart: {str(e)}")
elif chart_type == "Area Chart":
    # Verify required columns
    required_columns = ['ModelName_short', 'average_performance', 'Task_short']
    missing_columns = [col for col in required_columns if col not in newdf.columns]
    if missing_columns:
        st.error(f"Missing columns for Area Chart: {missing_columns}")
    else:
        try:
            chart = alt.Chart(newdf).mark_area(opacity=0.6).encode(
                x=alt.X('ModelName_short', sort='-y', title="Model Name",
                        axis=alt.Axis(grid=False, labelAngle=45, labelOverlap=False, labelLimit=0)),
                y=alt.Y('average_performance', title="Cumulative Average Performance",
                        axis=alt.Axis(grid=True, gridColor="#D3D3D3", gridDash=[2,2])),
                color=alt.Color("Task_short", title="Task", scale=alt.Scale(scheme='tableau20')),
                tooltip=['ModelName_short', 'average_performance', 'Task_short']
            ).properties(
                height=500
            ).configure_axis(
                titleFontSize=16,
                labelFontSize=14
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Area Chart: {str(e)}")
elif chart_type == "Heatmap":
    # Verify required columns
    required_columns = ['ModelName_short', 'Task_short', 'average_performance']
    missing_columns = [col for col in required_columns if col not in newdf.columns]
    if missing_columns:
        st.error(f"Missing columns for Heatmap: {missing_columns}")
    else:
        try:
            base = alt.Chart(newdf).encode(
                x=alt.X('ModelName_short', title="Model Name",
                        axis=alt.Axis(grid=False, labelAngle=45, labelOverlap=False, labelLimit=0)),
                y=alt.Y('Task_short', title="Task",
                        axis=alt.Axis(grid=False, labelOverlap=False, labelLimit=0)),
            )
            heatmap = base.mark_rect(stroke='white', strokeWidth=1).encode(
                color=alt.Color('average_performance',
                                scale=alt.Scale(scheme='viridis'),
                                legend=alt.Legend(title="Average Performance"))
            )
            color = (
                alt.when(alt.datum.average_performance > 0.70)
                .then(alt.value("black"))
                .otherwise(alt.value("white"))
            )
            text = base.mark_text(baseline='middle').encode(
                text=alt.Text('average_performance', format=".2f"),
                color=color
            )
            chart = (heatmap + text).configure_axis(
                titleFontSize=16,
                labelFontSize=14
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12
            ).configure_text(
                fontSize=14
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Heatmap: {str(e)}")
elif chart_type == "Models Line Chart":
    # Verify required columns
    required_columns = ['Task_short', 'average_performance', 'ModelName_short']
    missing_columns = [col for col in required_columns if col not in newdf.columns]
    if missing_columns:
        st.error(f"Missing columns for Models Line Chart: {missing_columns}")
    else:
        try:
            chart = alt.Chart(newdf).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X('Task_short', sort='-y', title="Task",
                        axis=alt.Axis(grid=False, labelAngle=45, labelOverlap=False, labelLimit=0)),
                y=alt.Y("average_performance", title="Average Performance",
                        axis=alt.Axis(grid=True, gridColor="#D3D3D3", gridDash=[2,2])),
                color=alt.Color("ModelName_short", title="Model Name", scale=alt.Scale(scheme='tableau20')),
                tooltip=['Task_short', 'average_performance', 'ModelName_short']  # Fixed syntax error
            ).transform_window(
                rank="rank()",
                sort=[alt.SortField("average_performance", order="descending")],
                groupby=["ModelName_short"]
            ).properties(
                height=550
            ).configure_axis(
                titleFontSize=16,
                labelFontSize=14
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Models Line Chart: {str(e)}")
elif chart_type == "Models Dot Chart":
    # Verify required columns
    required_columns = ['ModelName_short', 'average_performance', 'Task_short']
    missing_columns = [col for col in required_columns if col not in newdf.columns]
    if missing_columns:
        st.error(f"Missing columns for Models Dot Chart: {missing_columns}")
    else:
        # Handle missing values
        chart_df = newdf[required_columns].dropna()
        try:
            # Calculate the overall average performance for each ModelName_short
            model_performance = chart_df.groupby('ModelName_short')['average_performance'].mean().reset_index()
            
            # Sort models by average performance in descending order
            sorted_models = model_performance.sort_values(by='average_performance', ascending=False)['ModelName_short'].tolist()
            
            # Create the chart with the Y-axis sorted by performance
            chart = alt.Chart(chart_df).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X('average_performance', sort='-y', title="Average Performance",
                        axis=alt.Axis(grid=True, gridColor="#D3D3D3", gridDash=[2,2])),
                y=alt.Y('ModelName_short', title="Model Name", sort=sorted_models,
                        axis=alt.Axis(grid=False, labelOverlap=False, labelLimit=0)),
                color=alt.Color("ModelName_short", title="Model Name", scale=alt.Scale(scheme='tableau20')),
                tooltip=['average_performance', 'ModelName_short', 'Task_short']
            ).transform_window(
                rank="rank()",
                sort=[alt.SortField("average_performance", order="descending")],
                groupby=["Task_short"]
            ).properties(
                height=750
            ).configure_axis(
                titleFontSize=16,
                labelFontSize=14
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Models Dot Chart: {str(e)}")

# References
# https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
# https://docs.streamlit.io/develop/api-reference/data/st.dataframe