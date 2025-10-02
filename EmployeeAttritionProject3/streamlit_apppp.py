import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import random
import numpy as np
import os
import time

st.set_page_config(page_title="Employee Dashboard", layout="wide")

# ========== UPLOAD SECTION ==========
import subprocess  # ✅ Needed for subprocess.run

st.title("📁 Upload Employee CSV File")

uploaded_file = st.file_uploader("Upload a CSV file without 'Attrition'", type="csv")
if uploaded_file is None:
    st.info("📤 Please upload a CSV file to proceed.")
    st.stop()



input_path = "data/employee_dataset_3.csv"
with open(input_path, "wb") as f:
    f.write(uploaded_file.read())

st.toast("✅ File uploaded successfully!", icon="📁")
time.sleep(2)
# Run prediction script
try:
    result = subprocess.run(["python", "predict.py"], capture_output=True, text=True)

    if result.returncode != 0:
        st.error("❌ Prediction script failed.")
        st.code(result.stderr)
        st.stop()
    else:
        st.toast("✅ Prediction completed successfully!", icon="🤖")

    predicted_path = "predictions/predicted_attrition.csv"
    df = pd.read_csv(predicted_path)

    # DEBUG INFO - comment this later
    st.write("✅ Unique values in 'Attrition' column:", df['Attrition'].unique())
    st.write("✅ Number of employees who left:", df[df['Attrition'].astype(str).str.lower() == 'yes'].shape[0])

except Exception as e:
    st.error("❌ Error: Prediction script failed. Please check predict.py.")
    st.code(str(e))
    st.stop()

 # fallback to default
# ========== DISPLAY PREVIEW TABLE ==========
with st.container():
    st.markdown(
        """
        <div style="
            background-color: #2e2e2e;  /* matte grey */
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #444;
            margin-bottom: 1rem;
        ">
            <h4 style='color: #fafafa; margin: 0;'>📋 Data Preview</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(df, use_container_width=True)



# Fill missing columns (just like before)
updated = False
if 'Department' not in df.columns:
    df["Department"] = [random.choice(["HR", "Sales", "Tech", "R&D"]) for _ in range(len(df))]
    updated = True
if 'JobRole' not in df.columns:
    df["JobRole"] = [random.choice(["Manager", "Analyst", "Developer", "Sales Executive"]) for _ in range(len(df))]
    updated = True
if updated:
    df.to_csv("data/Employee_info2.csv", index=False)

# ========== SIDEBAR OPTIONS ==========
st.sidebar.title("🔍 Filter & View Options")

# Department filter
selected_department = st.sidebar.selectbox("Select Department", options=["All"] + sorted(df["Department"].unique().tolist()))
if selected_department != "All":
    df = df[df["Department"] == selected_department]

# Graph selection
# Add View Mode Selection
view_mode = st.sidebar.radio("Choose View Mode", ["Data Preview", "Graphs"])

graph_choice = st.sidebar.radio(
    # Add View Mode Selection

    "Select Graph to Display",
    (
        "KPIs",
        "Department-wise Count",
        "Job Role Distribution",
        "Age vs. Years at Company",
        "Job Satisfaction by Education",
        "Job Role Pie Chart",
        "Income vs Age",
        "Attrition Distribution",
        "Attrition by Department",
        "Monthly Income by Job Role",
        "Correlation Heatmap"
    )
)

# ========== DEFINE FUNCTIONS ==========

def show_kpis():
    st.title("📊 Employee Attrition Dashboard")
    print("Reached KPI section!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", df.shape[0])
    col2.metric("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

    total_employees = df.shape[0]
    employees_left = df[df['Attrition'].astype(str).str.lower() == 'yes'].shape[0]
    attrition_rate = (employees_left / total_employees) * 100 if total_employees else 0
    col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

def department_count():
    st.subheader("Department-wise Count")
    dept_chart = px.histogram(df, x="Department", color="Attrition", barmode="group")
    st.plotly_chart(dept_chart, use_container_width=True)

def job_role_distribution():
    st.subheader("Job Role Distribution")
    fig = px.bar(df, x="Department", y="MonthlyIncome", color="Department", title="Monthly Income by Department")
    st.plotly_chart(fig, use_container_width=True)

def age_vs_years():
    st.subheader("Age vs Years at Company")
    fig = px.line(df, x="Age", y="YearsAtCompany", title="Age vs. Years at Company")
    st.plotly_chart(fig, use_container_width=True)

def job_satisfaction_by_edu():
    st.subheader("Job Satisfaction by Education Level")
    fig = px.box(df, x="Education", y="JobSatisfaction", color="Education")
    st.plotly_chart(fig, use_container_width=True)

def job_role_pie():
    st.subheader("Job Role Distribution")
    fig = px.pie(df, names="JobRole", title="Job Role Distribution")
    st.plotly_chart(fig, use_container_width=True)

def income_vs_age():
    st.subheader("Income vs Age")
    fig = px.scatter(df, x="Age", y="MonthlyIncome", color="Attrition", hover_data=["JobRole"])
    st.plotly_chart(fig, use_container_width=True)

def attrition_distribution():
    st.subheader("Attrition Distribution")
    attr_counts = df['Attrition'].value_counts()
    fig = px.pie(names=attr_counts.index, values=attr_counts.values, 
                 title="Overall Attrition", color_discrete_sequence=["#00cc96", "#ff6361"])
    st.plotly_chart(fig, use_container_width=True)

def attrition_by_dept():
    st.subheader("Attrition by Department")
    filtered = df[df['Attrition'].str.lower() == 'yes']  # case-insensitive match

    if filtered.empty:
        st.warning("⚠️ No attrition records found in the dataset.")
        return

    attr_dept = filtered['Department'].value_counts().reset_index()
    attr_dept.columns = ['Department', 'Attrition Count']

    fig = px.bar(
        attr_dept,
        x='Department',
        y='Attrition Count',
        title="Number of Employees Who Left per Department",
        color='Attrition Count',
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig, use_container_width=True)


def income_by_jobrole():
    st.subheader("Monthly Income by Job Role")
    fig = px.box(df, x="JobRole", y="MonthlyIncome", color="Attrition",
                 color_discrete_map={"Yes": "red", "No": "green"})
    st.plotly_chart(fig, use_container_width=True)

def correlation_heatmap():
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    z = corr.values
    x = corr.columns.tolist()
    y = corr.columns.tolist()
    annot_text = [[f"{val:.2f}" for val in row] for row in z]
    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=annot_text,
        colorscale='YlGnBu',
        showscale=True,
        hoverinfo='z',
        font_colors=None
    )
    fig.update_layout(
        font=dict(size=12, family="Arial"),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(tickangle=45, showgrid=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(xgap=3, ygap=3)
    st.plotly_chart(fig, use_container_width=True)

# ========== RENDER BASED ON SELECTION ==========
if graph_choice == "KPIs":
    show_kpis()
elif graph_choice == "Department-wise Count":
    department_count()
elif graph_choice == "Job Role Distribution":
    job_role_distribution()
elif graph_choice == "Age vs. Years at Company":
    age_vs_years()
elif graph_choice == "Job Satisfaction by Education":
    job_satisfaction_by_edu()
elif graph_choice == "Job Role Pie Chart":
    job_role_pie()
elif graph_choice == "Income vs Age":
    income_vs_age()
elif graph_choice == "Attrition Distribution":
    attrition_distribution()
elif graph_choice == "Attrition by Department":
    attrition_by_dept()
elif graph_choice == "Monthly Income by Job Role":
    income_by_jobrole()
elif graph_choice == "Correlation Heatmap":
    correlation_heatmap()