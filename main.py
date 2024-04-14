import streamlit as st
import os
import matplotlib.pyplot as plt

# DARK COLORS
TEXT_COLOR = "white"
BG_COLOR = "black"

plt.rcParams["axes.facecolor"] = plt.rcParams["figure.facecolor"] = BG_COLOR
plt.rcParams["text.color"] = TEXT_COLOR
plt.rcParams["axes.labelcolor"] = TEXT_COLOR
plt.rcParams["axes.titlecolor"] = TEXT_COLOR
plt.rcParams["xtick.color"] = TEXT_COLOR
plt.rcParams["ytick.color"] = TEXT_COLOR
plt.rcParams.update({
	"axes.grid" : True,
	"grid.color": "green",
	"grid.alpha": 0.35,
	"grid.linestyle": (0, (10, 10)),
})

# BETTER SIZES
# DEFAULT_W, DEFAULT_H = (10, 6)
# plt.rcParams["figure.figsize"] = [DEFAULT_W, DEFAULT_H]
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 90

# Set page to wide mode
st.set_page_config(layout="wide")



# Custom CSS styles
st.markdown(
    """
    <style>
    /* Remove padding and margin for main title */
    .block-container {
        padding-top: 20px !important;
        margin-top: 0px !important;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }
    """,
    unsafe_allow_html=True
)


project_mapping = {
    "SCAL": "scal",
    "Economic Evaluation": "economic_evaluation",
    # Add more project mappings as needed
}

def main():
    # Sidebar section
    st.sidebar.title('Module Navigation')
    selected_project = st.sidebar.radio('Select Module', project_mapping.keys())
    
    
    # Project Seciton
    project_path = os.path.join('Modules', project_mapping[selected_project], 'app.py')
    if os.path.exists(project_path):
        with open(project_path, 'r') as f:
            exec(f.read())
    else:
        st.error("Project not found.")

if __name__ == "__main__":
    main()
