# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Title for the app
st.set_page_config(page_title="UTS & Elongation Predictor", layout="centered", initial_sidebar_state="expanded")
st.title("🔬 UTS & Elongation Predictor")
st.markdown("""This app predicts Ultimate Tensile Strength (UTS) and Elongation based on layer thickness and pattern.
It also displays the corresponding stress-strain graph.""")

# Load and clean datasets
@st.cache_data
def load_data():
    mdsp_df = pd.read_excel("MDSP Dataset.xlsx")
    uts_df = pd.read_excel("UTS_Elongation.xlsx", skiprows=2, usecols="B:E")
    uts_df.columns = ['Thickness', 'Pattern', 'UTS', 'Elongation']
    uts_df.dropna(inplace=True)
    return mdsp_df, uts_df

mdsp_df, uts_df = load_data()

# Encode 'Pattern' to numbers
label_encoder = LabelEncoder()
uts_df['Pattern_Code'] = label_encoder.fit_transform(uts_df['Pattern'])

# Prepare data for model
X = uts_df[['Thickness', 'Pattern_Code']]
y = uts_df[['UTS', 'Elongation']]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X, y)

# --- USER INPUT FROM STREAMLIT ---
thickness_options = sorted(uts_df['Thickness'].unique())
pattern_options = sorted(uts_df['Pattern'].unique())

user_thickness = st.selectbox("Select Layer Thickness (mm):", thickness_options)
user_pattern = st.selectbox("Select Pattern Type:", pattern_options)

if st.button("Predict UTS and Elongation"):
    user_pattern_code = label_encoder.transform([user_pattern])[0]
    user_input = [[user_thickness, user_pattern_code]]

    predicted_uts, predicted_elongation = model.predict(user_input)[0]

    st.success(f"\n**Predicted UTS:** {predicted_uts:.2f} MPa")
    st.success(f"**Predicted Elongation:** {predicted_elongation:.2f}%")

    # Filter matching data for stress-strain graph
    filtered_df = mdsp_df[(mdsp_df['Thickness'] == user_thickness) & (mdsp_df['Pattern'] == user_pattern)]

    if not filtered_df.empty:
        fig, ax = plt.subplots()
        plt.style.use('dark_background')
        ax.plot(filtered_df['Strain'], filtered_df['Stress'], marker='o', color='cyan', label='Stress-Strain')
        ax.set_title(f"Stress-Strain Curve\nPattern: {user_pattern}, Thickness: {user_thickness} mm", fontsize=12)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (MPa)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No stress-strain data available for this combination.")
