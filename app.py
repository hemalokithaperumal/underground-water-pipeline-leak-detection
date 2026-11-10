
import streamlit as st

st.set_page_config(
    page_title="Underground Pipeline Leak Detection",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Underground Water Pipeline Leak Detection")
st.write(
    "A GraphSAGE-v2 based system for detecting "
    "and classifying underground water pipeline leaks "
    "using multi-sensor signals."
)

st.info("Welcome! The prediction system will be added next.")

st.subheader("Leak Classification")
st.write("The system classifies pipeline conditions into five categories:")
st.write("CC | GL | LC | NL | OL")
