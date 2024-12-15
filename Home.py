import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

st.set_page_config(
    page_title="ML Model Generator and Simulation",
    page_icon="🤖",
)

st.markdown("<h1 style='text-align: center;'>MODELING AND SIMULATION - CSEC 413</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Modeling and Simulation with Python</h2>", unsafe_allow_html=True)
# st.markdown("<h5 style='text-align: center;'>GROUP 12</h5>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This project will explore the concepts of modeling and simulation using Python. The project will involve creating synthetic data and applying it to various modeling and simulation techniques. The goal is to gain hands-on experience with Python libraries and tools commonly used for modeling and simulation tasks. </p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.text("Member 1:          Sayson, Nestor Jr. B.")
st.text("Member 2:          Estadilla, Andrea Krystel T.")
st.text("Member 3:          Ballerda, Carlo James D.")
st.text("Section:           BSCS 4B")
st.text("Instructor:        Mr. Allan Ibo Jr.")
st.divider()