import streamlit as st

st.title("Streamlit Sandbox Connection Test")
st.write("If you can see this message, Streamlit's server architecture is operating perfectly!")

uploaded_file = st.file_uploader("Test file uploader UI mechanism", type=["pdf"])
if uploaded_file is not None:
    st.success(f"File metadata scanned: {uploaded_file.name}")