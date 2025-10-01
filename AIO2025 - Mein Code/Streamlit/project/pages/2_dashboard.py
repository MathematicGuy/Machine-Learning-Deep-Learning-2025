import streamlit as st
import pandas as pd

data = pd.read_csv('../../data/movies.csv')
st.write(data)