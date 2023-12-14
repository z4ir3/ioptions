"""
"""
import streamlit as st

import pandas as pd
import numpy as np

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objs as go 

from src.utils import strategynames
# from src.utils import get_Smax, get_Smin, bscolors
from models.optionstrategy import BSOptStrat


def dbpage_strategies():
    """
    """
    # Page title
    st.title("Option Strategies payoff calculator")
    # Copyright
    st.markdown("""
        <h6>An app made by  
            <!-- <a href='https://github.com/z4ir3'> -->
            <a href="http://leonardorocchi.info/">
                <b>z4ir3</b>
            </a>
        </h6>
    """, unsafe_allow_html=True)
    # Hiding "Made with Streamlit message"
    st.write('''
        <style>
            footer {visibility:hidden;}
        </style>
        ''',
        unsafe_allow_html = True
    )

    st.write(" ")

    st.subheader(":hourglass_flowing_sand: Work in progress...")
    