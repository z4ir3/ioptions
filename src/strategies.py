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
    
    st.write(" ")

    st.subheader(":hourglass_flowing_sand: Under development...")
    