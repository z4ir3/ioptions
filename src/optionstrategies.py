"""
"""

import streamlit as st

import pandas as pd
import numpy as np

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objs as go 

from src.utils import streategynames



def dbpage_strategies():
    """
    """
    # Page title
    st.title("Option Strategies payoff calculator")
    # st.write("---")

    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            # Underlying Price 
            S = st.number_input(
                label = "Underlying Price ($S$)",
                min_value = 0.1,
                format = "%f", 
                value = 100.0,
                # placeholder = 100, #"Enter Underlying price",
                help = "Price of the 'Underlying' of the Option(s)",
                # key = "strike"
                # on_change=
            )
        with col2:
            # Dividend Yield
            q = st.number_input(
                label = "Dividend Yield ($q$)",
                min_value = 0.0,
                max_value = None,
                format = "%f",
                value = 0.0,
                help = None,
                key = "div-yield"
                # on_change=
            )
        col1, col2 = st.columns([1,2])
        with col1:
            # Expiration type
            TType = st.selectbox(
                label = "Expiration type",
                options = ["Days","Years"],
                index = 0,
                key = "dte-type"
            ) 
        with col2:
            # Expiration Slider 
            T = st.slider(
                label = f"Time-to-Expiration ({TType}) ($\\tau$)", 
                min_value = 0 if TType == "Days" else 0.0, 
                # max_value = 1825 if TType == "Days" else float(5), 
                max_value = 1095 if TType == "Days" else float(3), 
                value = 182 if TType == "Days" else 0.50, 
                # value = 90 if TType == "Days" else 0.25, 
                step = 1 if TType == "Days" else 0.05, 
                # format = None, 
                key = "slider-exp", 
                help = "Expiration of (non-custom) strategy", 
                # on_change = get_T(TType, minvt, maxvt)
            )
            if TType == "Days": 
                T = T / 365
        # Interest Rate Slider 
        r = st.slider(
            label = "Interest Rate (%) ($r$)", 
            min_value = 0.0,
            max_value = 8.0,
            value = 2.0, 
            # key = "slider-irate", 
            help = None, 
            # on_change = get_T(TType, minvt, maxvt)
        )
        r = r / 100


    # Get default strategies
    strnames = streategynames()
    with st.container():
        # col1, col2, col3, col4 = st.columns([1,0.5,0.5,1])
        # with col1:
        #     st.write("Choose the strategy")

        col1, col2, col3, col4 = st.columns([1,0.5,0.5,1])
        with col1:
            chosen_strategy = st.selectbox(
                label="Choose the strategy",
                options=strnames,

            )
        if chosen_strategy == "Custom strategy":
            with col2:
                st.markdown("""
                    <p style="padding:6px;"</p>
                """,unsafe_allow_html=True)
                st.button(
                    label="Add Option",
                    # help="",
                    type="secondary",
                    use_container_width=True
                )
            with col3:
                st.markdown("""
                    <p style="padding:6px;"</p>
                """,unsafe_allow_html=True)
                st.button(
                    label="Reset strategy",
                    # help="",
                    type="secondary",
                    use_container_width=True
                )
        with col4:
            st.markdown("""
                <p style="padding:6px;"</p>
            """,unsafe_allow_html=True)
            st.button(
                label="Calculate",
                # help="",
                type="primary",
                use_container_width=True
            )

        
    if chosen_strategy == "Custom strategy":

        with st.expander(label="Enter strategy", expanded=True):        
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
            """)
            with st.form("Enter Option"):
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                # st.write("Inside the form")
                # Call or Put price
                with col1:
                    cpo = st.selectbox(
                        label = "Option type",
                        options = ["Call","Put"],
                        index = 0,
                        # placeholder = "Call or Put",
                        key = "option-type"
                    )
                    CPO = "C" if cpo == "Call" else "P"
                with col2:
                    # Underlying Price 
                    Ko = st.number_input(
                        label = "Strike Price ($K$)",
                        min_value = 0.1,
                        format = "%f", 
                        value = S
                    )
                with col3:
                    # Expiration 
                    To = st.number_input(
                        label = f"Expiration ({TType}) ($\\tau$)",
                        min_value = 1 if TType == "Days" else 0.03, 
                        max_value = 1095 if TType == "Days" else float(3), 
                        value = 182 if TType == "Days" else 0.50, 
                        format = "%d" if TType == "Days" else "%f"
                    )
                with col4:
                    # Volatility
                    Vo = st.number_input(
                        label = f"Volatility (%) ($\\sigma$)",
                        min_value = 5.0, 
                        max_value = 100.0, 
                        value = 30.0, 
                        format = "%f"
                    )
                    Vo = Vo / 100
                with col5:
                    # Quantity
                    Qo = st.number_input(
                        label = f"Net Position",
                        min_value = None, 
                        max_value = None, 
                        value = +1, 
                        format = "%d",
                        help = "Long minus short (nominal) position"
                    )
                with col6:
                    st.markdown("""
                        <p style="padding:6px;"</p>
                    """,unsafe_allow_html=True)
                    submitted = st.form_submit_button(
                        label="Submit",
                        use_container_width=True
                    )

    else:


        col1, col2 = st.columns(2)
        with col1:
            # fig = _plotoptions()
            #     Sens[ss], 
            #     CP, 
            #     K, 
            #     lcol = bscolors(ss),
            #     atmv = (ATM[ss]["x"], ATM[ss]["y"])
            # )
            # st.plotly_chart(fig, use_container_width=True)
            pass

        with col2:
            # fig = _plotstrategy()
            # st.plotly_chart(fig, use_container_width=True)
            pass

def _plotoptions():
    """
    """
    # # Create figure
    # fig = go.Figure()

    # # Creating a first trace for the horizontal zero line
    # fig.add_trace(
    #     go.Scatter(
    #         x = data.index, 
    #         y = np.repeat(0,len(data.index)), 
    #         name = None, 
    #         line_dash = "longdash",
    #         line_color = "#C6C6C6",
    #         line_width = 1,
    #         showlegend = False,
    #         hoverinfo = "none"
    #     )
    # )
    pass



def _plotstrategy():
    """
    """
    pass
