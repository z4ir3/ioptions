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
    # st.write("---")

    # # Chosen strategy (custom or predefined)
    # if "chosen_strategy" not in st.session_state:
    #     st.session_state["chosen_strategy"] = None
    
    # Flag: True is the "Add option" button is pushed, False otherwise
    if "flag_add_option" not in st.session_state:
        st.session_state["flag_add_option"] = False

    # # Flag: True is the "Reset strategy" button is pushed, False otherwise
    # if "flag_reset_strategy" not in st.session_state:
    #     st.session_state["flag_reset_strategy"] = False

    if "addopt_times" not in st.session_state:
        st.session_state["addopt_times"] = -1
    
    # Dictionary of options data for the custom strategy
    CusOptData = dict()
    OEntries = dict()



    # Sidebar data
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
    strnames = strategynames()
    with st.container():
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            chosen_strategy = st.selectbox(
                label="Choose the strategy",
                options=strnames,
            )
            # st.session_state["chosen_strategy"] = chosen_strategy

        if chosen_strategy == "Custom strategy":
            # with col2:
            #     st.markdown("""
            #         <p style="padding:6px;"</p>
            #     """,unsafe_allow_html=True)
            #     st.button(
            #         label="Add Option",
            #         # help="",
            #         type="secondary",
            #         use_container_width=True
            #     )
            with col2:
                st.markdown("""
                    <p style="padding:6px;"</p>
                """,unsafe_allow_html=True)
                st.button(
                    label="Reset strategy",
                    # help="",
                    type="secondary",
                    use_container_width=True
                )

        with col3:
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


        with st.expander(label="Enter custom strategy", expanded=True):  
            # if st.session_state["flag_add_option"] == 0

            #Must be like nested buttons 


            OEntries = _adding_options(CusOptData, OEntries, S, TType)
            st.write(OEntries)



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




def _adding_options(
    CusOptData: dict,
    OEntries: dict, 
    S: float, 
    TType: str, 
):
    """
    """
    if len(CusOptData.keys()) == 0:
        CusOptData[st.session_state["addopt_times"]+1] = dict()


    else:
        # _validate_options(K)

        # CusOptData[st.session_state["addopt_times"]+1] = dict()
        pass

    st.write(st.session_state["addopt_times"])

    
    # Updating number of times the "Add option" button is pushed
    st.session_state["addopt_times"] = st.session_state["addopt_times"]+1
          
    st.write(st.session_state["addopt_times"])

    with st.form("Enter Option"):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            # Call or Put price
            cpo = st.selectbox(
                label = "Option 1",
                options = ["Call","Put"],
                index = 0,
                # key = "option-type"
            )
            CPO = "C" if cpo == "Call" else "P"
        with col2:
            # Strike Price 
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
            # Quantity (Net Position)
            NP = st.number_input(
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
                label="Add Option",
                use_container_width=True
            )

        if submitted:

            st.session_state["flag_add_option"] = st.session_state["addopt_times"]

            # Save the current entries
            # As soon as the "Add options" button is pushed and new entry data appear, they are saved. 
            # This is required due to the possibility (quite likely) that the user changes some value 
            # in previous options before or after pushing the "Calculate" button.
            # If so, the corresponding value shall be stored correctly in the associated option data dictionary.
            # Therefore, we create a dynamic dictionary of entries whose values are going 
            # to be reviewed every time the "Calculate" button is pushed, so than right inserted values are taken.
            # st.write(st.session_state["addopt_times"])

            OEntries[st.session_state["addopt_times"]] = {
                "CP": cpo,
                "K": Ko, 
                "T": To,
                "v": Vo,
                "NP": NP
            }


            return OEntries

# def _validate_options(   
#     CusOptData: dict,
# ):
#     """
#     """
#     CusOptData[st.session_state["addopt_times"]]["K"] = {
#     "CP": cpo,
#     "K": Ko, 
#     "T": To,
#     "v": Vo,
#     "NP": NP











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
