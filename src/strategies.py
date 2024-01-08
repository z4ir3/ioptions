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

    # Chosen strategy (custom or predefined)
    if "chosen_strategy" not in st.session_state:
        st.session_state["chosen_strategy"] = None
    # else:
    #     st.session_state["chosen_strategy"] = st.session_state["chosen_strategy"]


    if "S" not in st.session_state: 
        st.session_state["S"] = 100
    if "q" not in st.session_state: 
        st.session_state["q"] = 0
    if "r" not in st.session_state: 
        st.session_state["r"] = 2
    if "TType" not in st.session_state: 
        st.session_state["TType"] = "Days"
    if "T" not in st.session_state: 
        st.session_state["T"] = 0




    
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

        try:
            st.write(  st.session_state["chosen_strategy"] )
        except:
            pass

        col1, col2 = st.columns(2)
        with col1:
            # Underlying Price 
            # S = st.number_input(
            st.number_input(
                label = "Underlying Price ($S$)",
                min_value = 0.1,
                format = "%f", 
                value = 100.0,
                key = "underlyingselected",
                help = "Price of the 'Underlying' asset of the Option(s)",
                on_change = _ss_S
            )
        with col2:
            # Dividend Yield
            # q = st.number_input(
            st.number_input(
                label = "Dividend Yield ($q$)",
                min_value = 0.0,
                max_value = None,
                format = "%f",
                value = 0.0,
                help = None,
                key = "divyieldselected",
                on_change = _ss_q
            )
        # Interest Rate Slider 
        # r = st.slider(
        st.slider(
            label = "Interest Rate (%) ($r$)", 
            min_value = 0.0,
            max_value = 8.0,
            value = 2.0, 
            help = None,
            key = "iratedselected",
            on_change = _ss_r
        )
        # st.session_state["r"] = st.session_state["r"] / 100

        # st.write(st.session_state["r"])





        st.write(st.session_state["chosen_strategy"])



        # if (st.session_state["chosen_strategy"] != "Custom strategy") and \
        #    (st.session_state["chosen_strategy"] is not None):
        #     col1, col2 = st.columns([1,2])
        #     with col1:
        #         # Expiration type
        #         st.session_state["TType"] = st.selectbox(
        #             label = "Expiration type",
        #             options = ["Days","Years"],
        #             index = 0,
        #             key = "dte-type"
        #         ) 
        #     with col2:
        #         # Expiration Slider 
        #         T = st.slider(
        #             label = f"Time-to-Expiration ({st.session_state["TType"]}) ($\\tau$)", 
        #             min_value = 0 if st.session_state["TType"] == "Days" else 0.0, 
        #             max_value = 1095 if st.session_state["TType"] == "Days" else float(3), 
        #             value = 182 if st.session_state["TType"] == "Days" else 0.50, 
        #             step = 1 if st.session_state["TType"] == "Days" else 0.05, 
        #             key = "slider-exp", 
        #             help = "Expiration of (non-custom) strategy", 
        #             # on_change = get_T(st.session_state["TType"], minvt, maxvt)
        #         )
        #         if st.session_state["TType"] == "Days": 
        #             T = T / 365



    # Get default strategies
    strnames = strategynames()
    with st.container():
        if (st.session_state["chosen_strategy"] == "Custom strategy"):
            col1, col2, col3 = st.columns([1,1,1])
        else:
            col1, col2, col3, col4 = st.columns([1,0.5,0.5,1])
        
        with col1:
            st.selectbox(
            # chosen_strategy = st.selectbox(
                label="Choose the strategy",
                options=strnames,
                index=None,
                key="strategyselected",
                on_change=_ss_chosen_strategy
            )
            
        if (st.session_state["chosen_strategy"] == "Custom strategy"):
            with col2:
                st.markdown("""
                    <p style="padding:6px;"</p>
                """,unsafe_allow_html=True)
                butt_reset = st.button(
                    label="Reset strategy",
                    type="secondary",
                    use_container_width=True
                )       
            with col3:
                butt_calculate = _calcbutton()

        else:
            with col2:
                # Expiration type
                # st.session_state["TType"] = st.selectbox(
                st.selectbox(
                    label = "Expiration type",
                    options = ["Days","Years"],
                    index = 0,
                    key = "ttypeselected",
                    on_change = _ss_ttype
                )        

            with col3:
                # Expiration Slider 
                # T = st.slider(
                # st.slider(
                #     label = f"Expiration ({st.session_state['st.session_state["TType"]']}) ($\\tau$)", 
                #     min_value = 0 if st.session_state["st.session_state["TType"]"] == "Days" else 0.0, 
                #     max_value = 1095 if st.session_state["st.session_state["TType"]"] == "Days" else float(3), 
                #     value = 182 if st.session_state["st.session_state["TType"]"] == "Days" else 0.50, 
                #     step = 1 if st.session_state["st.session_state["TType"]"] == "Days" else 0.05, 
                #     key = "expiryselected", 
                #     help = "Expiration of (non-custom) strategy", 
                #     on_change = _ss_expiry_slider
                # )
                st.number_input(
                    label = f"Expiration ({st.session_state['TType']}) ($\\tau$)", 
                    min_value = 0 if st.session_state["TType"] == "Days" else 0.0, 
                    max_value = 1095 if st.session_state["TType"] == "Days" else float(3), 
                    value = 182 if st.session_state["TType"] == "Days" else 0.50, 
                    step = 1 if st.session_state["TType"] == "Days" else 0.05, 
                    key = "expiryselected", 
                    help = "Expiration of the strategy", 
                    on_change = _ss_expiry_strategy
                )
                if st.session_state["TType"] == "Days": 
                    st.session_state["T"] = st.session_state["T"] / 365

            with col4:
                butt_calculate = _calcbutton()
                


    st.write(st.session_state["S"])
    st.write(st.session_state["q"])
    st.write(st.session_state["r"])

    # Create strategy class with the underlying price, 
    # the time-to-maturity, and the dividend yield
    Strategy = BSOptStrat(
        S = st.session_state["S"], 
        r = st.session_state["r"]/100, 
        q = st.session_state["q"]
    )



    if st.session_state["chosen_strategy"] == "Custom strategy":
        # Custom strategy

        with st.expander(label="Enter custom strategy", expanded=True):
            # if st.session_state["flag_add_option"] == 0

            #Must be like nested buttons 

            OEntries = _adding_options(
                CusOptData, 
                OEntries, 
                S, 
                # st.session_state["TType"]
            )
            st.write(OEntries)


    else:
        # Pre-defined strategy
        pass







    # else:
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         # fig = _plotoptions()
    #         #     Sens[ss], 
    #         #     CP, 
    #         #     K, 
    #         #     lcol = bscolors(ss),
    #         #     atmv = (ATM[ss]["x"], ATM[ss]["y"])
    #         # )
    #         # st.plotly_chart(fig, use_container_width=True)
    #         pass

    #     with col2:
    #         # fig = _plotstrategy()
    #         # st.plotly_chart(fig, use_container_width=True)
    #         pass



def _ss_S():
    st.session_state["S"] = st.session_state["underlyingselected"]

def _ss_q():
    st.session_state["q"] = st.session_state["divyieldselected"]

def _ss_r():
    st.session_state["r"] = st.session_state["irateselected"]

def _ss_ttype():
    st.session_state["TType"] = st.session_state["ttypeselected"]

def _ss_expiry_strategy():
    st.session_state["T"] = st.session_state["expiryselected"]

def _ss_chosen_strategy():
    st.session_state["chosen_strategy"] = st.session_state["strategyselected"]










def _calcbutton():
    """
    """
    st.markdown("""
        <p style="padding:6px;"</p>
    """,unsafe_allow_html=True)
    butt = st.button(
        label="Calculate",
        # help="",
        type="primary",
        use_container_width=True
    )
    return butt


def _adding_options(
    CusOptData: dict,
    OEntries: dict, 
    S: float, 
    # st.session_state["TType"]: str, 
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
                label = f"Expiration ({st.session_state['TType']}) ($\\tau$)",
                min_value = 1 if st.session_state["TType"] == "Days" else 0.03, 
                max_value = 1095 if st.session_state["TType"] == "Days" else float(3), 
                value = 182 if st.session_state["TType"] == "Days" else 0.50, 
                format = "%d" if st.session_state["TType"] == "Days" else "%f"
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
