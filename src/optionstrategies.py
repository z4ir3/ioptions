"""
"""

import streamlit as st

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
                options=strnames
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
                # st.write("Inside the form")
                # Call or Put price
                cp = st.selectbox(
                    label = "Option type",
                    options = ["Call","Put"],
                    index = 0,
                    # placeholder = "Call or Put",
                    key = "option-type"
                )
                CP = "C" if cp == "Call" else "P"
                checkbox_val = st.checkbox("Form checkbox")
                # Every form must have a submit button.
                submitted = st.form_submit_button("Submit")


