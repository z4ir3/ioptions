"""
src.pricing.py
"""
import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objs as go 

from src.utils import get_Smax, get_Smin, bscolors

from models.blackscholes import BlackScholesCall, BlackScholesPut
from models.black import BlackCall, BlackPut


def dbpage_pricing(
    nss: int = 80,
    sensname: list = ["Price","Delta","Gamma","Vega","Theta","Rho"],#,"Lambda"],
    rnd: int = 6
) -> None:
    """
    """
    # Page title
    st.title("Options Pricing Models")
    # # Copyright
    # st.markdown("""
    #     <h6>An app made by  
    #         <!-- <a href='https://github.com/z4ir3'> -->
    #         <a href="http://leonardorocchi.info/">
    #             <b>z4ir3</b>
    #         </a>
    #     </h6>
    # """, unsafe_allow_html=True)
    # Hiding "Made with Streamlit message"
    st.write('''
        <style>
            footer {visibility:hidden;}
        </style>
        ''',
        unsafe_allow_html = True
    )

    with st.sidebar:
        # Main options data

        col1, col2 = st.columns([1,1])
        with col1:
            ostyle = st.selectbox(
                label = "Option style",
                options = ["European","American"],
                index = None,
                placeholder = "European or American",
                key = "option-style"
            )
        with col2:
            # Call or Put price
            isdisabled = True if ostyle == "American" else False
            underlying_type = st.selectbox(
                label = "Underlying type",
                options = ["Stock","Index"],
                index = None,
                placeholder = "Stock or Index Index" if ostyle == "European" else "Stock",
                disabled = isdisabled,
                key = "underlying-style"
            )
            if isdisabled:
                underlying_type = "Stock"

        with st.container():
            # Call or Put price
            cp = st.selectbox(
                label = "Option type",
                options = ["Call","Put"],
                index = None,
                placeholder = "Call or Put",
                key = "option-type"
            )
            CP = "C" if cp == "Call" else "P" 
        col1, col2 = st.columns([1,1])
        with col1:
            # Strike price 
            K = st.number_input(
                label = "Option strike ($K$)",
                min_value = 0.1,
                format = "%f", 
                value = None, #100.0,
                placeholder = "Enter Strike price",
                help = "'Exercise' price of the option",
                key = "strike"
            ) 
        with col2:
            # Dividend Yield
            q = st.number_input(
                label = "Dividend Yield (%) ($q$)",
                min_value = 0.0,
                max_value = None,
                format = "%f",
                value = 0.0, #if ostyle == "European" else None,
                help = "Annual dividend yield stock return",
                disabled = True if (ostyle == "American" or underlying_type == "Index") else False,
                key = "div-yield"
            )
            q = q / 100

    cd1 = ostyle in {"European","American"} 
    cd2 = underlying_type in {"Stock","Index"} 
    cd3 = cp in {"Call","Put"}
    cd4 = K is not None 
    if (cd1 and cd2 and cd3 and cd4):

        # Print of the model pricing used 
        if ostyle == "European" and underlying_type == "Stock":
            st.subheader("Black-Scholes model")
        elif underlying_type == "Index":
            st.subheader("Black model")
        elif ostyle == "American":
            st.subheader("Binomial-Tree Model (Cox-Ross-Rubinstein)")
            st.write("...to be implemented yet")
            return 0

        # Rest of widgets: expiration, volatility, and interest rate
        col1, col2, col3, col4 = st.columns([0.5,1,1,0.5], gap="small") 
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
                label = "Years to Expiration ($\\tau$)" if TType == "Years" else "Days to Expiration ($\\tau$)",
                min_value = 0 if TType == "Days" else 0.0, 
                max_value = 1095 if TType == "Days" else float(3), 
                value = 182 if TType == "Days" else 0.50, 
                step = 1 if TType == "Days" else 0.10, 
                key = "slider-exp", 
                help = None
            )
            if TType == "Days": 
                T = T / 365
        with col3:
            # Volatilty Slider 
            v = st.slider(
                label =  "Volatility (%) ($\sigma$)", 
                min_value = 1.0,
                max_value = 99.0,
                value = 30.0, 
                step = 1.0, 
                # format = None, 
                key = "slider-vola", 
                help = "Implied Volatility", 
            )
            v = v / 100
        with col4:
            # Interes Rate Slider 
            r = st.slider(
                label = "Interest Rate (%) ($r$)", 
                min_value = 0.0,
                max_value = 5.0,
                value = 1.0, 
                key = "slider-irate", 
                help = "Risk-free rate"
            )
            r = r / 100

        # Main calculations 

        # Set up Options
        uset = np.linspace(get_Smin(K),get_Smax(K),nss)
        if (ostyle == "European") and (underlying_type == "Stock"):
            if CP == "C":
                Options = [BlackScholesCall(S=s, K=K, T=T, r=r, v=v, q=q) for s in uset]
            else:
                Options = [BlackScholesPut(S=s, K=K, T=T, r=r, v=v, q=q) for s in uset]
        elif (ostyle == "European") and (underlying_type == "Index"):
            if CP == "C":
                Options = [BlackCall(S=s, K=K, T=T, r=r, v=v) for s in uset]
            else:
                Options = [BlackPut(S=s, K=K, T=T, r=r, v=v) for s in uset]


            

        Sens = dict()
        for s in sensname: 
            # grk = [o.greeks(grk=s, rnd=rnd) for o in Options]
            grk = [o.greeks(grk=s) for o in Options]
            Sens[s] = pd.Series(grk, index=uset, name=s)

        # Save ATM points for metric and to be passed in plot functions
        ATM = {k: dict() for k in sensname}
        with st.container():
            atms = st.columns(len(sensname))        
            for idx, s in enumerate(sensname):
                # Save ATM points
                atmidx = np.argmin(pd.Series(Sens[s].index).apply(lambda x: abs(x - K)))
                ATM[s]["x"] = [Sens[s].index[atmidx]]
                ATM[s]["y"] = [Sens[s].values[atmidx]]
                with atms[idx]:
                    st.metric(
                        label = f"ATM {s}",
                        value = f"{ATM[s]['y'][0]:.2f}",
                        help = None
                    )

        # Price and Delta
        with st.container():
            plot1, plot2 = st.columns(2) #[1,1,1], gap="small") 
            with plot1:   
                ss = "Price"
                fig = _plotgreeks(
                    Sens[ss], 
                    CP, 
                    K, 
                    lcol = bscolors(ss),
                    atmv = (ATM[ss]["x"], ATM[ss]["y"])
                )
                st.plotly_chart(fig, use_container_width=True)
            with plot2:
                ss = "Delta"
                fig = _plotgreeks(
                    Sens[ss], 
                    CP, 
                    K, 
                    lcol = bscolors(ss),
                    atmv = (ATM[ss]["x"], ATM[ss]["y"]),
                    yaxside = "right"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Gamma and Vega
        with st.container():
            plot1, plot2 = st.columns(2)
            with plot1:   
                ss = "Gamma"
                fig = _plotgreeks(
                    Sens[ss], 
                    CP, 
                    K, 
                    lcol = bscolors(ss),
                    atmv = (ATM[ss]["x"], ATM[ss]["y"])
                )
                st.plotly_chart(fig, use_container_width=True)
            with plot2:
                ss = "Vega"
                fig = _plotgreeks(
                    Sens[ss], 
                    CP, 
                    K, 
                    lcol = bscolors(ss),
                    atmv = (ATM[ss]["x"], ATM[ss]["y"]),
                    yaxside = "right"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Theta and Rho
        with st.container():
            plot1, plot2 = st.columns(2)
            with plot1: 
                ss = "Theta"
                fig = _plotgreeks(
                    Sens[ss], 
                    CP, 
                    K, 
                    lcol = bscolors(ss),
                    atmv = (ATM[ss]["x"],ATM[ss]["y"]),
                    xlab = True
                )
                st.plotly_chart(fig, use_container_width=True)
            with plot2:
                # ss = "Lambda"
                ss = "Rho"
                fig = _plotgreeks(
                    Sens[ss], 
                    CP, 
                    K, 
                    lcol = bscolors(ss),
                    atmv = (ATM[ss]["x"],ATM[ss]["y"]),
                    yaxside = "right", 
                    xlab = True
                )
                st.plotly_chart(fig, use_container_width=True)


def _plotgreeks(
    data: pd.Series,
    CP: str,
    K: float,
    lcol: str,    
    atmv: tuple | None = None,
    yaxside: str = "left",
    gridcolor: str = "#EEF4F4",
    xlab: bool = False
):
    """
    """
    # Create figure
    fig = go.Figure()

    # Creating a first trace for the horizontal zero line
    fig.add_trace(
        go.Scatter(
            x = data.index, 
            y = np.repeat(0,len(data.index)), 
            name = None, 
            line_dash = "longdash",
            line_color = "#C6C6C6",
            line_width = 1,
            showlegend = False,
            hoverinfo = "none"
        )
    )

    # Adding option data trace
    fig.add_trace(
        go.Scatter(
            x = data.index,
            y = data.values, 
            name = data.name,
            line = dict(
                color = lcol,
                width = 1.8, 
                dash = "solid"
            ),
            showlegend = False, 
        )
    )

    # Label x-axis
    if xlab:
        xlabel = f"Underlying S (K={K})"
    else:
        xlabel = None

    # legend position
    if (CP == "P") and (data.name in ["Price","Lambda"]):
        legpos = dict(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99)
    elif (CP == "C") and (data.name == "Lambda"):
        legpos = dict(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99)
    elif data.name == "Theta":
        legpos = dict(yanchor = "bottom", y = 0.01, xanchor = "left", x = 0.01)
    else: 
        legpos = dict(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01)

    # Update layout
    fig.update_layout(
        # title = {"text": "", "font_size": 22},
        xaxis = dict(title = xlabel),
        yaxis = dict(title = f"{data.name}", side = yaxside),
        hovermode = "x",  
        hoverlabel = dict(
            #bgcolor = "white",
            font_size = 14,
            # font_family = "Rockwell"
        ),
        autosize = True,
        legend = legpos,
        plot_bgcolor = "#E6E6E6",
        legend_bgcolor = "#E6E6E6",
        legend_font_color = "#000000",
        legend_borderwidth = 0,
        margin_pad = 0, 
        width = 500,  # Specify the width of the plot in pixels
        height = 210,  # Specify the height of the plot in pixels
        margin = dict(l=0, r=0, t=32, b=0)  # Set margins around the plot
    )
    fig.update_xaxes(
        # zeroline = False, # not working 
        showgrid = True,
        gridcolor = gridcolor
    )
    fig.update_yaxes(
        showgrid = True,
        gridcolor = gridcolor
        # griddash="dot",
        # title_standoff=100
    )
    # Plot a marker for the ATM data 
    fig.add_trace(
        go.Scatter(
            x = atmv[0],
            y = atmv[1], 
            name = f"ATM {data.name}",
            marker = dict(
                color = "black",
                size = [8]
            ),
            showlegend = True #True
        )
    )
    
    return fig
