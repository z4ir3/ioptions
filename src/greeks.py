"""
"""
import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objs as go 

from src.utils import get_Smax, get_Smin, bscolors
from models.blackscholes import BSOption



def dbpage_greeks(
    nss: int = 80,
    sensname: list = ["Price","Delta","Gamma","Vega","Theta","Lambda"],
    rnd: int = 6
) -> None:
    """
    """
    # Page title
    st.title("Black-Scholes Option Greeks")

    par1, par2, par3, par4 = st.columns([1,1,1,0.5], gap="small") 
    with par1:
        # Call or Put price
        cp = st.selectbox(
            label = "Option type",
            options = ["Call","Put"],
            index = None,
            placeholder = "Call or Put",
            key = "option-type"
        )
        CP = "C" if cp == "Call" else "P" 
    with par2:
        # Strike price 
        K = st.number_input(
            label = "Option strike ($K$)",
            min_value = 0.1,
            format = "%f", 
            value = None, #100.0,
            placeholder = "Enter Strike price",
            help = "'Exercise' price of the option",
            key = "strike"
            # on_change=
        ) 
    with par3:
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
    with par4:
        # Expiration type
        TType = st.selectbox(
            label = "Expiration type",
            options = ["Days","Years"],
            index = 0,
            key = "dte-type"
        ) 

    st.write(CP)
    st.write(K)
    st.write(TType)


    with st.sidebar:
        with st.container():
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
                help = None, 
                # on_change = get_T(TType, minvt, maxvt)
            )
            if TType == "Days": 
                T = T / 365

        col1, col2 = st.columns(2)
        with col1:
            # Volatilty Slider 
            v = st.slider(
                label =  "Volatility (%) ($\sigma$)", 
                min_value = 1.0,
                max_value = 99.9,
                value = 30.0, 
                step = 1.0, 
                # format = None, 
                key = "slider-vola", 
                help = None, 
                # on_change = get_T(TType, minvt, maxvt)
            )
            v = v / 100
        with col2:
            # Interes Rate Slider 
            # r = st.sidebar.slider(
            r = st.slider(
                label = "Interest Rate (%) ($r$)", 
                min_value = 0.0,
                max_value = 8.0,
                value = 2.0, 
                key = "slider-irate", 
                help = None, 
                # on_change = get_T(TType, minvt, maxvt)
            )
            r = r / 100

    if (cp is not None) and (K is not None):
      
        # Call/Put and Strike inserted

        # Set up Options
        uset = np.linspace(get_Smin(K),get_Smax(K),nss)

        st.write(get_Smin(K))

        Options = [BSOption(CP=CP, S=s, K=K, T=T, r=r, v=v, q=q) for s in uset]

        Sens = dict()
        for s in sensname: 
            grk = [o.greeks(grk=s, rnd=rnd) for o in Options]
            Sens[s] = pd.Series(grk, index=uset, name=s)
        # st.write(Sens)
        # st.write(options[40].greeks()["Price"])

        # Save ATM points for metric and to be passed in plot functions
        ATM = {k: dict() for k in sensname}
        with st.container():
            atms = st.columns(len(sensname))        
            for idx, s in enumerate(sensname):
                # Save ATM points
                atmidx = np.argmin(pd.Series(Sens[s].index).apply(lambda x: abs(x - K)))
                ATM[s]["x"] = [Sens[s].index[atmidx]]
                ATM[s]["y"] = [Sens[s].values[atmidx]]
                # ATM[s]["x"] = [K]
                # ATM[s]["y"] = [Sens[s].loc[K]]
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

        # Theta and Lambda
        with st.container():
            plot1, plot2 = st.columns(2) #[1,1,1], gap="small") 
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
                ss = "Lambda"
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

    # except:
    else:
        # Call/Put and Strike not inserted yet 
        st.write("insert option and strike")


    # Hiding "Made with Streamlit message"
    st.write('''
        <style>
            footer {visibility:hidden;}
        </style>
        ''',
        unsafe_allow_html = True
    )


def _plotgreeks(
    data: pd.Series,
    CP: str,
    K: float,
    lcol: str,    
    atmv: tuple or None = None,
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

    # Lable x-axis
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
