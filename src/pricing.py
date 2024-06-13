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
    nss: int = 200,
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

        st.write(" ")
        st.write("Main option data:")

        col1, col2 = st.columns([1,1])
        with col1:
            ostyle = st.selectbox(
                label = "Option style",
                options = ["European","American"],
                index = 0,
                placeholder = "European or American",
                key = "option-style"
            )
        with col2:
            # Call or Put price
            isdisabled = True if ostyle == "American" else False
            underlying_type = st.selectbox(
                label = "Underlying type",
                options = ["Stock","Index"],
                index = 0,
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
                index = 0,
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
                value = 10.0, #100.0,
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
        with st.sidebar:
            st.divider()
            col1, col2 = st.columns([1,1])
            with col1:
                st.header("Pricing Model:")
            with col2:
                if ostyle == "European" and underlying_type == "Stock":
                    st.success("**Black-Scholes**")
                elif underlying_type == "Index":
                    st.success("**Black '76**")
                elif ostyle == "American":
                    st.header("Binomial-Tree") # (Cox-Ross-Rubinstein)")
                    st.write("...to be implemented yet")
                    return 0
        
        # Rest of widgets: expiration, volatility, and interest rate
        col1, col2, col3, col4, col5 = st.columns([1.25,0.625,0.5,0.5,0.25], gap="small") 
        # with col1:
        #     pass
        #     # # Expiration type
        #     # TType = st.selectbox(
        #     #     label = "Expiration type",
        #     #     options = ["Days","Years"],
        #     #     index = 0,
        #     #     key = "dte-type"
        #     # ) 
        with col1:
            # Expiration Slider 
            days_per_year = 365
            n_exp_years = 1
            T = st.slider(
                label = "Days to Expiration ($t$)", #if TType == "Years" else "Days to Expiration ($t$)",
                min_value = 0, #if TType == "Days" else 0.0, 
                max_value = n_exp_years * days_per_year, #if TType == "Days" else float(n_exp_years), 
                value = 90, #if TType == "Days" else n_exp_years / 2, 
                step = 1, #if TType == "Days" else 0.10, 
                format = "%d",
                key = "slider-exp", 
                help = "Natural days left until maturity" 
            )
            # if TType == "Days": 
            T = T / days_per_year
        with col2:
            # Volatilty Slider 
            v = st.slider(
                label =  "Volatility (%) ($\\sigma$)", 
                min_value = 1.0,
                max_value = 99.0,
                value = 30.0, 
                step = 1.0, 
                # format = None, 
                key = "slider-vola", 
                help = "Implied Volatility", 
            )
            v = v / 100
        with col3:
            # Interes Rate Slider 
            r = st.slider(
                label = "Interest Rate (%) ($r$)", 
                min_value = 0.0,
                max_value = 5.0,
                value = 1.0, 
                step = 0.1,
                key = "slider-irate", 
                help = "Risk-free rate"
            )
            r = r / 100
        with col4:
            atmprice = st.selectbox(
                label = "Moneyness",
                options = ["ATM Option (K=S)","Choose Underlying"], #"Custom Underlying"],
                index = 0,
                # help = "Enter a custom Underlying or let $S = K$ (ATM Option)"
            )
        # if atmprice is not None:
        with col5:
            if atmprice == "Choose Underlying":
                if cp == "Call":
                    helpmsg = "Enter $S > K$ for ITM Call, or $K < S$ for OTM Call"
                else:
                    helpmsg = "Enter $S < K$ for ITM Put, or $K > S$ for OTM Put"
            else:
                helpmsg = None
            underlying_moneyness = st.number_input(
                label = "Enter $S$",
                min_value = 0.0,
                max_value = None,
                format = "%f",
                value = K,
                help = helpmsg,
                disabled = True if atmprice == "ATM Option (K=S)" else False
            )

            # Calculate moneyness 
            if atmprice == "ATM Option (K=S)":
                moneyness = "ATM"
            else:
                if cp == "Call":
                    moneyness = "ITM" if underlying_moneyness > K else "OTM"
                else:
                    moneyness = "ITM" if underlying_moneyness < K else "OTM"

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

        st.markdown('''
        <style>
        .katex-html {
            text-align: left;
            /*font-family: monospace;*/
        }
        </style>''',
        unsafe_allow_html=True
        )

        # total_tabs = ["The Model"] + sensname + ["All Sensitivities"]
        total_tabs = sensname + ["All Sensitivities"]
        tabs = st.tabs(total_tabs)
        cols_size = [7,1]
        for idx, s in enumerate(total_tabs):
            if s == "Price":
                with tabs[idx]:
                    st.subheader(f"{cp} " + s)
                    col1, col2 = st.columns(cols_size)
                    with col1:
                        if (underlying_type == "Stock") and (cp == "Call"):
                            # Black-Scholes Call
                            desc = r'''
                            C = Se^{-qt}\Phi(d_1) - Ke^{-rt}\Phi(d_2)
                            \quad\text{where}\quad
                            d_1 = \frac{\ln(S/K)+(r-q-\sigma^2/2)t}{\sigma\sqrt{t}}
                            \quad\text{and}\quad
                            d_2 = d_1 - \sigma\sqrt{t}
                            '''
                        elif (underlying_type == "Stock") and (cp == "Put"):
                            # Black-Scholes Put
                            desc = r'''
                            P = -Se^{-qt}\Phi(-d_1) + Ke^{-rt}\Phi(-d_2)
                            \quad\text{where}\quad
                            d_1 = \frac{\ln(S/K)+(r-q-\sigma^2/2)t}{\sigma\sqrt{t}}
                            \quad\text{and}\quad
                            d_2 = d_1 - \sigma\sqrt{t}
                            '''
                        elif (underlying_type == "Index") and (cp == "Call"):
                            # Black '76 Call
                            desc = r'''
                            C = e^{-rt}\left(S\Phi(d_1) - K\Phi(d_2)\right)
                            \quad\text{where}\quad
                            d_1 = \frac{\ln(S/K)+ (\sigma^2/2)t}{\sigma\sqrt{t}}
                            \quad\text{and}\quad
                            d_2 = d_1 - \sigma\sqrt{t}
                            '''
                        else:
                            # Black '76 Put 
                            desc = r''' 
                            P = e^{-rt}\left(-S\Phi(-d_1) + K\Phi(-d_2)\right)
                            \quad\text{where}\quad
                            d_1 = \frac{\ln(S/K)+ (\sigma^2/2)t}{\sigma\sqrt{t}}
                            \quad\text{and}\quad
                            d_2 = d_1 - \sigma\sqrt{t}
                            '''
                        st.latex(desc)
                    with col2:
                        st.metric(
                            label = f"ATM {s}",
                            value = 222,
                            help = None
                        )
            elif s == "Delta":
                with tabs[idx]:
                    st.subheader(f"First derivative of the {cp}'s price with respect to the Underlying Price")
                    col1, col2 = st.columns(cols_size)
                    with col1:
                        if (underlying_type == "Stock") and (cp == "Call"):
                            # Black-Scholes Call
                            desc = r'''
                            \Delta = \frac{\partial C}{\partial S} = e^{-qt}\Phi(d_1)
                            '''
                        elif (underlying_type == "Stock") and (cp == "Put"):
                            # Black-Scholes Put
                            desc = r'''
                            \Delta = \frac{\partial P}{\partial S} = -e^{-qt}\Phi(-d_1)
                            '''
                        elif (underlying_type == "Index") and (cp == "Call"):
                            # Black '76 Call
                            desc = r'''
                            \Delta = \frac{\partial C}{\partial S} = e^{-rt}\Phi(d_1)
                            '''
                        else:
                            # Black '76 Put 
                            desc = r'''
                            \Delta = \frac{\partial P}{\partial S} = -e^{-rt}\Phi(-d_1)
                            '''
                        st.latex(desc)
                    with col2:
                        _delta = Metric[s]['y'][0]
                        st.metric(
                            label = f"{moneyness} {s}",
                            value = f"{_delta:.3f}",
                            help = None
                        )

                    with col3:
                        cash_delta = _delta * underlying_moneyness
                        st.metric(
                            label = f"{moneyness} Cash {s}",
                            value = f"{cash_delta:.3f}",
                            help = "If Multiplied by 1%, represent the Price movement for +1% of the Underlying Price"
                        )

            elif s == "Gamma":
                with tabs[idx]:
                    st.subheader(f"Second derivative of the {cp}'s price with respect to the Underlying Price")
                    col1, col2 = st.columns(cols_size)
                    with col1:
                        if (underlying_type == "Stock") and (cp == "Call"):
                            # Black-Scholes Call
                            desc = r'''
                            \Gamma 
                            = \frac{\partial^2 C}{\partial S^2} 
                            = \frac{\partial \Delta}{\partial S} 
                            = \frac{1}{2\sigma\sqrt{t}}e^{-qt}\phi(d_1)
                            '''
                        elif (underlying_type == "Stock") and (cp == "Put"):
                            # Black-Scholes Put
                            desc = r'''
                            \Gamma 
                            = \frac{\partial^2 P}{\partial S^2} 
                            = \frac{\partial \Delta}{\partial S} 
                            = \frac{1}{2\sigma\sqrt{t}}e^{-qt}\phi(d_1)
                            '''
                        elif (underlying_type == "Index") and (cp == "Call"):
                            # Black '76 Call
                            desc = r'''
                            \Gamma 
                            = \frac{\partial^2 C}{\partial S^2} 
                            = \frac{\partial \Delta}{\partial S} 
                            = \frac{1}{2\sigma\sqrt{t}}e^{-rt}\phi(d_1)
                            '''
                        else:
                            # Black '76 Put 
                            desc = r'''
                            \Gamma 
                            = \frac{\partial^2 P}{\partial S^2} 
                            = \frac{\partial \Delta}{\partial S} 
                            = \frac{1}{2\sigma\sqrt{t}}e^{-rt}\phi(d_1)
                            '''
                        st.latex(desc)
                    with col2:
                        st.metric(
                            label = f"ATM {s}",
                            value = 222,
                            help = None
                        )
            elif s == "Vega":
                with tabs[idx]:
                    st.subheader(f"First derivative of the {cp}'s price with respect to the implied volatility")
                    col1, col2 = st.columns(cols_size)
                    with col1:
                        if (underlying_type == "Stock") and (cp == "Call"):
                            # Black-Scholes Call
                            desc = r'''
                            \kappa = \frac{\partial C}{\partial \sigma} = S e^{-qt} \phi(d_1) \sqrt{t}
                            '''
                        elif (underlying_type == "Stock") and (cp == "Put"):
                            # Black-Scholes Put
                            desc = r'''
                            \kappa = \frac{\partial P}{\partial \sigma} = S e^{-qt} \phi(d_1) \sqrt{t}
                            '''
                        elif (underlying_type == "Index") and (cp == "Call"):
                            # Black '76 Call
                            desc = r'''
                            \kappa = \frac{\partial C}{\partial \sigma} = S e^{-rt} \phi(d_1) \sqrt{t}
                            '''
                        else:
                            # Black '76 Put 
                            desc = r'''
                             \kappa = \frac{\partial P}{\partial \sigma} = S e^{-rt} \phi(d_1) \sqrt{t}
                            '''
                        st.latex(desc)
                    with col2:
                        st.metric(
                            label = f"ATM {s}",
                            value = 222,
                            help = None
                        )
            elif s == "Theta":
                with tabs[idx]:
                    st.subheader(f"First derivative of the {cp}'s price with respect to the time to maturity")
                    col1, col2 = st.columns(cols_size)
                    with col1:
                        if (underlying_type == "Stock") and (cp == "Call"):
                            # Black-Scholes Call
                            desc = r'''
                            \theta 
                            = \frac{\partial C}{\partial t} 
                            = -\frac{\sigma S e^{-qt}}{2\sqrt{t}}\phi(d_1) 
                            + q S e^{-qt}\Phi(d_1)
                            - r K e^{-rt}\Phi(d_2)
                            '''
                        elif (underlying_type == "Stock") and (cp == "Put"):
                            # Black-Scholes Put
                            desc = r'''
                            \theta 
                            = \frac{\partial P}{\partial t} 
                            = -\frac{\sigma S e^{-qt}}{2\sqrt{t}}\phi(d_1) 
                            - q S e^{-qt}\Phi(-d_1)
                            + r K e^{-rt}\Phi(-d_2)
                            '''
                        elif (underlying_type == "Index") and (cp == "Call"):
                            # Black '76 Call
                            desc = r'''
                            \theta 
                            = \frac{\partial C}{\partial t} 
                            = -\frac{\sigma S e^{-rt}}{2\sqrt{t}}\phi(d_1) 
                            + r S e^{-rt}\Phi(d_1)
                            - r K e^{-rt}\Phi(d_2)
                            '''
                        else:
                            # Black '76 Put 
                            desc = r'''
                            \theta 
                            = \frac{\partial P}{\partial t} 
                            = -\frac{\sigma S e^{-rt}}{2\sqrt{t}}\phi(d_1) 
                            - r S e^{-rt}\Phi(-d_1)
                            + r K e^{-rt}\Phi(-d_2)
                            '''
                        st.latex(desc)
                    with col2:
                        st.metric(
                            label = f"ATM {s}",
                            value = 222,
                            help = None
                        )
            elif s == "Rho":
                with tabs[idx]:
                    st.subheader(f"First derivative of the {cp}'s price with respect to the interest rate")
                    col1, col2 = st.columns(cols_size)
                    with col1:
                        if (underlying_type == "Stock") and (cp == "Call"):
                            # Black-Scholes Call
                            desc = r'''
                            \rho = \frac{\partial C}{\partial r} = K t e^{-rt} \Phi(d_2) 
                            '''
                        elif (underlying_type == "Stock") and (cp == "Put"):
                            # Black-Scholes Put
                            desc = r'''
                            \rho = \frac{\partial P}{\partial r} = -K t e^{-rt} \Phi(-d_2) 
                            '''
                        elif (underlying_type == "Index") and (cp == "Call"):
                            # Black '76 Call
                            desc = r'''
                            \rho = \frac{\partial C}{\partial r} 
                            = -t e^{-rt} \left(S\Phi(d_1) - K \Phi(d_2)\right) 
                            '''
                        else:
                            # Black '76 Put 
                            desc = r'''
                            \rho = \frac{\partial P}{\partial r} 
                            = -t e^{-rt} \left(-S\Phi(-d_1) + K\Phi(-d_2)\right) 
                            '''
                        st.latex(desc)
                    with col2:
                        st.metric(
                            label = f"ATM {s}",
                            value = 222,
                            help = None
                        )








        """
        for idx, s in enumerate(total_tabs):
            match s:
                # case "The Model":
                #     if ostyle == "European" and underlying_type == "Stock":
                #         desc = r'''Black-Scholes pricing model C = Se^{-qt}'''
                #     elif underlying_type == "Index":
                #         desc = "Black '76 pricing model"
                #     elif ostyle == "American":
                #         desc = "Binomial-Tree Model (Cox-Ross-Rubinstein) pricing model"
                case "Price":
                    desc = r'''
                    C = Se^{-qt}\Phi(d_1) - Ke^{-rt}\Phi(d_2)
                    \quad\text{where}\quad
                    d_1 = \frac{\ln(S/K)+(r-q-\sigma^2/2)t}{\sigma\sqrt{t}}
                    \quad\text{and}\quad
                    d_2 = d_1 - \sigma\sqrt{t}
                    '''
                case "Delta":
                    desc = r'''
                    \text{First derivative of the price with respect to the Underlying Price:}
                    \;\,
                    \Delta = \frac{\partial C}{\partial S} = e^{-qt}\Phi(d_1)
                    '''
                case "Gamma":
                    desc = r'''
                    \text{Second derivative of the price with respect to the Underlying Price:}
                    \;\,
                    \Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{e^{-qt}\phi(d_1)}{2\sigma\sqrt{t}}
                    '''
                case "Vega":
                    desc = r'''
                    \text{First derivative of the price with respect to the Implied Volatility:}
                    \;\,
                    \kappa = \frac{\partial C}{\partial \sigma} = S e^{-qt} \phi(d_1) \sqrt{t}
                    '''
                case "Theta":
                    desc = r'''
                    \text{First derivative of the price with respect to the time to maturity:}
                    \;\,
                    \theta = \frac{\partial C}{\partial t} 
                    = -\frac{S\sigma e^{-qt}}{2\sqrt{t}}\phi(d_1) 
                    - r K e^{-rt}\Phi(d_2)
                    + q S e^{-qt}\Phi(d_1)
                    '''                
                case "Rho":
                    desc = r'''
                    \rho = \frac{\partial C}{\partial r} = K t e^{-rt} \Phi(d_2)
                    '''  
                case "All Sensitivities":
                    desc = "" 
            with tabs[idx]:
                # if s == "The Model":
                #     st.subheader(desc)
                #     st.latex(desc)
                if s != "All Sensitivities":
                    col1, col2 = st.columns([6,1])
                    with col1:
                        # st.subheader(f"{cp} " + s)
                        st.subheader("First derivative of the price with respect to the interest rate")
                        st.latex(desc)
                        # if s == "Delta":
                        #     st.
                    with col2:
                        st.subheader("")
                        st.metric(
                            label = f"ATM {s}",
                            value = 222,
                            help = None
                        )
        """





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
                        value = f"{ATM[s]['y'][0]:.3f}",
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
