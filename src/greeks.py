"""
"""
import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objs as go 

from src.utilities import get_Smax, get_Smin, dbcol
from models.blackscholes import BSOption




def dbpage_greeks(
    nss: int = 75
):
    """
    """
    # # Page title
    st.title("Option Greeks")
    # st.write("---")



    par1, par2, par3, par4 = st.columns([0.5,0.5,0.5,0.5], gap="small") 
    with par1:
        # Call or Put price
        cp = st.selectbox(
            label = "Option type",
            options = ["Call","Put"],
            index = 0,
            key = "option-type"
        )
        CP = "C" if cp == "Call" else "P" 
    with par2:
        # Strike price 
        K = st.number_input(
            label = "Option strike (K)",
            min_value = 0.1,
            format = "%f", 
            value = 100.0,
            placeholder = None,
            help = "'Exercise' price of the option",
            key = "strike"
            # on_change=
        ) 
    with par3:
        # Dividend Yield
        q = st.number_input(
            label = "Dividend Yield (q)",
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

    
    
    

    with st.sidebar:
        st.write("Black-Scholes Call Option")


    # Expiration Slider 
    T = st.sidebar.slider(
        label = f"Time-to-Expiration ({TType})", 
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

    # Volatilty Slider 
    v = st.sidebar.slider(
        label =  "Volatility (%)", 
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

    # Interes Rate Slider 
    r = st.sidebar.slider(
        label = "Interest Rate (%)", 
        min_value = 0.0,
        max_value = 8.0,
        value = 2.0, 
        key = "slider-irate", 
        help = None, 
        # on_change = get_T(TType, minvt, maxvt)
    )
    r = r / 100

    with st.sidebar:
        st.markdown('''
            <style>
            .katex-html {
                text-align: left;
                font-family: monospace;
                font-size: 15px;
            }
            </style>''',
            unsafe_allow_html=True
        )
        if CP == "C":
            
            st.latex(r'''\text{Price}:\newline
                C_t = S_t N(h) - Ke^{-r\tau}N(h-\sigma\sqrt{\tau})
            ''')
        else:
            st.latex(r'''P_t = -S_t N(-h) + Ke^{-r\tau}N(-h+\sigma\sqrt{\tau}), 	
            ''')










    # oprices = Option.oprices()
    # # st.write( Option.price() ) 
    # st.write( Option.delta() ) 
    # st.write( oprices ) 

    Sset = np.linspace(get_Smin(K),get_Smax(K),nss)
    # st.write( Sset ) 

    options = [BSOption(CP=CP, S=s, K=K, T=T, r=r, v=v, q=q) for s in Sset]

    # st.write( oprices ) 
    # st.write( oprices ) 

    # Price and Delta
    plot1, plot2 = st.columns(2) #[1,1,1], gap="small") 
    with plot1:   
        oprices = [o.price() for o in options]
        oprices = pd.Series(oprices, index=Sset, name="Price")
        fig = _plotgreeks(
            oprices, 
            CP = CP,
            K = K, 
            lcol = dbcol(oprices.name)
        )
        st.plotly_chart(fig, use_container_width=True)
    with plot2:
        odeltas = [o.delta() for o in options]
        odeltas = pd.Series(odeltas, index=Sset, name="Delta")    
        fig = _plotgreeks(
            odeltas, 
            CP = CP, 
            K = K, 
            yaxside = "right", 
            lcol = dbcol(odeltas.name)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Gamma and Vega
    plot1, plot2 = st.columns(2) #[1,1,1], gap="small") 
    with plot1:   
        ogamma = [o.gamma() * 100 for o in options]
        ogamma = pd.Series(ogamma, index=Sset, name="Gamma")
        fig = _plotgreeks(
            ogamma, 
            CP = CP,
            K = K, 
            lcol = dbcol(ogamma.name)
        )
        st.plotly_chart(fig, use_container_width=True)
    with plot2:
        ovega = [o.vega() for o in options]
        ovega = pd.Series(ovega, index=Sset, name="Vega")    
        fig = _plotgreeks(
            ovega, 
            CP = CP, 
            K = K,
            yaxside = "right", 
            lcol = dbcol(ovega.name)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Theta and Lambda
    plot1, plot2 = st.columns(2) #[1,1,1], gap="small") 
    with plot1: 
        otheta = [o.theta() for o in options]
        otheta = pd.Series(otheta, index=Sset, name="Theta")
        fig = _plotgreeks(
            otheta,
            CP = CP,
            K = K, 
            lcol = dbcol(otheta.name), 
            xlab = True
        )
        st.plotly_chart(fig, use_container_width=True)
    with plot2:
        olambda = [o.llambda() for o in options]
        olambda = pd.Series(olambda, index=Sset, name="Lambda")    
        fig = _plotgreeks(
            olambda, 
            CP = CP, 
            K = K, 
            yaxside = "right", 
            lcol = dbcol(olambda.name), 
            xlab=True
        )
        st.plotly_chart(fig, use_container_width=True)

def _plotgreeks(
    data: pd.Series,
    CP: str,
    K: float,
    lcol: str,    
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
            y = np.repeat(0, len(data.index)), 
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
        # shapedefaults = {'line': {'color': '#2a3f5f'}},
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
    atmidx = np.argmin(pd.Series(data.index).apply(lambda x: abs(x - K)))
    atmx = [data.index[atmidx]]
    atmy = [data.values[atmidx]]
    fig.add_trace(
        go.Scatter(
            x = atmx,
            y = atmy, 
            name = f"ATM {data.name} ({atmy[0]:.1f})",
            marker = dict(
                color = "black",
                size = [8]
            ),
            showlegend = True
        )
    )



    return fig

    


    # st.header("Option Greeks")

    # with st.sidebar:
    #     # Sidebar title
    #     st.sidebar.title("")
        
    #     # Currency selection    
    #     sel_ccy = st.sidebar.selectbox(
    #         "Choose Account Currency",
    #         [1,2,3],
    #         index = 0,
    #     ) 
        
        # # Including (or not) closed accounts
        # closedacc = st.sidebar.selectbox(
        #     "Include closed accounts?",
        #     ("Yes","No"),
        #     index = 0,
        # )
        
        # # Choosing accounts
        # accstochoosefrom = list(TOT[sel_ccy].columns)
        # if closedacc == "No":
        #     caccsmaps = [mapaccountnames(ac) for ac in caccs]        
        #     showaccs = [ac for ac in accstochoosefrom if ac not in caccsmaps]
        # else:
        #     showaccs = accstochoosefrom
        # sel_acc = st.sidebar.multiselect(
        #     "Choose Specific Account",
        #     showaccs, 
        #     default = "Total",
        #     help = "To plot selected specific accounts in the given currency"
        # ) 


        
    # info1, info2, info3 = st.columns([1,1,2], gap="small") #(3) #, gap = "large")
    # with info1:
    #     # Session Date
    #     sessiondate = TOT[sel_ccy].index[-1].strftime("%d/%m/%Y")
    #     st.metric(
    #         label="Today's date", 
    #         value=sessiondate
    #     )

    # with info2:
    #     # Total amount (sum of all account)
    #     # This is already the total over all opened accounts
    #     totamount = TOT[sel_ccy]["Total"].tail(1).values[0]

    #     # st.info("Overall Amount") #, icon=":pushpin:")
    #     st.metric(
    #         label=f"Sum of all {sel_ccy} opened accounts", 
    #         value=f"{totamount:,.0f}€"
    #     )

    # with info3:
    #     accfilt = [acc for acc in sel_acc if acc != "Total"]
    #     amtfilt = TOT[sel_ccy][accfilt].tail(1).sum(axis=1).values[0]    
    #     st.metric(
    #         label=f"Sum of filtered opened {sel_ccy} accounts (not Total)", 
    #         value=f"{amtfilt:,.0f}€"
    #     )


    # # Tabs
    # tplot, tdata, tpnl = st.tabs(["Plot", "Data", "PnL"])
    # with tplot:
    #     # # Call the create_plot function from the main script
    #     # fig = balanceplot(Balance, figshow=False)
        
    #     # # Display the plot using Streamlit
    #     # st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    #     fig = balplot(
    #         TOT[sel_ccy][sel_acc], 
    #         ccy = sel_ccy,
    #         fshow = False
    #     )
    #     st.plotly_chart(fig, use_container_width=True) #, theme="streamlit")


    # with tdata:    
    #     # Show data-filtered dataframe of balances 
    #     # with st.expander("Show/Hide selected balance data"):
    #     td = TOT[sel_ccy][sel_acc].reset_index(drop=False).rename(columns={0:"Date"})
    #     # td = td.set_index("Date")
    #     st.table(td) 

    # with tpnl:
    #     pnltype = st.selectbox(
    #         "Select the type of PnL",
    #         ("Absolute", "Percentage")
    #     )     
    #     sel_pnl = "abs" if pnltype == "Absolute" else "rel"
    #     st.write(f"{pnltype} PnLs of the 'Total' account")       
    #     fig = pnlplot(
    #         PNL[sel_ccy]["Total"], 
    #         ccy = sel_ccy,
    #         pnltype = sel_pnl,
    #         fshow = False
    #     )
    #     st.plotly_chart(fig, use_container_width=True) #, theme="streamlit")