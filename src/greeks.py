"""
"""

import streamlit as st

from models.blackscholes import BSOption


def dbpage_greeks():
    """
    """
    # Page title
    st.title("The Black-Scholes Option Playgound")
    st.header("Option Greeks")
    st.write("---")

    par1, par2, par3, par4 = st.columns([1,1,0.5,0.5], gap="small") 
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
            label = "Option strike",
            min_value = 0.1,
            format = "%f", 
            value = 100.0,
            placeholder = None,
            help = "'Exercise' price of the option",
            key = "strike"
            # on_change=
        ) 
    with par3:
        # Expiration type
        TType = st.selectbox(
            label = "Expiration type",
            options = ["Days","Years"],
            index = 0,
            key = "dte-type"
        ) 
    with par4:
        # Expiration  
        ff = "%d" if TType == "Days" else "%f"
        minv = 1 if TType == "Days" else 0.0028
        maxv = 1825 if TType == "Days" else 5.0
        vval = 90 if TType == "Days" else 0.25
        T = st.number_input(
            label = "Expiration (T)",
            min_value = minv,
            max_value = maxv,
            format = ff,
            value = vval,
            help = None,
            key = "dte"
            # on_change=
        ) 

    par1, par2, par3 = st.columns(3) #[1,1,1], gap="small") 
    with par1:
        # Volatility (%)
        v = st.number_input(
            label = "Volatility (in percentage) (v)",
            min_value = 1.0,
            max_value = 99.9,
            format = "%f",
            value = 30.0,
            help = "'Implied' Volatility",
            key = "volatility"
            # on_change=
        ) 
    with par2:
        # Interest Rate (%)
        r = st.number_input(
            label = "Interest Rate (in percentage) (r)",
            min_value = None,
            max_value = None,
            format = "%f",
            value = 3.5,
            help = None,
            key = "interest-rate"
            # on_change=
        ) 
    with par3:
        # Dividend Yield
        q = st.number_input(
            label = "Dividend Yield (q)",
            min_value = None,
            max_value = None,
            format = "%f",
            value = 0.0,
            help = None,
            key = "div-yield"
            # on_change=
        )

    Option = BSOption(CP, 98, K, r/100, T, v/100, q)
    st.write( Option.price() ) 
    st.write( Option.delta() ) 

    


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