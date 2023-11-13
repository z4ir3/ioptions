'''
'''
import streamlit as st
from streamlit_option_menu import option_menu 

# from main import mainbalance
# from src.balcalcs import mapaccountnames
# from src.plots import balplot, pnlplot



def main():
    """
    """
    # Page configuration
    # Streamlit supported icons:
    # https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
    st.set_page_config(
        page_title = "Option Playground",
        page_icon = ":arrows_counterclockwise:",
        layout = "wide",
        initial_sidebar_state = "expanded",
        menu_items={
            "About": "# Option playground"
        }
    )

    # Remove extra white space
    st.write('''
        <style>
            div.block-container {padding-top: 0rem;} 
        </style>
        ''',
        unsafe_allow_html = True
    )

    with st.sidebar:
        # Sidebar title
        st.sidebar.title("")

        PageSelected = option_menu(
            menu_title = None,
            menu_icon = "cast",
            options=[
                "Home",
                "Option Greeks",
                "Option Strategies"
            ],
            icons = [
                "house",
                "box-arrow-in-right",
                "stack"
            ], # icons from the bootstrap webpage
            default_index = 0,
            orientation = "vertical"
        )

    if PageSelected == "Home":
        _ = dbpage_home()
    elif PageSelected == "Option Greeks":
        _ = dbpage_greeks()
    else:
        _ = dbpage_strategies()

    # Hiding "Made with Streamlit message"
    st.write('''
        <style>
            footer {visibility:hidden;}
        </style>
        ''',
        unsafe_allow_html = True
    )

def dbpage_home():
    """
    """
    # Page title
    st.title("The Black-Scholes Option Playground")
    st.write("---")



    # and $\tau:=T-t$ be the time-to-maturity. 
    # The solution to the Black-Scholes equation $(9)$ 
    # in case of a European Call option with boundary condition $(10)$ is  
    # $$
    # C_t = S_t N(h) - Ke^{-r\tau}N(h-\sigma\sqrt{\tau}),
    # \qquad\qquad (12) 
    # $$
    # and in case of a European Put with boundary condition $(11)$ is
    # $$
    # P_t = -S_t N(-h) + Ke^{-r\tau}N(-h+\sigma\sqrt{\tau}), 	
    # \qquad\qquad (13) 
    # $$
    # where in both cases
    # $$ 
    # h = \frac{\ln\left(S_t/K\right) 
    # + \left(r + \sigma^2 / 2 \right)\tau}{\sigma\sqrt{\tau}},
    # \qquad\qquad (14)
    # $$
    # and where $N(\cdot)$ denotes the Cumulative Distribution Function (CDF) 
    # of the standard Normal distribution, i.e., 
    # $$
    # N(h) = \int_{-\infty}^h \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{s^2}{2}\right) ds.
    # \qquad\qquad (15)
    # $$


    st.markdown('''
        <style>
        .katex-html {
            text-align: left;
            font-family: monospace;
            font-size: 16px;
        }
        </style>''',
    unsafe_allow_html=True
    )
    st.latex(r'''
        \text{Let } T 
        \text{ be the maturity of the contract so that } 
        \tau := T - t 
        \text{ denotes the time to expiration (in days)}
    ''')
    st.latex(r'''
        \text{The Black-Scholes price of a European
        {\bf Call} and {\bf Put} Options are given by}
    ''')
    st.latex(r'''
        \bullet\qquad C_t = S_t N(h) - Ke^{-r\tau}N(h-\sigma\sqrt{\tau})
    ''')
    st.latex(r''' 
        \bullet\qquad P_t = -S_t N(-h) + Ke^{-r\tau}N(-h+\sigma\sqrt{\tau}), 	
    ''')





def dbpage_greeks():
    """
    """
    # Page title
    st.title("The Black-Scholes Option Playgound")
    st.header("Option Greeks")
    st.write("---")


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

def dbpage_strategies():
    """
    """
    # Page title
    st.title("The Black-Scholes Option Playgound")
    st.header("Option Strategies calculator")
    st.write("---")
    
    pass



if __name__ == "__main__":
    main()

