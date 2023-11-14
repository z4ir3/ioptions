"""
"""

import streamlit as st

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

