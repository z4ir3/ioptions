'''
'''
import streamlit as st
from streamlit_option_menu import option_menu 

from src.home import dbpage_home
from src.greeks import dbpage_greeks
from src.optionstrategies import dbpage_strategies


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




if __name__ == "__main__":
    main()
