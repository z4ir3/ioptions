"""
"""
import streamlit as st
from streamlit_option_menu import option_menu 

# from src.home import dbpage_home
from src.greeks import dbpage_greeks
from src.strategies import dbpage_strategies


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
            div.block-container {
                padding-top: 0rem; 
            }
        </style>
        ''',
        unsafe_allow_html = True
    )

    with st.sidebar:
        # Sidebar title
        st.sidebar.title("")

        PageSelected = option_menu(
            menu_title = "Option Playgrounds",
            menu_icon = "bar-chart",
            options=[
                # "Home",
                "Option Greeks",
                "Option Strategies"
            ],
            icons = [
                # "house",
                "box-arrow-in-right",
                "stack"
            ], # icons from the bootstrap webpage
            default_index = 0,
            orientation = "vertical",
            # orientation = "horizontal"
            # styles={
            #     "container": {"padding": "0!important", "background-color": "#fafafa"},
            #     "icon": {"color": "orange", "font-size": "25px"}, 
            #     "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            #     "nav-link-selected": {"background-color": "green"},
            # }
            styles={
                "nav-link": {"--hover-color": "#aaa"},
            }
        )
        # ss = """
        # <style>
        #     .nav-link:hover {
        #     color:rgb(100,100,150);
        # }
        # </style>
        # """
        # st.markdown(ss, unsafe_allow_html=True)

    if PageSelected == "Option Greeks":
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
