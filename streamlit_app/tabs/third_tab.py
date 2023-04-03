import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

title = "Tensorboard"
sidebar_name = "Tensorboard"


def run():

    st.title(title)

    
    components.iframe("http://localhost:6109/#projector", height=1200)
