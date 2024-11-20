import os
import requests
import warnings
import logging
import streamlit as st
from langchain.globals import set_verbose, get_verbose
import pandas as pd
import csv
import io
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text
import re
from urllib.parse import urlparse
import traceback
import chromadb

# Initial setup remains the same
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set")

if "USER_AGENT" not in st.session_state:
    os.environ["USER_AGENT"] = "MyStreamlitApp/1.0"
    st.session_state.USER_AGENT_set = True
    print("USER_AGENT set to:", os.getenv("USER_AGENT"))
else:
    print("USER_AGENT already set to:", os.getenv("USER_AGENT"))

set_verbose(True)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_portfolio_data(df):
    errors = []
    if list(df.columns) != ['Techstack', 'Links']:
        errors.append("File must have exactly two columns: 'Techstack' and 'Links'")
        return errors
    
    for idx, row in df.iterrows():
        if ' | ' not in str(row['Techstack']):
            errors.append(f"Row {idx + 1}: Tech stack must contain technologies separated by ' | '")
        if not is_valid_url(str(row['Links'])):
            errors.append(f"Row {idx + 1}: Invalid URL format")
    
    return errors

def create_streamlit_app(llm, portfolio, clean_text):
    # Initialize session state
    if 'context' not in st.session_state:
        st.session_state.context = ""
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Settings"

    # Set page config
    st.set_page_config(
        page_title="Cold Email Synthesizer", 
        page_icon="🌌", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply alien theme CSS
    st.markdown("""
    <style>
    /* Base theme override */
    .stApp {
          background: hsla(270, 94%, 25%, 1);

background: linear-gradient(135deg, hsla(270, 94%, 25%, 1) 20%, hsla(158, 94%, 49%, 1) 100%);

background: -moz-linear-gradient(315deg, hsla(270, 94%, 25%, 1) 20%, hsla(158, 94%, 49%, 1) 100%);

background: -webkit-linear-gradient(315deg, hsla(270, 94%, 25%, 1) 20%, hsla(158, 94%, 49%, 1) 100%);

filter: progid: DXImageTransform.Microsoft.gradient( startColorstr="#42047e", endColorstr="#07f49e", GradientType=1 );
    }
    
    /* Headers and text */
    h1, h2, h3 {
        color: #4CAF50 !important;
        font-family: 'Space Mono', monospace;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(16, 24, 39, 0.8) !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
        color: #4CAF50 !important;
        border-radius: 10px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50, #2196F3) !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        transform: skew(-10deg) !important;
    }
    
    .stButton > button:hover {
        transform: skew(-10deg) translateY(-2px) !important;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.5) !important;
    }
    
    /* File uploader */
    .stUploadButton {
        background: rgba(16, 24, 39, 0.8) !important;
        border: 2px dashed rgba(76, 175, 80, 0.5) !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    
    /* Data editor/table */
    .stDataFrame {
        background: rgba(16, 24, 39, 0.6) !important;
        border: 1px solid rgba(138, 43, 226, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(16, 24, 39, 0.8) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        color: #4CAF50 !important;
        padding: 10px 20px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(76, 175, 80, 0.2) !important;
        border-color: #4CAF50 !important;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError {
        background-color: rgba(16, 24, 39, 0.8) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 10px !important;
        color: #4CAF50 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(16, 24, 39, 0.9) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Context example box */
    .context-example {
        background: rgba(16, 24, 39, 0.8);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #4CAF50;
    }
    
    /* Loading animation */
    .stSpinner {
        border-color: #4CAF50 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(76, 175, 80, 0.3);
        border-radius: 5px;
    }
    
    /* Alien glow effect for headers */
    @keyframes alienGlow {
        0% { text-shadow: 0 0 5px #4CAF50, 0 0 10px #4CAF50; }
        50% { text-shadow: 0 0 10px #4CAF50, 0 0 20px #4CAF50; }
        100% { text-shadow: 0 0 5px #4CAF50, 0 0 10px #4CAF50; }
    }
    
    h1 {
        animation: alienGlow 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and subheader with alien theme
    st.markdown("<h1 style='text-align: center; font-size: 2.5em;'>🌌 Cold Email Synthesizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4CAF50; font-size: 1.2em;'>GenAI-Enhanced Synthesizing Protocol</p>", unsafe_allow_html=True)

    # Create tabs with alien theme
    tab1, tab2 = st.tabs(["Configuration Matrix", "Email Synthesizer"])

    # Configuration Matrix Tab
    with tab1:
        st.markdown("### Enter Your Context")
        
        if st.button("Access Example Context"):
            st.markdown("""
            <div class="context-example">
            <strong>Context Reference:</strong>
            You are Aditya, a business development executive at AdityaTechLabs. AdityaTechLabs is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AdityaTechLabs 
            in fulfilling their needs.
            </div>
            """, unsafe_allow_html=True)

        st.session_state.context = st.text_area(
            "Initialize Your Context",
            value=st.session_state.context,
            height=200,
            help="Describe your company, services, and how you want to present yourself in the cold emails",
            key="context_input"
        )

        st.markdown("### Upload Your Portfolio")
        st.markdown("Synchronize your Portfolio (in CSV format)")
        
        uploaded_file = st.file_uploader("Initialize Data Transfer", type=['csv'], key="portfolio_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                errors = validate_portfolio_data(df)
                
                if errors:
                    st.error("Portfolio Synchronization Failed:")
                    for error in errors:
                        st.error(error)
                else:
                    st.success("Portfolio Successfully Integrated!")
                    
                    st.markdown("### Portfolio Preview")
                    edited_df = st.data_editor(
                        df,
                        num_rows="dynamic",
                        use_container_width=True,
                        hide_index=True,
                        key="portfolio_editor"
                    )

                    if st.button("Commit Synchronization"):
                        st.session_state.portfolio_data = edited_df
                        st.success("Synchronization State Successfully Updated!")

            except Exception as e:
                st.error(f"Synchronization Error: {str(e)}")

        template_data = """Techstack,Links
React | Node.js | MongoDB,https://example.com/react-portfolio
Angular | .NET | SQL Server,https://example.com/angular-portfolio"""
        
        st.download_button(
            label="Download Example Template",
            data=template_data,
            file_name="my_portfolio.csv",
            mime="text/csv"
        )

    # Transmission Generator Tab
    with tab2:
        if not st.session_state.context:
            st.error("Error: Entity Parameters Not Initialized!")
            st.stop()
            
        if st.session_state.portfolio_data is None:
            st.error("Error: Portfolio Not Synchronized!")
            st.stop()

        try:
            portfolio.load_custom_portfolio(st.session_state.portfolio_data)

        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Portfolio Integration Error: {e}")
            print(f"Quantum Trace:\n{error_details}")

        url_input = st.text_input(
            "Target Signal Coordinates", 
            placeholder="Input Job page URL",
            help="Initialize target web page URL"
        )
        
        generate_email_button = st.button("Synthesize Email")
        
        if generate_email_button:
            if not url_input:
                st.error("Error: Target Web Page URL Required!")
        else:
            try:
                with st.spinner('Initializing Cold Email Protocol...'):
                    loader = WebBaseLoader([url_input])
                    data = clean_text(loader.load().pop().page_content)
                
                # Add logging/print to verify data extraction
                print("Extracted Job Data:", data)
                
                jobs = llm.extract_jobs(data)
                
                # Add logging/print to verify job extraction
                print("Extracted Jobs:", jobs)
                
                if not jobs:
                    st.warning("No jobs could be extracted from the provided URL.")
                
                for job in jobs:
                    skills = job.get('skills', [])
                    
                    # Add logging for skills
                    print("Job Skills:", skills)
                    
                    links = portfolio.query_links(skills)
                    
                    # Add logging for links
                    print("Matching Portfolio Links:", links)
                    
                    email = llm.write_mail(job, links, st.session_state.context)
                    
                    # Add logging for email generation
                    print("Generated Email:", email)
                    
                    st.markdown(f"### 📡 Synthesized Cold Email")
                    st.code(email, language='markdown')
        
            except Exception as e:
            # More detailed error tracing
                st.error(f"Quantum Synthesis Error: {e}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    llm = Chain()
    portfolio = Portfolio()
    create_streamlit_app(llm, portfolio, clean_text)
