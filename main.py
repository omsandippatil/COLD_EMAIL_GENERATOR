import os
import requests
import warnings
import logging
import streamlit as st

# Import the appropriate verbose methods from langchain.globals
from langchain.globals import set_verbose, get_verbose

# Set logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings related to USER_AGENT
warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set")

# Set the USER_AGENT environment variable only once
if "USER_AGENT" not in st.session_state:
    os.environ["USER_AGENT"] = "MyStreamlitApp/1.0"
    st.session_state.USER_AGENT_set = True
    print("USER_AGENT set to:", os.getenv("USER_AGENT"))
else:
    print("USER_AGENT already set to:", os.getenv("USER_AGENT"))

# Set verbose mode using the recommended function from langchain.globals
set_verbose(True)  # Set this according to your needs (True or False)

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

# Function to check if the URL is valid
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Validate portfolio data
def validate_portfolio_data(df):
    errors = []
    
    # Check columns
    if list(df.columns) != ['Techstack', 'Links']:
        errors.append("File must have exactly two columns: 'Techstack' and 'Links'")
        return errors
    
    for idx, row in df.iterrows():
        # Check tech stack format
        if ' | ' not in str(row['Techstack']):
            errors.append(f"Row {idx + 1}: Tech stack must contain technologies separated by ' | '")
        
        # Check URL format
        if not is_valid_url(str(row['Links'])):
            errors.append(f"Row {idx + 1}: Invalid URL format")
    
    return errors

# Function to create the Streamlit app
def create_streamlit_app(llm, portfolio, clean_text):
    # Initialize session state
    if 'context' not in st.session_state:
        st.session_state.context = ""
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Settings"

    # Set page config with a modern theme
    st.set_page_config(
        page_title="Cold Mail Generator", 
        page_icon="🥶", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Original CSS plus new styles
    st.markdown("""
    <style>
    header {
        background: none !important;
        display: none !important;
    }

    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #3498db;
        padding: 10px;
        font-size: 16px;
    }

    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }

    .email-output {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .preview-table {
        margin-top: 20px;
        border-radius: 10px;
        overflow: hidden;
    }

    .context-example {
        background-color: #000000;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and subheader
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🥶 Cold Mail Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Generate personalized cold emails with AI-powered precision</p>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2 = st.tabs(["Settings", "Generate Email"])

    # Settings Tab
    with tab1:
        st.markdown("### Company Context")
        
        # Example context button
        if st.button("View Example Context"):
            st.markdown("""
            <div class="context-example">
            <strong>Example Context:</strong>
            You are Aditya, a business development executive at AdityaTechLabs. AdityaTechLabs is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AdityaTechLabs 
            in fulfilling their needs.
            </div>
            """, unsafe_allow_html=True)

        # Context input with session state
        st.session_state.context = st.text_area(
            "Enter your company context",
            value=st.session_state.context,
            height=200,
            help="Describe your company, services, and how you want to present yourself in the cold emails",
            key="context_input"
        )

        st.markdown("### Portfolio Upload")
        st.markdown("Upload your portfolio in CSV format with columns: Techstack, Links")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="portfolio_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                errors = validate_portfolio_data(df)
                
                if errors:
                    st.error("Validation errors found:")
                    for error in errors:
                        st.error(error)
                else:
                    st.success("Portfolio uploaded successfully!")
                    
                    # Preview table with edit/delete functionality
                    st.markdown("### Portfolio Preview")
                    edited_df = st.data_editor(
                        df,
                        num_rows="dynamic",
                        use_container_width=True,
                        hide_index=True,
                        key="portfolio_editor"
                    )

                    # Save changes to session state
                    if st.button("Save Changes"):
                        st.session_state.portfolio_data = edited_df
                        st.success("Changes saved successfully!")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

        # Download template button
        template_data = """Techstack,Links
React | Node.js | MongoDB,https://example.com/react-portfolio
Angular | .NET | SQL Server,https://example.com/angular-portfolio"""
        
        st.download_button(
            label="Download Template",
            data=template_data,
            file_name="portfolio_template.csv",
            mime="text/csv"
        )

    # Generate Email Tab
    with tab2:
        # Check if context and portfolio are set
        if not st.session_state.context:
            st.error("Please provide company context in the Settings tab!")
            st.stop()
            
        if st.session_state.portfolio_data is None:
            st.error("Please upload your portfolio in the Settings tab!")
            st.stop()

        try:
            # Load the portfolio only once
            portfolio.load_custom_portfolio(st.session_state.portfolio_data)

        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error loading portfolio: {e}")
            print(f"Traceback:\n{error_details}")

        # URL Input
        url_input = st.text_input(
            "Enter Job Description URL", 
            placeholder="Paste the job description URL here",
            help="Paste the URL of the job description"
        )
        
        # Submit Button
        generate_email_button = st.button("Generate Cold Email")
        
        if generate_email_button:
            if not url_input:
                st.error("Please enter a job description URL!")
            else:
                try:
                    with st.spinner('Generating personalized email...'):
                        loader = WebBaseLoader([url_input])
                        data = clean_text(loader.load().pop().page_content)
                        jobs = llm.extract_jobs(data)
                        
                        for job in jobs:
                            skills = job.get('skills', [])
                            links = portfolio.query_links(skills)
                            email = llm.write_mail(job, links, st.session_state.context)
                            
                            st.markdown(f"### 📝 Generated Cold Email")
                            st.code(email, language='markdown')
                
                except Exception as e:
                    st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    llm = Chain()
    portfolio = Portfolio()
    create_streamlit_app(llm, portfolio, clean_text)
