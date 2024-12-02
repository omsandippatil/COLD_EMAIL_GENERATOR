import os
import requests
import warnings
import logging
import streamlit as st
from langchain.globals import set_verbose
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

# Logging and warning setup
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set")

# Set USER_AGENT safely
if "USER_AGENT" not in st.session_state:
    os.environ["USER_AGENT"] = "MyStreamlitApp/1.0"
    st.session_state.USER_AGENT_set = True
    st.write("USER_AGENT set to:", os.getenv("USER_AGENT"))

set_verbose(True)

def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_portfolio_data(df):
    """Validate portfolio CSV data"""
    errors = []
    
    # Check column names
    if list(df.columns) != ['Techstack', 'Links']:
        errors.append("File must have exactly two columns: 'Techstack' and 'Links'")
        return errors
    
    # Validate each row
    for idx, row in df.iterrows():
        if ' | ' not in str(row['Techstack']):
            errors.append(f"Row {idx + 1}: Tech stack must contain technologies separated by ' | '")
        
        if not is_valid_url(str(row['Links'])):
            errors.append(f"Row {idx + 1}: Invalid URL format")
    
    return errors

def create_streamlit_app(llm, portfolio, clean_text):
    """Main Streamlit application"""
    # Initialize session state
    if 'context' not in st.session_state:
        st.session_state.context = ""
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Settings"
    
    # Page configuration
    st.set_page_config(
        page_title="Cold Email Synthesizer", 
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS (truncated for brevity, same as previous code)
    st.markdown("""<style>...</style>""", unsafe_allow_html=True)
    
    # Title and description
    st.markdown("<h1 style='text-align: center; font-size: 2.5em;'>🌌 Cold Email Synthesizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4CAF50; font-size: 1.2em;'>GenAI-Enhanced Synthesizing Protocol</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Configuration Matrix", "Email Synthesizer"])
    
    # Configuration Matrix Tab
    with tab1:
        st.markdown("### Enter Your Context")
        
        if st.button("Access Example Context"):
            st.markdown("""
            <div class="context-example">
            <strong>Context Reference:</strong>
            You are Aditya, a business development executive at AdityaTechLabs. AdityaTechLabs is an AI & Software Consulting company dedicated to facilitating the seamless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, process optimization, cost reduction, and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AdityaTechLabs in fulfilling their needs.
            </div>
            """, unsafe_allow_html=True)
        
        # Context input
        st.session_state.context = st.text_area(
            "Initialize Your Context",
            value=st.session_state.context,
            height=200,
            help="Describe your company, services, and how you want to present yourself in the cold emails",
            key="context_input"
        )
        
        # Portfolio upload
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
        
        # Template download
        template_data = """Techstack,Links\nReact | Node.js | MongoDB,https://example.com/react-portfolio\nAngular | .NET | SQL Server,https://example.com/angular-portfolio"""
        st.download_button(
            label="Download Example Template",
            data=template_data,
            file_name="my_portfolio.csv",
            mime="text/csv"
        )
    
    # Email Synthesizer Tab
    with tab2:
        # Prerequisite checks
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
            st.error(error_details)
            st.stop()
        
        # URL input and email generation
        url_input = st.text_input(
            "Target Webpage URL", 
            placeholder="Input Job page URL",
            help="Initialize target web page URL"
        )
        
        # Modify button logic to prevent unnecessary errors
        generate_email_button = st.button("Synthesize Email")
        
        if generate_email_button:
            if not url_input:
                st.error("Error: Target Web Page URL Required!")
            else:
                try:
                    with st.spinner('Initializing Cold Email Protocol...'):
                        # Web page loading
                        loader = WebBaseLoader([url_input])
                        data = clean_text(loader.load().pop().page_content)
                        
                        # Log for debugging
                        st.write("Extracted Job Data:", data)
                        
                        # Job extraction
                        jobs = llm.extract_jobs(data)
                        
                        # Log for debugging
                        st.write("Extracted Jobs:", jobs)
                        
                        if not jobs:
                            st.warning("No jobs could be extracted from the provided URL.")
                            st.stop()
                        
                        # Generate emails for each job
                        for job in jobs:
                            skills = job.get('skills', [])
                            links = portfolio.query_links(skills)
                            
                            # Log for debugging
                            st.write("Job Skills:", skills)
                            st.write("Matching Portfolio Links:", links)
                            
                            email = llm.write_mail(job, links, st.session_state.context)
                            
                            # Log for debugging
                            st.write("Generated Email:", email)
                            
                            st.markdown(f"### 📡 Synthesized Cold Email")
                            st.code(email, language='markdown')
                
                except Exception as e:
                    # Comprehensive error tracing
                    st.error(f"Quantum Synthesis Error: {e}")
                    st.error(traceback.format_exc())

def main():
    """Main entry point"""
    llm = Chain()
    portfolio = Portfolio()
    create_streamlit_app(llm, portfolio, clean_text)

if __name__ == "__main__":
    main()
