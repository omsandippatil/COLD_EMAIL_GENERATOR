import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import textwrap

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
        
        # Default context is now None to force context input
        self.default_context = None

    def extract_jobs(self, cleaned_text):
        # Import textwrap to handle text truncation
        max_length = 4000  # Adjust based on model's context window
        truncated_text = textwrap.shorten(cleaned_text, width=max_length, placeholder="...")

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            Extract the job postings in a concise JSON format with keys: `role`, `experience`, `skills`, and `description`.
            Limit to the most relevant job details if multiple exist.
            Only return valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
    
        try:
            chain_extract = prompt_extract | self.llm
            res = chain_extract.invoke(input={"page_data": truncated_text})
            json_parser = JsonOutputParser()
            parsed_res = json_parser.parse(res.content)
        
            return parsed_res if isinstance(parsed_res, list) else [parsed_res]
    
        except Exception as e:
            # Log the error or print for debugging
            print(f"Job extraction error: {e}")
            return []  # Return empty list if extraction fails

    def write_mail(self, job, links, context=None):
        # Check if context is provided
        if not context:
            raise ValueError("Please provide context")
            
        # Use provided context for email generation
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            {context}
            Also add the most relevant ones from the following links to showcase the portfolio: {link_list}
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        
        try:
            chain_email = prompt_email | self.llm
            res = chain_email.invoke({
                "job_description": str(job), 
                "link_list": links,
                "context": context
            })
            return res.content
        except Exception as e:
            print(f"Email generation error: {e}")
            return "Error generating email. Please try again."

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))