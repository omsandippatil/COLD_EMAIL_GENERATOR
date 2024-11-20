import pandas as pd
import chromadb
import uuid
import re
import traceback

class Portfolio:
    def __init__(self, file_path='./resource/my_portfolio.csv'):
        """
        Initialize the Portfolio with a default file path and set up the database connection.
        """
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
    
    def load_portfolio(self):
        """
        Load the default portfolio data into the ChromaDB collection if it's empty.
        """
        if self.collection.count() == 0:  # Check if the collection is empty
            for _, row in self.data.iterrows():
                document = f"Techstack: {row['Techstack']}, Links: {row['Links']}"
                self.collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[document],
                    metadatas=[{"techstack": row["Techstack"]}]
                )
    
    def load_custom_portfolio(self, portfolio_data):
        """
        Load custom portfolio data from a pandas DataFrame into the ChromaDB collection.
        Args:
            portfolio_data (pd.DataFrame): A DataFrame containing portfolio data with 'Techstack' and 'Links' columns.
        """
        if not isinstance(portfolio_data, pd.DataFrame):
            raise ValueError("portfolio_data must be a pandas DataFrame with columns: 'Techstack' and 'Links'")
        
        # Validate columns
        if list(portfolio_data.columns) != ['Techstack', 'Links']:
            raise ValueError("DataFrame must have exactly two columns: 'Techstack' and 'Links'")
        
        try:
            # Clear the existing collection by deleting all entries
            existing_ids = self.collection.get()["ids"]  # Retrieve all current IDs in the collection
            if existing_ids:  # If there are existing entries
                self.collection.delete(ids=existing_ids)
            
            # Load the new portfolio data
            for _, row in portfolio_data.iterrows():
                document = f"Techstack: {row['Techstack']}, Links: {row['Links']}"
                self.collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[document],
                    metadatas=[{"techstack": row["Techstack"]}]
                )
            
            print("Custom portfolio loaded successfully.")
        
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error loading custom portfolio: {e}")
            print(f"Traceback:\n{error_details}")
    
    def query_links(self, skills):
        """
        Query portfolio links based on skills with robust error handling.
        
        Args:
            skills (list or str): A list of skills or a single skill string
        
        Returns:
            list: Matching portfolio links
        """
        # Ensure skills is a list
        if isinstance(skills, str):
            skills = [skills]
        
        # Filter out None or empty skills
        skills = [skill.strip() for skill in skills if skill and isinstance(skill, str)]
        
        # If no valid skills, return empty list
        if not skills:
            return []
        
        # Create query by joining non-empty skills
        query = " OR ".join(skills)
        
        # Perform vector search with error handling
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3  # Limit to top 3 results
            )
            
            # Extract and return links from results
            return results.get('documents', [])[0] if results.get('documents') else []
        
        except Exception as e:
            print(f"Error in query_links: {e}")
            return []
