import pandas as pd
import chromadb
import uuid
import re
import traceback
import os

class Portfolio:
    def __init__(self, file_path=None):
        """
        Initialize the Portfolio with a default or custom file path and set up the database connection.
        """
        # Determine file path, with fallback to current directory
        if file_path is None:
            file_path = os.path.join(os.getcwd(), 'resource', 'my_portfolio.csv')
        
        self.file_path = file_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create default CSV if not exists
        if not os.path.exists(file_path):
            default_data = pd.DataFrame({
                'Techstack': ['React | Node.js', 'Python | Django | SQL'],
                'Links': ['https://example.com/react-project', 'https://example.com/python-project']
            })
            default_data.to_csv(file_path, index=False)
        
        self.data = pd.read_csv(self.file_path)
        
        # Ensure chroma_db directory exists
        chroma_path = os.path.join(os.getcwd(), 'chroma_db')
        os.makedirs(chroma_path, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
        
        # Load portfolio data
        self.load_portfolio()
    
    def load_portfolio(self):
        """
        Load the default portfolio data into the ChromaDB collection if it's empty.
        """
        try:
            if self.collection.count() == 0:
                for _, row in self.data.iterrows():
                    document = f"Techstack: {row['Techstack']}, Links: {row['Links']}"
                    self.collection.add(
                        ids=[str(uuid.uuid4())],
                        documents=[document],
                        metadatas=[{"techstack": row["Techstack"]}]
                    )
        except Exception as e:
            print(f"Error loading portfolio: {e}")
    
    def load_custom_portfolio(self, portfolio_data):
        """
        Load custom portfolio data from a pandas DataFrame into the ChromaDB collection.
        """
        if not isinstance(portfolio_data, pd.DataFrame):
            raise ValueError("portfolio_data must be a pandas DataFrame")
        
        if list(portfolio_data.columns) != ['Techstack', 'Links']:
            raise ValueError("DataFrame must have 'Techstack' and 'Links' columns")
        
        try:
            # Clear existing collection
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)
            
            # Load new portfolio data
            for _, row in portfolio_data.iterrows():
                document = f"Techstack: {row['Techstack']}, Links: {row['Links']}"
                self.collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[document],
                    metadatas=[{"techstack": row["Techstack"]}]
                )
            
            print("Custom portfolio loaded successfully.")
        
        except Exception as e:
            print(f"Error loading custom portfolio: {e}")
            traceback.print_exc()
    
    def query_links(self, skills):
        """
        Query portfolio links based on skills.
        
        Args:
            skills (list or str): Skills to match against portfolio
        
        Returns:
            list: Matching portfolio links
        """
        # Normalize skills input
        skills = [skills] if isinstance(skills, str) else skills
        skills = [str(skill).strip() for skill in skills if skill]
        
        if not skills:
            return []
        
        # Create query string
        query = " OR ".join(skills)
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3
            )
            
            # Extract links from results
            links = []
            for doc in results.get('documents', []):
                if doc:
                    # Extract links from document
                    link_match = re.search(r'Links: (https?://\S+)', str(doc))
                    if link_match:
                        links.append(link_match.group(1))
            
            return links
        
        except Exception as e:
            print(f"Error in query_links: {e}")
            return []
