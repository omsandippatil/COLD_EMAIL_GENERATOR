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
        Query the portfolio for matching tech stacks and return corresponding links.

        Args:
            skills (str | list): A skill or list of skills to query.

        Returns:
            list: A list of [techstack, link] pairs that match the query.
        """
        # Ensure skills is a list
        if isinstance(skills, str):
            skills = [skills]

        try:
            # Join skills with OR for broader matching
            query = " OR ".join(skills)

            # Perform the query
            results = self.collection.query(
                query_texts=[query],
                n_results=2
            )

            # Safely extract links
            links_list = []
            if results.get('documents') and results['documents']:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    # Extract links using regex or string parsing
                    link_match = re.search(r'Links: (.*)', doc)
                    if link_match:
                        links_list.append([
                            metadata.get('techstack', ''),
                            link_match.group(1)
                        ])

            return links_list

        except Exception as e:
            print(f"Error in query_links: {e}")
            traceback.print_exc()
            return []
