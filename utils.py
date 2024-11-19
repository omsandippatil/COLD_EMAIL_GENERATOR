import re

def clean_text(text):
    """
    Clean the input text by performing several preprocessing steps:
    - Remove HTML tags
    - Remove URLs
    - Remove special characters
    - Replace multiple spaces with a single space
    - Trim leading and trailing whitespace
    """
    try:
        # Remove HTML tags
        text = re.sub(r'<[^>]*?>', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s{2,}', ' ', text)
        # Trim leading and trailing whitespace
        text = text.strip()
        # Remove extra whitespace between words
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text  # Return the original text in case of an error
