"""
HotpotQA-specific tools: Search and Retrieve functions.
"""

from typing import Dict, List


def create_search_fn(context: Dict[str, List]):
    """
    Create a Search function for a specific HotpotQA example.
    Returns titles of paragraphs that match the query (fuzzy match).
    
    Args:
        context: HotpotQA context dict with 'title' and 'sentences'
    
    Returns:
        Async function that takes a query string and returns matching document titles
    """
    titles = context.get("title", [])
    
    async def search(query: str) -> str:
        """Search for relevant document titles using fuzzy matching."""
        query_lower = query.lower()
        matching_titles = []
        
        # Fuzzy match: check if query words appear in title or sentences
        for i, title in enumerate(titles):
            title_lower = title.lower()
            # Check if query words appear in title
            if any(word in title_lower for word in query_lower.split() if len(word) > 2):
                matching_titles.append(title)
            # Also check first sentence of each paragraph
            elif i < len(context.get("sentences", [])):
                first_sent = " ".join(context["sentences"][i][:1]).lower() if context["sentences"][i] else ""
                if any(word in first_sent for word in query_lower.split() if len(word) > 2):
                    matching_titles.append(title)
        
        # If no matches, return all titles (fallback)
        if not matching_titles:
            matching_titles = titles[:5]  # Return top 5 as fallback
        
        # Format as list string
        if matching_titles:
            return f"Found {len(matching_titles)} relevant document(s): {', '.join(matching_titles)}"
        else:
            return f"No documents found. Available documents: {', '.join(titles[:10])}"
    
    return search


def create_retrieve_fn(context: Dict[str, List]):
    """
    Create a Retrieve function for a specific HotpotQA example.
    Returns the actual paragraph text for a given title.
    
    Args:
        context: HotpotQA context dict with 'title' and 'sentences'
    
    Returns:
        Async function that takes a title string and returns the paragraph text
    """
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    
    # Create title -> full text mapping
    title_to_text = {}
    for i, title in enumerate(titles):
        if i < len(sentences):
            # Join all sentences for this paragraph
            paragraph_text = " ".join(sentences[i])
            title_to_text[title] = paragraph_text
    
    async def retrieve(title: str) -> str:
        """Retrieve the full paragraph text for a given title."""
        # Try exact match first
        if title in title_to_text:
            return title_to_text[title]
        
        # Try fuzzy match (case-insensitive)
        title_lower = title.lower()
        for doc_title, text in title_to_text.items():
            if doc_title.lower() == title_lower:
                return text
        
        # Try partial match
        for doc_title, text in title_to_text.items():
            if title_lower in doc_title.lower() or doc_title.lower() in title_lower:
                return text
        
        # Not found - return available titles
        available = ", ".join(list(title_to_text.keys())[:10])
        return f"Document '{title}' not found. Available documents: {available}"
    
    return retrieve
