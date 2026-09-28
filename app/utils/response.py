from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

def format_response(text: str, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Format the final response with optional citation of sources

    Args:
        text: Response text
        sources: Optional list of source dictionaries

    Returns:
        Formatted response dictionary
    """
    if not text:
        return {"text": "", "sources": []}

    response = {
        "text": text,
        "sources": []
    }

    if sources:
        response["sources"] = format_sources(sources)

    return response

def format_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format source information for consistent output

    Args:
        sources: List of source dictionaries

    Returns:
        Formatted source list
    """
    if not sources:
        return []

    formatted_sources = []

    for source in sources:
        # Create a standardized source object
        formatted_source = {
            "title": source.get("title", "Unknown"),
            "url": source.get("url", "")
        }

        # Add snippet if available
        if "content" in source:
            snippet = source["content"]
            # Truncate long snippets
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            formatted_source["snippet"] = snippet

        # Add source type if available
        if "source" in source:
            formatted_source["type"] = source["source"]

        formatted_sources.append(formatted_source)

    return formatted_sources

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to a maximum length while preserving complete sentences

    Args:
        text: Text to truncate
        max_length: Maximum length in characters

    Returns:
        Truncated text
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    try:
        # Find the last sentence end before max_length
        end_markers = ['. ', '! ', '? ']
        last_end = 0

        for i in range(min(max_length, len(text))):
            for marker in end_markers:
                if i + len(marker) <= len(text) and text[i:i+len(marker)] == marker:
                    last_end = i + 1

        # If no sentence end found, just truncate at max_length
        if last_end == 0:
            return text[:max_length] + "..."

        return text[:last_end+1]
    except Exception as e:
        logger.error(f"Error truncating text: {str(e)}")
        return text[:max_length] + "..."

def clean_html(text: str) -> str:
    """Remove HTML tags from text

    Args:
        text: Text containing HTML tags

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    try:
        # Remove HTML tags
        clean = re.sub(r'<.*?>', '', text)

        # Replace HTML entities
        clean = clean.replace('&nbsp;', ' ')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&quot;', '"')
        clean = clean.replace('&#39;', "'")

        # Remove excessive whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean
    except Exception as e:
        logger.error(f"Error cleaning HTML from text: {str(e)}")
        return text

def highlight_matches(text: str, query_terms: List[str]) -> str:
    """Highlight query terms in text

    Args:
        text: Original text
        query_terms: List of terms to highlight

    Returns:
        Text with highlighted terms
    """
    if not text or not query_terms:
        return text

    try:
        highlighted = text

        for term in query_terms:
            if term.strip():
                # Create pattern that matches whole words
                pattern = r'\b(' + re.escape(term.strip()) + r')\b'

                # Replace with highlighted version
                highlighted = re.sub(pattern, r'**\1**', highlighted, flags=re.IGNORECASE)

        return highlighted
    except Exception as e:
        logger.error(f"Error highlighting matches: {str(e)}")
        return text
