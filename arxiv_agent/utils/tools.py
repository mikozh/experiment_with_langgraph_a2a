import arxiv
import httpx
from langchain_core.tools import tool


@tool
def search_for_publications_in_arxiv(search_string: str,
                                     top_k: int = 10):
    """Use this to search for the arxiv papers summaries chunks relevant to the search string.

    Args:
        search_string: the search string (e.g., "quantum dots").
        top_k: Number of the top K best matching results to return (e.g., 15).
    Returns:
        A dictionary containing the document title, summary and pdf url, or an error message if
        the request fails.
    """
    try:
        arxiv_client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=3)
        search = arxiv.Search(query=search_string,
                              sort_by=arxiv.SortCriterion.Relevance,
                              sort_order=arxiv.SortOrder.Descending, max_results=100)
        results_generator = arxiv_client.results(search)

        result = []
        for paper in results_generator:
            result.append({"title": paper.title,
                           "summary": paper.summary,
                           "pdf_url": paper.pdf_url})
        return result
    except httpx.HTTPError as e:
        return {'error': f'API request failed: {e}'}
    except ValueError:
        return {'error': 'Invalid JSON response from API.'}