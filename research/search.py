import abc
from typing import List, Optional

from pydantic import BaseModel
from langchain.utilities import GoogleSerperAPIWrapper


class OrganicSearchResult(BaseModel):
    position: int
    title: str
    link: str


class SearchResults(BaseModel):
    answer_snippet: Optional[str]
    organic_results: List[OrganicSearchResult]


class SearchResultsProvider(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, n_results: int = 3) -> SearchResults:
        raise NotImplementedError()


class GoogleSerperSearchResultsProvider(SearchResultsProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def search(self, query: str, n_results: int = 3) -> SearchResults:
        assert n_results > 0, 'n_results must be greater than 0'

        api_wrapper = GoogleSerperAPIWrapper(serper_api_key=self.api_key, k=n_results)
        results = api_wrapper.results(query)

        return SearchResults(
            snippet=results.get('answerBox', {}).get('snippet'),
            organic_results=[
                OrganicSearchResult(
                    position=organic_result['position'],
                    title=organic_result['title'],
                    link=organic_result['link']
                ) for organic_result in results['organic'][:n_results]
            ]
        )
