import abc
import os
from typing import Optional

import requests
from tenacity import retry, wait_random, wait_fixed, stop_after_attempt, retry_if_exception_type

from chat.web_research.errors import TransientHTTPError, NonTransientHTTPError


class PageRetriever(abc.ABC):
    def retrieve_html(self, url: str) -> str:
        raise NotImplementedError()


class ScraperAPIPageRetriever(PageRetriever):
    def __init__(self, api_key: Optional[str] = None, render_js: bool = False):
        if api_key is None:
            if 'SCRAPERAPI_API_KEY' not in os.environ:
                raise ValueError('SCRAPERAPI_API_KEY environment variable is required or api_key argument.')

            api_key = os.environ['SCRAPERAPI_API_KEY']

        self.api_key = api_key
        self.render_js = render_js

    @retry(retry=retry_if_exception_type(TransientHTTPError),
           wait=wait_fixed(2) + wait_random(0, 2),
           stop=stop_after_attempt(5))
    def retrieve_html(self, url: str) -> str:
        payload = {
            'api_key': self.api_key,
            'url': url,
            'render': self.render_js
        }
        r = requests.get('https://api.scraperapi.com/', params=payload)

        if r.status_code < 300:
            return r.text

        if r.status_code >= 500:
            raise TransientHTTPError(r.status_code, r.text)

        raise NonTransientHTTPError(r.status_code, r.text)


class SimpleRequestsPageRetriever(PageRetriever):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @retry(retry=retry_if_exception_type(TransientHTTPError),
           wait=wait_fixed(2) + wait_random(0, 2),
           stop=stop_after_attempt(5))
    def retrieve_html(self, url: str) -> str:
        r = requests.get(url, **self.kwargs)
        if r.status_code < 300:
            return r.text

        if r.status_code >= 500:
            raise TransientHTTPError(r.status_code, r.text)

        raise NonTransientHTTPError(r.status_code, r.text)
