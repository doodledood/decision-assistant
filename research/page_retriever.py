import abc
import os
from typing import Optional

import requests


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

    def retrieve_html(self, url: str) -> str:
        payload = {
            'api_key': self.api_key,
            'url': url,
            'render': self.render_js
        }
        r = requests.get('https://api.scraperapi.com/', params=payload)

        r.raise_for_status()

        return r.text
