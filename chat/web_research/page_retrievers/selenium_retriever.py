import json
import logging
import os
import time

from bs4 import BeautifulSoup
from selenium.common import StaleElementReferenceException
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.remote.remote_connection import LOGGER
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_attempt, wait_random

from . import PageRetriever
from ..errors import TransientHTTPError, NonTransientHTTPError

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchFrameException
    from webdriver_manager.chrome import ChromeDriverManager
except ModuleNotFoundError as e:
    raise ImportError(
        "Selenium or webdriver_manager is not installed. "
        "Please install these packages to use the SeleniumPageRetriever. "
        "You can do this by running `pip install selenium webdriver_manager`."
    ) from e

LOGGER.setLevel(logging.NOTSET)


class SeleniumPageRetriever(PageRetriever):
    def __init__(self, headless: bool = True, main_page_timeout: int = 30, iframe_timeout: int = 10,
                 main_page_min_wait: int = 2):

        assert main_page_timeout >= main_page_min_wait, "Timeout must be greater than or equal to minimum_wait_time."

        self.chrome_options = Options()

        self.main_page_min_wait = main_page_min_wait
        self.main_page_timeout = main_page_timeout
        self.iframe_timeout = iframe_timeout
        self.headless = headless

        self.configure_chrome_options()

    def configure_chrome_options(self):
        if self.headless:
            self.chrome_options.add_argument("--headless")

        self.chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
        self.chrome_options.add_argument("--disable-gpu")  # Applicable to windows os only
        self.chrome_options.add_argument("start-maximized")  # Open the browser in maximized mode
        self.chrome_options.add_argument("disable-infobars")  # Disabling infobars
        self.chrome_options.add_argument("--disable-extensions")  # Disabling extensions
        self.chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        self.chrome_options.add_argument("--ignore-certificate-errors")  # Ignore certificate errors
        self.chrome_options.add_argument("--incognito")  # Incognito mode
        self.chrome_options.add_argument("--log-level=0")  # To disable the logging
        # To solve tbsCertificate logging issue
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # Enable Performance Logging
        self.chrome_options.set_capability("goog:loggingPrefs", {'performance': 'ALL'})

    def extract_html_from_driver(self, driver: WebDriver) -> str:
        # Wait for minimum time first
        time.sleep(self.main_page_min_wait)

        try:
            # Wait for the main document to be ready
            WebDriverWait(driver, self.main_page_timeout - self.main_page_min_wait).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            # Capture the main document HTML
            main_html = driver.page_source
            soup = BeautifulSoup(main_html, 'html.parser')

            # Find all iframe elements
            iframes = driver.find_elements(By.TAG_NAME, "iframe")

            # Iterate over each iframe, switch to it, and capture its HTML
            for index, iframe in enumerate(iframes):
                try:
                    # Wait for the iframe to be available and for its document to be fully loaded
                    WebDriverWait(driver, self.iframe_timeout).until(
                        lambda d: EC.frame_to_be_available_and_switch_to_it(iframe)(d) and
                                  d.execute_script("return document.readyState") == "complete"
                    )

                    # Capture the iframe HTML
                    iframe_html = driver.page_source
                    iframe_soup = BeautifulSoup(iframe_html, 'html.parser')
                    iframe_body = iframe_soup.find('body')

                    # Insert the iframe body after the iframe element in the main document
                    soup_iframe = soup.find_all('iframe')[index]
                    soup_iframe.insert_after(iframe_body)

                    # Switch back to the main content after each iframe
                    driver.switch_to.default_content()
                except StaleElementReferenceException:
                    # If the iframe is no longer available, skip it
                    continue

            # The soup object now contains the modified HTML
            full_html = str(soup)

            return full_html
        except (WebDriverException, NoSuchFrameException) as e:
            return f'An error occurred while retrieving the page: {e}'

    @retry(retry=retry_if_exception_type(TransientHTTPError),
           wait=wait_fixed(2) + wait_random(0, 2),
           stop=stop_after_attempt(5))
    def retrieve_html(self, url: str) -> str:
        driver = None
        service = None
        try:
            service = Service(ChromeDriverManager().install(), log_output=os.devnull)
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            driver.get(url)

            # Wait and extract the HTML
            full_html = self.extract_html_from_driver(driver)

            # Now retrieve the logs and check the status code
            logs = driver.get_log("performance")
            status_code = None
            for entry in logs:
                log = json.loads(entry["message"])["message"]
                if log["method"] == "Network.responseReceived" and "response" in log["params"]:
                    status_code = log["params"]["response"]["status"]
                    break

            if status_code is None:
                raise Exception("No HTTP response received.")
            elif status_code >= 500:
                raise TransientHTTPError(status_code, "Server error encountered.")
            elif 400 <= status_code < 500:
                raise NonTransientHTTPError(status_code, "Client error encountered.")

            return full_html  # or driver.page_source if you wish to return the original source
        except TimeoutException:
            raise TransientHTTPError(408, "Timeout while waiting for the page to load.")
        finally:
            if driver:
                driver.quit()

            if service:
                service.stop()
