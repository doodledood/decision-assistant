import json
import time
from typing import Optional

from ..errors import TransientHTTPError, NonTransientHTTPError
from .base import PageRetriever

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


class SeleniumPageRetriever:
    def __init__(self, headless: bool = True, timeout: int = 30, minimum_wait_time: int = 2,
                 wait_and_extract_html: Optional[callable] = None):

        assert timeout >= minimum_wait_time, "Timeout must be greater than or equal to minimum_wait_time."
        
        self.chrome_options = Options()

        if headless:
            self.chrome_options.add_argument("--headless")

        self.chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
        self.chrome_options.add_argument("--disable-gpu")  # Applicable to windows os only
        self.chrome_options.add_argument("start-maximized")  # Open the browser in maximized mode
        self.chrome_options.add_argument("disable-infobars")  # Disabling infobars
        self.chrome_options.add_argument("--disable-extensions")  # Disabling extensions
        self.chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

        # Enable Performance Logging
        self.chrome_options.set_capability("goog:loggingPrefs", {'performance': 'ALL'})

        self.service = Service(ChromeDriverManager().install())
        self.minimum_wait_time = minimum_wait_time
        self.timeout = timeout

        if wait_and_extract_html is None:
            # The default wait condition waits for document readiness and all iframes as well.
            self.wait_and_extract_html = self.default_wait_and_extract_html
        else:
            self.wait_and_extract_html = wait_and_extract_html

    def default_wait_and_extract_html(self, driver):
        # Wait for minimum time first
        time.sleep(self.minimum_wait_time)

        try:
            # Wait for the main document to be ready
            WebDriverWait(driver, self.timeout - self.minimum_wait_time).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            # Retrieve all iframe elements
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            main_html = driver.page_source  # Capture the main document HTML

            # Function to switch to an available frame and capture its HTML
            def capture_frame_html(frame):
                WebDriverWait(driver, self.timeout).until(
                    EC.frame_to_be_available_and_switch_to_it(frame)
                )
                frame_html = driver.page_source  # Get the iframe's HTML
                driver.switch_to.default_content()  # Switch back to main content
                return frame_html

            # Iterate over each iframe, switch to it, and capture its HTML
            for iframe in iframes:
                iframe_html = capture_frame_html(iframe)
                frame_id = iframe.get_attribute('id')
                # Replace the iframe placeholder in the main HTML with the actual iframe content
                main_html = main_html.replace(f'<iframe id="{frame_id}"',
                                              f'<iframe id="{frame_id}">{iframe_html}</iframe>', 1)

            return main_html  # Return modified main HTML including iframe contents
        except (WebDriverException, NoSuchFrameException, TimeoutException):
            return False

    def retrieve_html(self, url: str) -> str:
        try:
            with webdriver.Chrome(service=self.service, options=self.chrome_options) as driver:
                driver.get(url)

                html = self.wait_and_extract_html(driver)
                if not html:
                    raise Exception("Failed to load the page correctly with iframes.")

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

                return html
        except TimeoutException:
            raise TransientHTTPError(408, "Timeout while waiting for the page to load.")
