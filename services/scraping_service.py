"""
Service for web scraping football fixtures and data
"""

import os
import time
import logging
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
# import undetected_chromedriver as uc  # Temporarily disabled
from datetime import datetime, timedelta
import re

from config import FIXTURE_URLS, CHROME_OPTIONS, DELAY_SECONDS, MAX_RETRIES

logger = logging.getLogger(__name__)

class ScrapingService:
    """Service for scraping football fixtures and data"""
    
    def __init__(self):
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_fixtures(self, league_name: str) -> Optional[List[Dict]]:
        """Scrape upcoming fixtures for a specific league"""
        try:
            if league_name not in FIXTURE_URLS:
                logger.error(f"League {league_name} not supported for fixture scraping")
                return None
            
            url = FIXTURE_URLS[league_name]
            logger.info(f"Scraping fixtures for {league_name} from {url}")
            
            # Try different scraping methods
            fixtures = self._scrape_with_selenium(url)
            
            if not fixtures:
                fixtures = self._scrape_with_requests(url)
            
            if not fixtures:
                fixtures = self._scrape_with_trafilatura(url)
            
            if fixtures:
                logger.info(f"Successfully scraped {len(fixtures)} fixtures for {league_name}")
                return fixtures
            else:
                logger.error(f"Failed to scrape fixtures for {league_name}")
                return None
            
        except Exception as e:
            logger.error(f"Error scraping fixtures for {league_name}: {str(e)}")
            return None
    
    def _scrape_with_selenium(self, url: str) -> Optional[List[Dict]]:
        """Scrape fixtures using Selenium WebDriver"""
        try:
            if not self._init_driver():
                return None
            
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Give time for dynamic content to load
            time.sleep(3)
            
            # Look for common fixture selectors
            fixtures = []
            
            # Try different common selectors for fixture data
            selectors = [
                '.fixture-row',
                '.match-row',
                '.fixture',
                '.match',
                '[data-fixture]',
                '.game-row'
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        fixtures = self._parse_fixture_elements(elements)
                        if fixtures:
                            break
                except:
                    continue
            
            # If no specific selectors work, try parsing the entire page content
            if not fixtures:
                page_source = self.driver.page_source
                fixtures = self._parse_page_content(page_source)
            
            return fixtures
            
        except Exception as e:
            logger.error(f"Error scraping with Selenium: {str(e)}")
            return None
        finally:
            self._close_driver()
    
    def _scrape_with_requests(self, url: str) -> Optional[List[Dict]]:
        """Scrape fixtures using requests and BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for fixture data in common patterns
            fixtures = []
            
            # Try to find fixture containers
            fixture_containers = soup.find_all(['div', 'tr', 'li'], class_=re.compile(r'fixture|match|game', re.I))
            
            for container in fixture_containers:
                fixture = self._extract_fixture_from_element(container)
                if fixture:
                    fixtures.append(fixture)
            
            # If no specific containers found, try parsing text content
            if not fixtures:
                text_content = soup.get_text()
                fixtures = self._parse_text_for_fixtures(text_content)
            
            return fixtures
            
        except Exception as e:
            logger.error(f"Error scraping with requests: {str(e)}")
            return None
    
    def _scrape_with_trafilatura(self, url: str) -> Optional[List[Dict]]:
        """Scrape fixtures using trafilatura for content extraction"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            
            text_content = trafilatura.extract(downloaded)
            if not text_content:
                return None
            
            # Parse the extracted text for fixture information
            fixtures = self._parse_text_for_fixtures(text_content)
            
            return fixtures
            
        except Exception as e:
            logger.error(f"Error scraping with trafilatura: {str(e)}")
            return None
    
    def _init_driver(self) -> bool:
        """Initialize Selenium WebDriver"""
        try:
            # Configure Chrome options
            chrome_options = Options()
            for option in CHROME_OPTIONS:
                chrome_options.add_argument(option)
            
            # Try undetected chrome driver first
            try:
                self.driver = uc.Chrome(options=chrome_options)
                return True
            except:
                pass
            
            # Fallback to regular Chrome driver
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
                return True
            except:
                pass
            
            logger.error("Failed to initialize Chrome WebDriver")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing WebDriver: {str(e)}")
            return False
    
    def _close_driver(self):
        """Close Selenium WebDriver"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
        except:
            pass
    
    def _parse_fixture_elements(self, elements) -> List[Dict]:
        """Parse fixture elements from Selenium"""
        fixtures = []
        
        for element in elements:
            try:
                fixture = self._extract_fixture_from_selenium_element(element)
                if fixture:
                    fixtures.append(fixture)
            except:
                continue
        
        return fixtures
    
    def _extract_fixture_from_selenium_element(self, element) -> Optional[Dict]:
        """Extract fixture information from a Selenium element"""
        try:
            text = element.text.strip()
            if not text:
                return None
            
            # Look for team names and match information
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                fixture = self._parse_fixture_line(line)
                if fixture:
                    return fixture
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting fixture from element: {str(e)}")
            return None
    
    def _extract_fixture_from_element(self, element) -> Optional[Dict]:
        """Extract fixture information from a BeautifulSoup element"""
        try:
            text = element.get_text().strip()
            if not text:
                return None
            
            # Look for team names and match information
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                fixture = self._parse_fixture_line(line)
                if fixture:
                    return fixture
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting fixture from element: {str(e)}")
            return None
    
    def _parse_text_for_fixtures(self, text: str) -> List[Dict]:
        """Parse text content for fixture information"""
        fixtures = []
        
        try:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                fixture = self._parse_fixture_line(line)
                if fixture:
                    fixtures.append(fixture)
            
        except Exception as e:
            logger.debug(f"Error parsing text for fixtures: {str(e)}")
        
        return fixtures
    
    def _parse_fixture_line(self, line: str) -> Optional[Dict]:
        """Parse a single line for fixture information"""
        try:
            # Common patterns for fixtures
            patterns = [
                r'(\w+(?:\s+\w+)*)\s+vs?\s+(\w+(?:\s+\w+)*)',  # Team1 vs Team2
                r'(\w+(?:\s+\w+)*)\s+-\s+(\w+(?:\s+\w+)*)',    # Team1 - Team2
                r'(\w+(?:\s+\w+)*)\s+v\s+(\w+(?:\s+\w+)*)',    # Team1 v Team2
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    home_team = match.group(1).strip()
                    away_team = match.group(2).strip()
                    
                    # Basic validation
                    if len(home_team) > 2 and len(away_team) > 2 and home_team != away_team:
                        # Try to extract date/time if available
                        date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}', line)
                        match_date = date_match.group() if date_match else None
                        
                        return {
                            'home_team': home_team,
                            'away_team': away_team,
                            'date': match_date,
                            'source_line': line
                        }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing fixture line '{line}': {str(e)}")
            return None
    
    def _parse_page_content(self, content: str) -> List[Dict]:
        """Parse entire page content for fixtures"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text()
            return self._parse_text_for_fixtures(text_content)
        except Exception as e:
            logger.error(f"Error parsing page content: {str(e)}")
            return []
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        self._close_driver()