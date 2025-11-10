#!/usr/bin/env python3
"""
개선된 구글 이미지 크롤러
실제 이미지 클릭을 통해 원본 URL을 추출하는 방식
"""

import os
import time
import requests
import json
from urllib.parse import urlencode, urlparse, parse_qs
from typing import List, Dict, Optional
import logging
from pathlib import Path
import re

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedGoogleImageCrawler:
    """개선된 구글 이미지 크롤러 - 실제 이미지 클릭 방식"""
    
    def __init__(self, headless: bool = True, timeout: int = 20):
        """
        크롤러 초기화
        
        Args:
            headless: 헤드리스 모드 여부
            timeout: 타임아웃 (초)
        """
        self.timeout = timeout
        self.headless = headless
        
        # Chrome 옵션 설정
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless')
        
        # 기본 옵션들
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        # User-Agent 설정
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        # 검색 키워드 (한국어 중심)
        self.search_keywords = {
            'brick': [
                '조적 외벽',
                '벽돌 외관',
                '적벽돌 건물',
                '치장벽돌',
                'brick exterior wall'
            ],
            'stucco': [
                '스타코 외벽',
                '드라이비트 마감',
                '미장 외벽',
                '화이트 스타코',
                'stucco exterior'
            ],
            'metal': [
                '징크 패널',
                '금속 외장재',
                '알루미늄 패널',
                '금속 사이딩',
                'metal panel facade'
            ],
            'stone': [
                '석재 외벽',
                '화강석 외장',
                '대리석 외벽',
                '석재 마감',
                'stone cladding'
            ],
            'wood': [
                '목재 사이딩',
                '우드 외장재',
                '목재 외벽',
                '나무 사이딩',
                'wood siding'
            ]
        }
    
    def _get_driver(self) -> webdriver.Chrome:
        """Chrome 드라이버 생성"""
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            driver.set_page_load_timeout(self.timeout)
            
            # 자동화 감지 방지
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            return driver
        except Exception as e:
            logger.error(f"드라이버 생성 실패: {e}")
            raise
    
    def collect_image_urls_from_keyword(self, keyword: str, max_images: int = 30) -> List[str]:
        """
        키워드로 구글 이미지 검색하여 URL 수집
        
        Args:
            keyword: 검색 키워드
            max_images: 최대 이미지 수
            
        Returns:
            이미지 URL 리스트
        """
        driver = None
        image_urls = []
        
        try:
            driver = self._get_driver()
            
            # 구글 이미지 검색 URL 생성
            search_url = f"https://www.google.com/search?q={keyword}&tbm=isch&hl=ko"
            logger.info(f"검색 시작: {keyword}")
            
            driver.get(search_url)
            time.sleep(3)
            
            # 쿠키 동의 버튼 처리
            try:
                accept_button = driver.find_element(By.XPATH, "//button[contains(text(), '모두 허용') or contains(text(), 'Accept all')]")
                accept_button.click()
                time.sleep(2)
            except:
                pass
            
            # 이미지 썸네일들 찾기
            self._scroll_to_load_images(driver, max_images)
            
            # 이미지 썸네일 요소들 가져오기
            thumbnail_elements = driver.find_elements(By.CSS_SELECTOR, "img[data-src]")
            logger.info(f"발견된 썸네일: {len(thumbnail_elements)}개")
            
            # 각 썸네일을 클릭하여 원본 이미지 URL 추출
            for i, thumbnail in enumerate(thumbnail_elements[:max_images]):
                if len(image_urls) >= max_images:
                    break
                
                try:
                    # 썸네일 클릭
                    driver.execute_script("arguments[0].click();", thumbnail)
                    time.sleep(1.5)
                    
                    # 원본 이미지 URL 추출 시도
                    original_url = self._extract_original_image_url(driver)
                    
                    if original_url and self._is_valid_image_url(original_url):
                        image_urls.append(original_url)
                        logger.debug(f"이미지 URL 수집 ({len(image_urls)}/{max_images}): {original_url}")
                    
                except Exception as e:
                    logger.debug(f"썸네일 {i} 처리 오류: {e}")
                    continue
            
            logger.info(f"키워드 '{keyword}': {len(image_urls)}개 URL 수집 완료")
            return image_urls
            
        except Exception as e:
            logger.error(f"검색 오류 ({keyword}): {e}")
            return image_urls
        finally:
            if driver:
                driver.quit()
    
    def _scroll_to_load_images(self, driver: webdriver.Chrome, target_count: int):
        """이미지 로딩을 위한 스크롤"""
        scroll_count = 0
        max_scrolls = 5
        
        while scroll_count < max_scrolls:
            # 페이지 끝까지 스크롤
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # 현재 로드된 이미지 수 확인
            current_images = len(driver.find_elements(By.CSS_SELECTOR, "img[data-src]"))
            
            if current_images >= target_count:
                break
            
            # "결과 더보기" 버튼 클릭 시도
            try:
                more_button = driver.find_element(By.XPATH, "//input[@value='결과 더보기' or @value='Show more results']")
                if more_button.is_displayed():
                    driver.execute_script("arguments[0].click();", more_button)
                    time.sleep(3)
            except:
                pass
            
            scroll_count += 1
    
    def _extract_original_image_url(self, driver: webdriver.Chrome) -> Optional[str]:
        """
        클릭된 이미지에서 원본 URL 추출
        
        Args:
            driver: Selenium 드라이버
            
        Returns:
            원본 이미지 URL 또는 None
        """
        try:
            # 방법 1: 오른쪽 패널의 큰 이미지에서 URL 추출
            selectors = [
                "img[jsname='kn3ccd']",  # 구글 이미지 뷰어의 메인 이미지
                "img[jsname='HiaYvf']",  # 다른 버전의 메인 이미지
                ".n3VNCb img",           # 이미지 컨테이너 내 이미지
                ".islrc img"             # 이미지 결과 컨테이너
            ]
            
            for selector in selectors:
                try:
                    img_element = driver.find_element(By.CSS_SELECTOR, selector)
                    img_url = img_element.get_attribute('src')
                    
                    if img_url and self._is_valid_image_url(img_url):
                        return img_url
                except:
                    continue
            
            # 방법 2: 페이지 소스에서 이미지 URL 패턴 찾기
            page_source = driver.page_source
            
            # 일반적인 이미지 URL 패턴 검색
            url_patterns = [
                r'https://[^"\']*\.(?:jpg|jpeg|png|webp)[^"\']*',
                r'"(https://[^"]*\.(?:jpg|jpeg|png|webp)[^"]*)"',
                r"'(https://[^']*\.(?:jpg|jpeg|png|webp)[^']*)'"
            ]
            
            for pattern in url_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                for match in matches:
                    url = match if isinstance(match, str) else match[0]
                    if self._is_valid_image_url(url) and 'googleusercontent' not in url:
                        return url
            
            return None
            
        except Exception as e:
            logger.debug(f"원본 URL 추출 오류: {e}")
            return None
    
    def _is_valid_image_url(self, url: str) -> bool:
        """이미지 URL 유효성 검사"""
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        # 제외할 패턴들
        exclude_patterns = [
            'google.com/images/branding',
            'gstatic.com',
            'data:image',
            'base64',
            'logo',
            'icon',
            'button'
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in exclude_patterns):
            return False
        
        # 유효한 이미지 확장자 확인
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        return any(ext in url_lower for ext in valid_extensions)
    
    def collect_images_for_category(self, category: str, max_images_per_keyword: int = 15) -> List[str]:
        """카테고리별 이미지 URL 수집"""
        if category not in self.search_keywords:
            logger.error(f"지원하지 않는 카테고리: {category}")
            return []
        
        all_urls = []
        keywords = self.search_keywords[category]
        
        for keyword in keywords:
            try:
                urls = self.collect_image_urls_from_keyword(keyword, max_images_per_keyword)
                all_urls.extend(urls)
                
                # 키워드 간 대기
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 처리 실패: {e}")
                continue
        
        # 중복 제거
        unique_urls = list(set(all_urls))
        logger.info(f"카테고리 '{category}': {len(all_urls)}개 → 중복제거 후 {len(unique_urls)}개")
        
        return unique_urls
    
    def collect_all_categories(self, max_images_per_category: int = 50) -> Dict[str, List[str]]:
        """모든 카테고리 이미지 수집"""
        results = {}
        
        for category in self.search_keywords.keys():
            logger.info(f"\n🏗️ === {category.upper()} 카테고리 수집 시작 ===")
            
            # 키워드당 이미지 수 계산
            num_keywords = len(self.search_keywords[category])
            images_per_keyword = max(5, max_images_per_category // num_keywords)
            
            try:
                urls = self.collect_images_for_category(category, images_per_keyword)
                
                # 최대 개수 제한
                if len(urls) > max_images_per_category:
                    urls = urls[:max_images_per_category]
                
                results[category] = urls
                logger.info(f"✅ {category} 완료: {len(urls)}개 URL")
                
                # 카테고리 간 대기
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ {category} 실패: {e}")
                results[category] = []
        
        return results


def test_improved_crawler():
    """개선된 크롤러 테스트"""
    print("🔍 개선된 구글 이미지 크롤러 테스트")
    print("=" * 50)
    
    # 헤드리스 모드 비활성화로 테스트 (디버깅용)
    crawler = ImprovedGoogleImageCrawler(headless=False, timeout=30)
    
    # 단일 키워드 테스트
    test_keyword = "벽돌 외벽"
    print(f"테스트 키워드: {test_keyword}")
    
    urls = crawler.collect_image_urls_from_keyword(test_keyword, max_images=5)
    
    print(f"\n수집 결과: {len(urls)}개 URL")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")
    
    return len(urls) > 0


if __name__ == "__main__":
    success = test_improved_crawler()
    
    if success:
        print("\n✅ 개선된 크롤러 테스트 성공!")
    else:
        print("\n❌ 개선된 크롤러 테스트 실패!")