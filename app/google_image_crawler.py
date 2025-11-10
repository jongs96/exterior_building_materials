#!/usr/bin/env python3
"""
구글 이미지 검색 기반 크롤러
건축 외장재 이미지를 구글 이미지 검색을 통해 수집하는 모듈
"""

import os
import time
import requests
import hashlib
import json
from urllib.parse import urlencode, urlparse
from typing import List, Dict, Optional
import logging
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from PIL import Image
import io

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleImageCrawler:
    """구글 이미지 검색을 통한 이미지 수집 클래스"""
    
    def __init__(self, headless: bool = True, timeout: int = 15):
        """
        구글 이미지 크롤러 초기화
        
        Args:
            headless: 브라우저 헤드리스 모드 여부
            timeout: 페이지 로딩 타임아웃 (초)
        """
        self.timeout = timeout
        self.headless = headless
        
        # Chrome 옵션 설정
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        
        # User-Agent 설정 (봇 차단 방지)
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        # 건축 외장재별 검색 키워드 (한국어 + 영어)
        self.search_keywords = {
            'brick': [
                '조적 외벽',
                '벽돌 시공 사례', 
                '적벽돌 외관',
                '치장 벽돌',
                'brick wall exterior',
                'brick building facade',
                'red brick architecture'
            ],
            'stucco': [
                '스타코 마감 주택',
                '드라이비트 외벽',
                '미장 마감',
                '화이트 스타코',
                'stucco exterior wall',
                'dryvit building',
                'white stucco house'
            ],
            'metal': [
                '징크 패널 시공',
                '금속 외장재',
                '알루미늄 패널 건물',
                'zinc panel building',
                'metal cladding exterior',
                'aluminum facade panel'
            ],
            'stone': [
                '화강석 외벽',
                '석재 외장 마감',
                '대리석 건축물',
                '건축물 석재 타일',
                'granite exterior wall',
                'stone cladding building',
                'marble facade architecture'
            ],
            'wood': [
                '목재 사이딩 시공',
                '우드 외장재',
                '목조 주택 외관',
                '나무 외벽',
                'wood siding exterior',
                'wooden cladding house',
                'timber facade building'
            ]
        }
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _get_driver(self) -> webdriver.Chrome:
        """Chrome 드라이버 인스턴스 생성"""
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception as e:
            logger.error(f"드라이버 생성 실패: {e}")
            raise
    
    def _build_google_images_url(self, keyword: str, image_size: str = 'medium') -> str:
        """
        구글 이미지 검색 URL 생성
        
        Args:
            keyword: 검색 키워드
            image_size: 이미지 크기 ('small', 'medium', 'large')
            
        Returns:
            구글 이미지 검색 URL
        """
        # 구글 이미지 검색 파라미터
        params = {
            'q': keyword,
            'tbm': 'isch',  # 이미지 검색
            'hl': 'ko',     # 한국어
            'safe': 'off',  # 세이프서치 끄기
            'tbs': f'isz:{image_size[0]}'  # 이미지 크기 (m=medium, l=large, s=small)
        }
        
        base_url = 'https://www.google.com/search'
        return f"{base_url}?{urlencode(params)}"
    
    def collect_image_urls_from_keyword(self, keyword: str, max_images: int = 50) -> List[str]:
        """
        특정 키워드로 구글 이미지 검색하여 이미지 URL 수집
        
        Args:
            keyword: 검색 키워드
            max_images: 최대 수집할 이미지 수
            
        Returns:
            이미지 URL 리스트
        """
        driver = None
        image_urls = []
        
        try:
            # 드라이버 초기화
            driver = self._get_driver()
            
            # 구글 이미지 검색 페이지로 이동
            search_url = self._build_google_images_url(keyword)
            logger.info(f"구글 이미지 검색: {keyword}")
            logger.debug(f"검색 URL: {search_url}")
            
            driver.get(search_url)
            
            # 페이지 로딩 대기
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img[data-src], img[src]"))
            )
            
            # 스크롤하여 더 많은 이미지 로드
            self._scroll_and_load_images(driver, max_images)
            
            # 이미지 요소들 찾기
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img[data-src], img[src]")
            logger.info(f"발견된 이미지 요소: {len(img_elements)}개")
            
            for i, img_element in enumerate(img_elements):
                if len(image_urls) >= max_images:
                    break
                
                try:
                    # 이미지 URL 추출 (data-src 우선, 없으면 src)
                    img_url = img_element.get_attribute('data-src') or img_element.get_attribute('src')
                    
                    if img_url and self._is_valid_image_url(img_url):
                        # 구글 프록시 URL을 실제 이미지 URL로 변환
                        actual_url = self._extract_actual_image_url(img_url)
                        if actual_url:
                            image_urls.append(actual_url)
                            logger.debug(f"이미지 URL 수집: {actual_url}")
                
                except Exception as e:
                    logger.debug(f"이미지 요소 처리 오류 (인덱스 {i}): {e}")
                    continue
            
            logger.info(f"키워드 '{keyword}'로 {len(image_urls)}개 이미지 URL 수집 완료")
            return image_urls
            
        except TimeoutException:
            logger.error(f"페이지 로딩 타임아웃: {keyword}")
            return image_urls
        except WebDriverException as e:
            logger.error(f"Selenium 오류: {keyword} - {e}")
            return image_urls
        except Exception as e:
            logger.error(f"예상치 못한 오류: {keyword} - {e}")
            return image_urls
        finally:
            if driver:
                driver.quit()
    
    def _scroll_and_load_images(self, driver: webdriver.Chrome, target_images: int):
        """
        페이지를 스크롤하여 더 많은 이미지 로드
        
        Args:
            driver: Selenium 드라이버
            target_images: 목표 이미지 수
        """
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scroll_attempts = 10
        
        while scroll_attempts < max_scroll_attempts:
            # 페이지 끝까지 스크롤
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # 로딩 대기
            time.sleep(2)
            
            # "결과 더보기" 버튼 클릭 시도
            try:
                more_results_button = driver.find_element(By.CSS_SELECTOR, "input[value*='결과 더보기'], input[value*='Show more results']")
                if more_results_button.is_displayed():
                    driver.execute_script("arguments[0].click();", more_results_button)
                    time.sleep(3)
                    logger.debug("'결과 더보기' 버튼 클릭")
            except NoSuchElementException:
                pass
            except Exception as e:
                logger.debug(f"'결과 더보기' 버튼 클릭 실패: {e}")
            
            # 새로운 높이 확인
            new_height = driver.execute_script("return document.body.scrollHeight")
            
            # 현재 로드된 이미지 수 확인
            current_images = len(driver.find_elements(By.CSS_SELECTOR, "img[data-src], img[src]"))
            logger.debug(f"현재 로드된 이미지: {current_images}개")
            
            # 목표 이미지 수에 도달했거나 더 이상 스크롤할 수 없으면 중단
            if current_images >= target_images or new_height == last_height:
                break
                
            last_height = new_height
            scroll_attempts += 1
        
        logger.debug(f"스크롤 완료: {scroll_attempts}회 시도")
    
    def _extract_actual_image_url(self, google_url: str) -> Optional[str]:
        """
        구글 프록시 URL에서 실제 이미지 URL 추출
        
        Args:
            google_url: 구글 이미지 URL
            
        Returns:
            실제 이미지 URL 또는 None
        """
        try:
            # 구글 이미지 프록시 URL 패턴 확인
            if 'googleusercontent.com' in google_url or 'ggpht.com' in google_url:
                return google_url
            
            # 일반적인 이미지 URL인 경우 그대로 반환
            if any(ext in google_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                return google_url
            
            # 구글 검색 결과 URL에서 실제 URL 추출 시도
            if 'imgurl=' in google_url:
                from urllib.parse import parse_qs, urlparse
                parsed = urlparse(google_url)
                params = parse_qs(parsed.query)
                if 'imgurl' in params:
                    return params['imgurl'][0]
            
            return google_url
            
        except Exception as e:
            logger.debug(f"URL 추출 오류: {google_url} - {e}")
            return None
    
    def _is_valid_image_url(self, url: str) -> bool:
        """
        이미지 URL 유효성 검사
        
        Args:
            url: 검사할 URL
            
        Returns:
            유효성 여부
        """
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        # 구글 로고나 아이콘 등 제외
        exclude_patterns = [
            'logo', 'icon', 'button', 'arrow', 'search',
            'google.com/images/branding', 'gstatic.com',
            'data:image', 'base64'
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in exclude_patterns):
            return False
        
        # 이미지 파일 확장자 또는 구글 이미지 서비스 URL 확인
        valid_patterns = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp',
            'googleusercontent.com', 'ggpht.com'
        ]
        
        return any(pattern in url_lower for pattern in valid_patterns)
    
    def collect_images_for_category(self, category: str, max_images_per_keyword: int = 20) -> List[str]:
        """
        특정 카테고리에 대한 모든 키워드로 이미지 URL 수집
        
        Args:
            category: 카테고리명 ('brick', 'stucco', 'metal', 'stone', 'wood')
            max_images_per_keyword: 키워드당 최대 이미지 수
            
        Returns:
            수집된 이미지 URL 리스트
        """
        if category not in self.search_keywords:
            logger.error(f"지원하지 않는 카테고리: {category}")
            return []
        
        all_image_urls = []
        keywords = self.search_keywords[category]
        
        for keyword in keywords:
            logger.info(f"=== '{keyword}' 키워드 검색 시작 ===")
            
            try:
                urls = self.collect_image_urls_from_keyword(keyword, max_images_per_keyword)
                all_image_urls.extend(urls)
                
                logger.info(f"키워드 '{keyword}': {len(urls)}개 URL 수집")
                
                # 키워드 간 요청 간격 (구글 차단 방지)
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 처리 중 오류: {e}")
                continue
        
        # 중복 제거
        unique_urls = list(set(all_image_urls))
        logger.info(f"카테고리 '{category}' 총 수집: {len(all_image_urls)}개 → 중복 제거 후: {len(unique_urls)}개")
        
        return unique_urls
    
    def collect_all_categories(self, max_images_per_category: int = 100) -> Dict[str, List[str]]:
        """
        모든 카테고리에 대한 이미지 URL 수집
        
        Args:
            max_images_per_category: 카테고리당 최대 이미지 수
            
        Returns:
            카테고리별 이미지 URL 딕셔너리
        """
        results = {}
        
        for category in self.search_keywords.keys():
            logger.info(f"\n🏗️ === {category.upper()} 카테고리 수집 시작 ===")
            
            # 키워드당 이미지 수 계산
            num_keywords = len(self.search_keywords[category])
            images_per_keyword = max(10, max_images_per_category // num_keywords)
            
            try:
                urls = self.collect_images_for_category(category, images_per_keyword)
                
                # 최대 개수 제한
                if len(urls) > max_images_per_category:
                    urls = urls[:max_images_per_category]
                
                results[category] = urls
                logger.info(f"✅ {category} 카테고리 완료: {len(urls)}개 URL")
                
                # 카테고리 간 대기 시간
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ {category} 카테고리 처리 실패: {e}")
                results[category] = []
        
        return results


def test_google_crawler():
    """구글 이미지 크롤러 테스트 함수"""
    print("🔍 구글 이미지 크롤러 테스트")
    print("=" * 50)
    
    # 크롤러 초기화 (헤드리스 모드 비활성화로 테스트)
    crawler = GoogleImageCrawler(headless=False, timeout=20)
    
    # 단일 키워드 테스트
    test_keyword = "벽돌 외벽"
    print(f"테스트 키워드: {test_keyword}")
    
    urls = crawler.collect_image_urls_from_keyword(test_keyword, max_images=10)
    
    print(f"\n수집 결과: {len(urls)}개 URL")
    for i, url in enumerate(urls[:5], 1):
        print(f"  {i}. {url}")
    
    if len(urls) > 5:
        print(f"  ... 외 {len(urls) - 5}개")
    
    return len(urls) > 0


if __name__ == "__main__":
    # 테스트 실행
    success = test_google_crawler()
    
    if success:
        print("\n✅ 구글 이미지 크롤러 테스트 성공!")
    else:
        print("\n❌ 구글 이미지 크롤러 테스트 실패!")