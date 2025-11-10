#!/usr/bin/env python3
"""
건축 외장재 전문 이미지 수집기
건축 관련 웹사이트에서 실제 외장재 이미지를 수집하는 모듈
"""

import os
import time
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import logging
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArchitectureImageCollector:
    """건축 외장재 전문 이미지 수집 클래스"""
    
    def __init__(self, headless: bool = True):
        """
        초기화
        
        Args:
            headless: 헤드리스 모드 여부
        """
        self.headless = headless
        
        # Chrome 옵션 설정
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--window-size=1920,1080')
        
        # 건축 관련 웹사이트 URL들
        self.target_sites = {
            'brick': [
                'https://www.archdaily.com/search/projects/categories/houses?ad_name=brick',
                'https://www.dezeen.com/tag/brick/',
                'https://architizer.com/materials/masonry/brick/',
            ],
            'stucco': [
                'https://www.archdaily.com/search/projects?q=stucco',
                'https://www.dezeen.com/tag/stucco/',
                'https://architizer.com/materials/stucco/',
            ],
            'metal': [
                'https://www.archdaily.com/search/projects?q=metal+cladding',
                'https://www.dezeen.com/tag/metal-cladding/',
                'https://architizer.com/materials/metal/',
            ],
            'stone': [
                'https://www.archdaily.com/search/projects?q=stone+cladding',
                'https://www.dezeen.com/tag/stone/',
                'https://architizer.com/materials/stone/',
            ],
            'wood': [
                'https://www.archdaily.com/search/projects?q=wood+siding',
                'https://www.dezeen.com/tag/wood-cladding/',
                'https://architizer.com/materials/wood/',
            ]
        }
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get_driver(self) -> webdriver.Chrome:
        """Chrome 드라이버 생성"""
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            driver.set_page_load_timeout(30)
            return driver
        except Exception as e:
            logger.error(f"드라이버 생성 실패: {e}")
            raise
    
    def collect_from_archdaily(self, search_url: str, max_images: int = 20) -> List[str]:
        """
        ArchDaily에서 이미지 수집
        
        Args:
            search_url: 검색 URL
            max_images: 최대 이미지 수
            
        Returns:
            이미지 URL 리스트
        """
        driver = None
        image_urls = []
        
        try:
            driver = self._get_driver()
            logger.info(f"ArchDaily 접속: {search_url}")
            
            driver.get(search_url)
            time.sleep(5)
            
            # 프로젝트 링크들 수집
            project_links = []
            try:
                project_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/projects/']")
                for element in project_elements[:10]:  # 최대 10개 프로젝트
                    href = element.get_attribute('href')
                    if href and href not in project_links:
                        project_links.append(href)
            except Exception as e:
                logger.debug(f"프로젝트 링크 수집 오류: {e}")
            
            # 각 프로젝트 페이지에서 이미지 수집
            for project_url in project_links:
                if len(image_urls) >= max_images:
                    break
                
                try:
                    driver.get(project_url)
                    time.sleep(3)
                    
                    # 이미지 요소들 찾기
                    img_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='images.adsttc.com']")
                    
                    for img in img_elements:
                        if len(image_urls) >= max_images:
                            break
                        
                        img_src = img.get_attribute('src')
                        if img_src and self._is_valid_architecture_image(img_src):
                            image_urls.append(img_src)
                
                except Exception as e:
                    logger.debug(f"프로젝트 페이지 처리 오류: {project_url} - {e}")
                    continue
            
            logger.info(f"ArchDaily에서 {len(image_urls)}개 이미지 URL 수집")
            return image_urls
            
        except Exception as e:
            logger.error(f"ArchDaily 수집 오류: {e}")
            return image_urls
        finally:
            if driver:
                driver.quit()
    
    def collect_from_dezeen(self, search_url: str, max_images: int = 20) -> List[str]:
        """
        Dezeen에서 이미지 수집
        
        Args:
            search_url: 검색 URL
            max_images: 최대 이미지 수
            
        Returns:
            이미지 URL 리스트
        """
        driver = None
        image_urls = []
        
        try:
            driver = self._get_driver()
            logger.info(f"Dezeen 접속: {search_url}")
            
            driver.get(search_url)
            time.sleep(5)
            
            # 기사 링크들 수집
            article_links = []
            try:
                article_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/20']")  # 연도가 포함된 기사 링크
                for element in article_elements[:8]:  # 최대 8개 기사
                    href = element.get_attribute('href')
                    if href and 'dezeen.com' in href and href not in article_links:
                        article_links.append(href)
            except Exception as e:
                logger.debug(f"기사 링크 수집 오류: {e}")
            
            # 각 기사에서 이미지 수집
            for article_url in article_links:
                if len(image_urls) >= max_images:
                    break
                
                try:
                    driver.get(article_url)
                    time.sleep(3)
                    
                    # 이미지 요소들 찾기
                    img_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='dezeen.com']")
                    
                    for img in img_elements:
                        if len(image_urls) >= max_images:
                            break
                        
                        img_src = img.get_attribute('src')
                        if img_src and self._is_valid_architecture_image(img_src):
                            image_urls.append(img_src)
                
                except Exception as e:
                    logger.debug(f"기사 페이지 처리 오류: {article_url} - {e}")
                    continue
            
            logger.info(f"Dezeen에서 {len(image_urls)}개 이미지 URL 수집")
            return image_urls
            
        except Exception as e:
            logger.error(f"Dezeen 수집 오류: {e}")
            return image_urls
        finally:
            if driver:
                driver.quit()
    
    def _is_valid_architecture_image(self, url: str) -> bool:
        """
        건축 이미지 URL 유효성 검사
        
        Args:
            url: 검사할 URL
            
        Returns:
            유효성 여부
        """
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        # 제외할 패턴들 (로고, 아이콘, 광고 등)
        exclude_patterns = [
            'logo', 'icon', 'avatar', 'profile', 'banner',
            'advertisement', 'ads', 'social', 'share',
            'thumbnail-small', 'thumb', 'preview-small'
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in exclude_patterns):
            return False
        
        # 이미지 파일 확장자 확인
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if not any(ext in url_lower for ext in valid_extensions):
            return False
        
        # 최소 크기 확인 (URL에 크기 정보가 있는 경우)
        size_indicators = ['1200', '1000', '800', 'large', 'medium']
        has_good_size = any(indicator in url_lower for indicator in size_indicators)
        
        return has_good_size or 'small' not in url_lower
    
    def collect_images_for_category(self, category: str, max_images_per_site: int = 15) -> List[str]:
        """
        카테고리별 이미지 수집
        
        Args:
            category: 카테고리명
            max_images_per_site: 사이트당 최대 이미지 수
            
        Returns:
            수집된 이미지 URL 리스트
        """
        if category not in self.target_sites:
            logger.error(f"지원하지 않는 카테고리: {category}")
            return []
        
        all_urls = []
        sites = self.target_sites[category]
        
        for site_url in sites:
            try:
                if 'archdaily.com' in site_url:
                    urls = self.collect_from_archdaily(site_url, max_images_per_site)
                elif 'dezeen.com' in site_url:
                    urls = self.collect_from_dezeen(site_url, max_images_per_site)
                else:
                    # 기본 수집 방법
                    urls = self._collect_generic(site_url, max_images_per_site)
                
                all_urls.extend(urls)
                
                # 사이트 간 대기
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"사이트 수집 실패: {site_url} - {e}")
                continue
        
        # 중복 제거
        unique_urls = list(set(all_urls))
        logger.info(f"카테고리 '{category}': {len(all_urls)}개 → 중복제거 후 {len(unique_urls)}개")
        
        return unique_urls
    
    def _collect_generic(self, url: str, max_images: int) -> List[str]:
        """일반적인 웹사이트에서 이미지 수집"""
        # 기본적인 이미지 수집 로직
        # 실제로는 각 사이트의 구조에 맞게 구현해야 함
        return []
    
    def collect_all_categories(self, max_images_per_category: int = 50) -> Dict[str, List[str]]:
        """
        모든 카테고리 이미지 수집
        
        Args:
            max_images_per_category: 카테고리당 최대 이미지 수
            
        Returns:
            카테고리별 이미지 URL 딕셔너리
        """
        results = {}
        
        for category in self.target_sites.keys():
            logger.info(f"\n🏗️ === {category.upper()} 카테고리 수집 시작 ===")
            
            try:
                # 사이트당 이미지 수 계산
                num_sites = len(self.target_sites[category])
                images_per_site = max(10, max_images_per_category // num_sites)
                
                urls = self.collect_images_for_category(category, images_per_site)
                
                # 최대 개수 제한
                if len(urls) > max_images_per_category:
                    urls = urls[:max_images_per_category]
                
                results[category] = urls
                logger.info(f"✅ {category} 완료: {len(urls)}개 URL")
                
                # 카테고리 간 대기
                time.sleep(15)
                
            except Exception as e:
                logger.error(f"❌ {category} 실패: {e}")
                results[category] = []
        
        return results


def test_architecture_collector():
    """건축 이미지 수집기 테스트"""
    print("🏗️ 건축 외장재 이미지 수집기 테스트")
    print("=" * 50)
    
    collector = ArchitectureImageCollector(headless=False)
    
    # 벽돌 카테고리 테스트
    test_category = 'brick'
    print(f"테스트 카테고리: {test_category}")
    
    urls = collector.collect_images_for_category(test_category, max_images_per_site=5)
    
    print(f"\n수집 결과: {len(urls)}개 URL")
    for i, url in enumerate(urls[:5], 1):
        print(f"  {i}. {url}")
    
    return len(urls) > 0


if __name__ == "__main__":
    success = test_architecture_collector()
    
    if success:
        print("\n✅ 건축 이미지 수집기 테스트 성공!")
    else:
        print("\n❌ 건축 이미지 수집기 테스트 실패!")
        print("💡 대안: 수동으로 건축 외장재 이미지를 수집하거나")
        print("   기존 건축 데이터셋을 활용하는 것을 권장합니다.")