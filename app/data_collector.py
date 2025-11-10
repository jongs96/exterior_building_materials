"""
건축 외장 마감재 분류를 위한 데이터 수집 모듈
웹 크롤링, 이미지 다운로드, 검증 및 데이터셋 구조화 기능 제공
"""

import os
import time
import hashlib
import json
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

from PIL import Image
import cv2
import numpy as np


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebCrawler:
    """웹 크롤링을 통한 이미지 URL 수집 클래스"""
    
    def __init__(self, headless: bool = True, timeout: int = 10):
        """
        웹 크롤러 초기화
        
        Args:
            headless: 브라우저 헤드리스 모드 여부
            timeout: 페이지 로딩 타임아웃 (초)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Selenium 드라이버 설정
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        
        # 5가지 외장재 카테고리별 크롤링 타겟 사이트 설정
        self.target_sites = {
            'brick': [
                'https://unsplash.com/s/photos/brick-wall',
                'https://pixabay.com/images/search/brick/',
                'https://www.pexels.com/search/brick%20wall/'
            ],
            'stucco': [
                'https://unsplash.com/s/photos/stucco-wall',
                'https://pixabay.com/images/search/stucco/',
                'https://www.pexels.com/search/stucco%20wall/'
            ],
            'metal': [
                'https://unsplash.com/s/photos/metal-siding',
                'https://pixabay.com/images/search/metal-panel/',
                'https://www.pexels.com/search/metal%20siding/'
            ],
            'stone': [
                'https://unsplash.com/s/photos/stone-wall',
                'https://pixabay.com/images/search/stone-wall/',
                'https://www.pexels.com/search/stone%20wall/'
            ],
            'wood': [
                'https://unsplash.com/s/photos/wood-siding',
                'https://pixabay.com/images/search/wood-siding/',
                'https://www.pexels.com/search/wood%20siding/'
            ]
        }
    
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
    
    def _make_request_with_retry(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """HTTP 요청 에러 처리 및 재시도 로직"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"요청 실패 (시도 {attempt + 1}/{max_retries}): {url} - {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 지수 백오프
                else:
                    logger.error(f"최대 재시도 횟수 초과: {url}")
                    return None
    
    def collect_image_urls_beautifulsoup(self, url: str, max_images: int = 50) -> List[str]:
        """BeautifulSoup을 사용한 이미지 URL 수집"""
        image_urls = []
        
        response = self._make_request_with_retry(url)
        if not response:
            return image_urls
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 다양한 이미지 태그 패턴 검색
            img_tags = soup.find_all('img')
            
            for img in img_tags:
                # src, data-src, data-lazy-src 등 다양한 속성 확인
                img_url = None
                for attr in ['src', 'data-src', 'data-lazy-src', 'data-original']:
                    if img.get(attr):
                        img_url = img.get(attr)
                        break
                
                if img_url:
                    # 상대 URL을 절대 URL로 변환
                    img_url = urljoin(url, img_url)
                    
                    # 이미지 URL 유효성 검사
                    if self._is_valid_image_url(img_url):
                        image_urls.append(img_url)
                        
                        if len(image_urls) >= max_images:
                            break
            
            logger.info(f"BeautifulSoup으로 {len(image_urls)}개 이미지 URL 수집: {url}")
            return image_urls
            
        except Exception as e:
            logger.error(f"BeautifulSoup 파싱 오류: {url} - {e}")
            return image_urls
    
    def collect_image_urls_selenium(self, url: str, max_images: int = 50) -> List[str]:
        """Selenium을 사용한 동적 콘텐츠 이미지 URL 수집"""
        image_urls = []
        driver = None
        
        try:
            driver = self._get_driver()
            driver.get(url)
            
            # 페이지 로딩 대기
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "img"))
            )
            
            # 스크롤하여 더 많은 이미지 로드
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 5
            
            while scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1
            
            # 이미지 요소 찾기
            img_elements = driver.find_elements(By.TAG_NAME, "img")
            
            for img in img_elements:
                try:
                    # 다양한 속성에서 이미지 URL 추출
                    img_url = None
                    for attr in ['src', 'data-src', 'data-lazy-src', 'data-original']:
                        img_url = img.get_attribute(attr)
                        if img_url:
                            break
                    
                    if img_url and self._is_valid_image_url(img_url):
                        image_urls.append(img_url)
                        
                        if len(image_urls) >= max_images:
                            break
                            
                except Exception as e:
                    logger.debug(f"이미지 요소 처리 오류: {e}")
                    continue
            
            logger.info(f"Selenium으로 {len(image_urls)}개 이미지 URL 수집: {url}")
            return image_urls
            
        except TimeoutException:
            logger.error(f"페이지 로딩 타임아웃: {url}")
            return image_urls
        except WebDriverException as e:
            logger.error(f"Selenium 오류: {url} - {e}")
            return image_urls
        except Exception as e:
            logger.error(f"예상치 못한 오류: {url} - {e}")
            return image_urls
        finally:
            if driver:
                driver.quit()
    
    def _is_valid_image_url(self, url: str) -> bool:
        """이미지 URL 유효성 검사"""
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        # 이미지 파일 확장자 확인
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
        
        # URL에 이미지 확장자가 있거나, 이미지 관련 키워드가 있는 경우
        has_extension = any(path.endswith(ext) for ext in valid_extensions)
        has_image_keyword = any(keyword in url.lower() for keyword in ['image', 'img', 'photo', 'picture'])
        
        return has_extension or has_image_keyword
    
    def collect_images_for_category(self, category: str, max_images_per_site: int = 30) -> List[str]:
        """특정 카테고리에 대한 이미지 URL 수집"""
        if category not in self.target_sites:
            logger.error(f"지원하지 않는 카테고리: {category}")
            return []
        
        all_image_urls = []
        sites = self.target_sites[category]
        
        for site_url in sites:
            logger.info(f"{category} 카테고리 크롤링 시작: {site_url}")
            
            # BeautifulSoup 먼저 시도
            urls_bs = self.collect_image_urls_beautifulsoup(site_url, max_images_per_site)
            all_image_urls.extend(urls_bs)
            
            # 충분한 이미지를 얻지 못한 경우 Selenium 사용
            if len(urls_bs) < max_images_per_site // 2:
                logger.info(f"Selenium으로 추가 수집 시도: {site_url}")
                urls_selenium = self.collect_image_urls_selenium(site_url, max_images_per_site)
                all_image_urls.extend(urls_selenium)
            
            # 사이트 간 요청 간격
            time.sleep(1)
        
        # 중복 제거
        unique_urls = list(set(all_image_urls))
        logger.info(f"{category} 카테고리 총 {len(unique_urls)}개 고유 이미지 URL 수집")
        
        return unique_urls
    
    def collect_all_categories(self, max_images_per_category: int = 150) -> Dict[str, List[str]]:
        """모든 카테고리에 대한 이미지 URL 수집"""
        results = {}
        
        for category in self.target_sites.keys():
            logger.info(f"=== {category} 카테고리 수집 시작 ===")
            urls = self.collect_images_for_category(category, max_images_per_category // len(self.target_sites[category]))
            results[category] = urls[:max_images_per_category]  # 최대 개수 제한
            logger.info(f"{category} 카테고리 완료: {len(results[category])}개 URL")
        
        return results


class ImageDownloader:
    """이미지 다운로드 및 검증 시스템"""
    
    def __init__(self, download_dir: str = "data/raw", min_resolution: Tuple[int, int] = (224, 224), 
                 max_file_size: int = 10 * 1024 * 1024):  # 10MB
        """
        이미지 다운로더 초기화
        
        Args:
            download_dir: 다운로드 디렉토리
            min_resolution: 최소 해상도 (width, height)
            max_file_size: 최대 파일 크기 (bytes)
        """
        self.download_dir = Path(download_dir)
        self.min_resolution = min_resolution
        self.max_file_size = max_file_size
        
        # 다운로드 디렉토리 생성
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_image(self, url: str, filename: str, max_retries: int = 3) -> Optional[str]:
        """
        이미지 파일 다운로드 및 로컬 저장
        
        Args:
            url: 이미지 URL
            filename: 저장할 파일명
            max_retries: 최대 재시도 횟수
            
        Returns:
            저장된 파일 경로 또는 None (실패 시)
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Content-Length 확인 (있는 경우)
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > self.max_file_size:
                    logger.warning(f"파일 크기 초과: {url} ({content_length} bytes)")
                    return None
                
                # 파일 저장
                file_path = self.download_dir / filename
                
                with open(file_path, 'wb') as f:
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            downloaded_size += len(chunk)
                            if downloaded_size > self.max_file_size:
                                logger.warning(f"다운로드 중 파일 크기 초과: {url}")
                                f.close()
                                file_path.unlink()  # 파일 삭제
                                return None
                            f.write(chunk)
                
                logger.debug(f"이미지 다운로드 완료: {filename}")
                return str(file_path)
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"다운로드 실패 (시도 {attempt + 1}/{max_retries}): {url} - {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"최대 재시도 횟수 초과: {url}")
                    return None
            except Exception as e:
                logger.error(f"예상치 못한 다운로드 오류: {url} - {e}")
                return None
    
    def validate_image(self, image_path: str) -> bool:
        """
        이미지 품질 검증 (해상도, 파일 크기, 손상 여부)
        
        Args:
            image_path: 검증할 이미지 파일 경로
            
        Returns:
            검증 통과 여부
        """
        try:
            file_path = Path(image_path)
            
            # 파일 존재 확인
            if not file_path.exists():
                logger.warning(f"파일이 존재하지 않음: {image_path}")
                return False
            
            # 파일 크기 확인
            file_size = file_path.stat().st_size
            if file_size == 0:
                logger.warning(f"빈 파일: {image_path}")
                return False
            
            if file_size > self.max_file_size:
                logger.warning(f"파일 크기 초과: {image_path} ({file_size} bytes)")
                return False
            
            # PIL로 이미지 열기 시도 (손상 여부 확인)
            try:
                with Image.open(image_path) as img:
                    # 이미지 로드 강제 실행
                    img.load()
                    
                    # 해상도 확인
                    width, height = img.size
                    if width < self.min_resolution[0] or height < self.min_resolution[1]:
                        logger.warning(f"해상도 부족: {image_path} ({width}x{height})")
                        return False
                    
                    # 이미지 모드 확인 (RGB, RGBA, L 등)
                    if img.mode not in ['RGB', 'RGBA', 'L']:
                        logger.warning(f"지원하지 않는 이미지 모드: {image_path} ({img.mode})")
                        return False
                    
                    logger.debug(f"이미지 검증 통과: {image_path} ({width}x{height}, {img.mode})")
                    return True
                    
            except Exception as e:
                logger.warning(f"이미지 손상 또는 형식 오류: {image_path} - {e}")
                return False
                
        except Exception as e:
            logger.error(f"이미지 검증 중 오류: {image_path} - {e}")
            return False
    
    def calculate_image_hash(self, image_path: str) -> Optional[str]:
        """
        이미지 해시 계산 (중복 제거용)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            이미지 해시값 또는 None (실패 시)
        """
        try:
            with Image.open(image_path) as img:
                # 이미지를 작은 크기로 리사이즈하여 해시 계산
                img_resized = img.resize((8, 8), Image.Resampling.LANCZOS)
                img_gray = img_resized.convert('L')
                
                # 픽셀 값을 바이트로 변환
                pixels = list(img_gray.getdata())
                pixel_bytes = bytes(pixels)
                
                # SHA256 해시 계산
                hash_value = hashlib.sha256(pixel_bytes).hexdigest()
                return hash_value
                
        except Exception as e:
            logger.error(f"해시 계산 오류: {image_path} - {e}")
            return None
    
    def remove_duplicates(self, image_paths: List[str]) -> List[str]:
        """
        중복 이미지 제거 알고리즘 (해시 기반)
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            
        Returns:
            중복이 제거된 이미지 파일 경로 리스트
        """
        unique_images = []
        seen_hashes = set()
        
        for image_path in image_paths:
            if not Path(image_path).exists():
                continue
                
            # 이미지 해시 계산
            image_hash = self.calculate_image_hash(image_path)
            if not image_hash:
                continue
            
            # 중복 확인
            if image_hash not in seen_hashes:
                seen_hashes.add(image_hash)
                unique_images.append(image_path)
            else:
                # 중복 이미지 삭제
                try:
                    Path(image_path).unlink()
                    logger.debug(f"중복 이미지 삭제: {image_path}")
                except Exception as e:
                    logger.warning(f"중복 이미지 삭제 실패: {image_path} - {e}")
        
        logger.info(f"중복 제거 완료: {len(image_paths)} -> {len(unique_images)}개")
        return unique_images
    
    def download_and_validate_images(self, urls: List[str], category: str) -> List[str]:
        """
        이미지 URL 리스트를 다운로드하고 검증
        
        Args:
            urls: 이미지 URL 리스트
            category: 카테고리명
            
        Returns:
            검증을 통과한 이미지 파일 경로 리스트
        """
        # 카테고리별 디렉토리 생성
        category_dir = self.download_dir / category
        category_dir.mkdir(exist_ok=True)
        
        downloaded_images = []
        
        for i, url in enumerate(urls):
            try:
                # 파일명 생성
                parsed_url = urlparse(url)
                file_extension = Path(parsed_url.path).suffix
                if not file_extension or file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    file_extension = '.jpg'
                
                filename = f"{category}_{i:04d}{file_extension}"
                
                # 이미지 다운로드
                file_path = self.download_image(url, category_dir / filename)
                if not file_path:
                    continue
                
                # 이미지 검증
                if self.validate_image(file_path):
                    downloaded_images.append(file_path)
                else:
                    # 검증 실패 시 파일 삭제
                    try:
                        Path(file_path).unlink()
                    except Exception:
                        pass
                
            except Exception as e:
                logger.error(f"이미지 처리 오류: {url} - {e}")
                continue
        
        # 중복 이미지 제거
        unique_images = self.remove_duplicates(downloaded_images)
        
        logger.info(f"{category} 카테고리 다운로드 완료: {len(unique_images)}개 이미지")
        return unique_images


class DatasetOrganizer:
    """데이터셋 구조화 및 메타데이터 생성 클래스"""
    
    def __init__(self, base_dir: str = "data"):
        """
        데이터셋 구성자 초기화
        
        Args:
            base_dir: 데이터셋 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"
        
        # 디렉토리 생성
        for dir_path in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_dataset_structure(self, categories: List[str]) -> None:
        """
        train/validation/test 폴더 자동 생성
        
        Args:
            categories: 카테고리 리스트
        """
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            split_dir = self.processed_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for category in categories:
                category_dir = split_dir / category
                category_dir.mkdir(exist_ok=True)
                
        logger.info(f"데이터셋 구조 생성 완료: {splits} x {categories}")
    
    def distribute_images(self, category_images: Dict[str, List[str]], 
                         train_ratio: float = 0.8, val_ratio: float = 0.1, 
                         test_ratio: float = 0.1) -> Dict[str, Dict[str, int]]:
        """
        이미지 분배 (8:1:1 비율)
        
        Args:
            category_images: 카테고리별 이미지 파일 경로 딕셔너리
            train_ratio: 훈련 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            
        Returns:
            분배 결과 통계
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("비율의 합이 1.0이 되어야 합니다.")
        
        distribution_stats = {}
        
        for category, image_paths in category_images.items():
            if not image_paths:
                logger.warning(f"{category} 카테고리에 이미지가 없습니다.")
                continue
            
            # 이미지 리스트 셔플
            import random
            random.shuffle(image_paths)
            
            total_images = len(image_paths)
            train_count = int(total_images * train_ratio)
            val_count = int(total_images * val_ratio)
            test_count = total_images - train_count - val_count
            
            # 분할 인덱스 계산
            train_end = train_count
            val_end = train_end + val_count
            
            # 이미지 분배
            splits = {
                'train': image_paths[:train_end],
                'validation': image_paths[train_end:val_end],
                'test': image_paths[val_end:]
            }
            
            # 파일 복사
            for split, images in splits.items():
                target_dir = self.processed_dir / split / category
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for i, src_path in enumerate(images):
                    src_file = Path(src_path)
                    if not src_file.exists():
                        continue
                    
                    # 새 파일명 생성
                    dst_filename = f"{category}_{split}_{i:04d}{src_file.suffix}"
                    dst_path = target_dir / dst_filename
                    
                    try:
                        # 파일 복사
                        import shutil
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        logger.error(f"파일 복사 실패: {src_path} -> {dst_path} - {e}")
            
            # 통계 저장
            distribution_stats[category] = {
                'total': total_images,
                'train': train_count,
                'validation': val_count,
                'test': test_count
            }
            
            logger.info(f"{category} 분배 완료 - Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        return distribution_stats
    
    def generate_dataset_metadata(self, distribution_stats: Dict[str, Dict[str, int]]) -> Dict:
        """
        각 카테고리별 이미지 개수 및 통계 정보 JSON 파일 생성
        
        Args:
            distribution_stats: 분배 통계 정보
            
        Returns:
            메타데이터 딕셔너리
        """
        metadata = {
            'dataset_info': {
                'name': 'Building Material Classification Dataset',
                'description': '건축 외장 마감재 분류를 위한 이미지 데이터셋',
                'categories': list(distribution_stats.keys()),
                'num_categories': len(distribution_stats),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'splits': ['train', 'validation', 'test'],
                'split_ratios': {
                    'train': 0.8,
                    'validation': 0.1,
                    'test': 0.1
                }
            },
            'category_stats': distribution_stats,
            'total_stats': {
                'total_images': sum(stats['total'] for stats in distribution_stats.values()),
                'train_images': sum(stats['train'] for stats in distribution_stats.values()),
                'validation_images': sum(stats['validation'] for stats in distribution_stats.values()),
                'test_images': sum(stats['test'] for stats in distribution_stats.values())
            }
        }
        
        # 메타데이터 파일 저장
        metadata_file = self.metadata_dir / 'dataset_info.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"데이터셋 메타데이터 저장: {metadata_file}")
        return metadata
    
    def generate_quality_report(self, category_images: Dict[str, List[str]]) -> Dict:
        """
        데이터셋 품질 리포트 생성
        
        Args:
            category_images: 카테고리별 이미지 파일 경로 딕셔너리
            
        Returns:
            품질 리포트 딕셔너리
        """
        quality_report = {
            'report_info': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_categories': len(category_images)
            },
            'category_quality': {},
            'overall_quality': {
                'total_images': 0,
                'avg_resolution': {'width': 0, 'height': 0},
                'avg_file_size': 0,
                'format_distribution': {}
            }
        }
        
        all_resolutions = []
        all_file_sizes = []
        format_counts = {}
        total_images = 0
        
        for category, image_paths in category_images.items():
            category_quality = {
                'image_count': len(image_paths),
                'resolutions': [],
                'file_sizes': [],
                'formats': {},
                'avg_resolution': {'width': 0, 'height': 0},
                'avg_file_size': 0
            }
            
            for image_path in image_paths:
                try:
                    file_path = Path(image_path)
                    if not file_path.exists():
                        continue
                    
                    # 파일 크기
                    file_size = file_path.stat().st_size
                    category_quality['file_sizes'].append(file_size)
                    all_file_sizes.append(file_size)
                    
                    # 이미지 정보
                    with Image.open(image_path) as img:
                        width, height = img.size
                        category_quality['resolutions'].append({'width': width, 'height': height})
                        all_resolutions.append({'width': width, 'height': height})
                        
                        # 파일 형식
                        format_name = img.format or 'Unknown'
                        category_quality['formats'][format_name] = category_quality['formats'].get(format_name, 0) + 1
                        format_counts[format_name] = format_counts.get(format_name, 0) + 1
                        
                        total_images += 1
                
                except Exception as e:
                    logger.warning(f"품질 분석 오류: {image_path} - {e}")
                    continue
            
            # 카테고리별 평균 계산
            if category_quality['resolutions']:
                avg_width = sum(r['width'] for r in category_quality['resolutions']) / len(category_quality['resolutions'])
                avg_height = sum(r['height'] for r in category_quality['resolutions']) / len(category_quality['resolutions'])
                category_quality['avg_resolution'] = {'width': int(avg_width), 'height': int(avg_height)}
            
            if category_quality['file_sizes']:
                category_quality['avg_file_size'] = int(sum(category_quality['file_sizes']) / len(category_quality['file_sizes']))
            
            # 리스트 데이터는 요약 통계로 변경 (JSON 크기 최적화)
            category_quality['resolution_stats'] = {
                'min_width': min(r['width'] for r in category_quality['resolutions']) if category_quality['resolutions'] else 0,
                'max_width': max(r['width'] for r in category_quality['resolutions']) if category_quality['resolutions'] else 0,
                'min_height': min(r['height'] for r in category_quality['resolutions']) if category_quality['resolutions'] else 0,
                'max_height': max(r['height'] for r in category_quality['resolutions']) if category_quality['resolutions'] else 0,
            }
            
            category_quality['file_size_stats'] = {
                'min_size': min(category_quality['file_sizes']) if category_quality['file_sizes'] else 0,
                'max_size': max(category_quality['file_sizes']) if category_quality['file_sizes'] else 0,
            }
            
            # 상세 데이터 제거 (메모리 절약)
            del category_quality['resolutions']
            del category_quality['file_sizes']
            
            quality_report['category_quality'][category] = category_quality
        
        # 전체 평균 계산
        if all_resolutions:
            avg_width = sum(r['width'] for r in all_resolutions) / len(all_resolutions)
            avg_height = sum(r['height'] for r in all_resolutions) / len(all_resolutions)
            quality_report['overall_quality']['avg_resolution'] = {'width': int(avg_width), 'height': int(avg_height)}
        
        if all_file_sizes:
            quality_report['overall_quality']['avg_file_size'] = int(sum(all_file_sizes) / len(all_file_sizes))
        
        quality_report['overall_quality']['total_images'] = total_images
        quality_report['overall_quality']['format_distribution'] = format_counts
        
        # 품질 리포트 파일 저장
        report_file = self.metadata_dir / 'quality_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"품질 리포트 저장: {report_file}")
        return quality_report


class DataCollectionPipeline:
    """전체 데이터 수집 파이프라인을 관리하는 클래스"""
    
    def __init__(self, base_dir: str = "data"):
        """
        데이터 수집 파이프라인 초기화
        
        Args:
            base_dir: 데이터 기본 디렉토리
        """
        self.crawler = WebCrawler()
        self.downloader = ImageDownloader(download_dir=f"{base_dir}/raw")
        self.organizer = DatasetOrganizer(base_dir=base_dir)
        
        # 카테고리 정의
        self.categories = ['brick', 'stucco', 'metal', 'stone', 'wood']
    
    def run_full_pipeline(self, max_images_per_category: int = 150) -> Dict:
        """
        전체 데이터 수집 파이프라인 실행
        
        Args:
            max_images_per_category: 카테고리당 최대 이미지 수
            
        Returns:
            파이프라인 실행 결과
        """
        logger.info("=== 데이터 수집 파이프라인 시작 ===")
        
        # 1. 이미지 URL 수집
        logger.info("1단계: 이미지 URL 수집")
        url_results = self.crawler.collect_all_categories(max_images_per_category)
        
        # 2. 이미지 다운로드 및 검증
        logger.info("2단계: 이미지 다운로드 및 검증")
        category_images = {}
        for category, urls in url_results.items():
            images = self.downloader.download_and_validate_images(urls, category)
            category_images[category] = images
        
        # 3. 데이터셋 구조 생성
        logger.info("3단계: 데이터셋 구조 생성")
        self.organizer.create_dataset_structure(self.categories)
        
        # 4. 이미지 분배
        logger.info("4단계: 이미지 분배 (8:1:1)")
        distribution_stats = self.organizer.distribute_images(category_images)
        
        # 5. 메타데이터 생성
        logger.info("5단계: 메타데이터 생성")
        metadata = self.organizer.generate_dataset_metadata(distribution_stats)
        
        # 6. 품질 리포트 생성
        logger.info("6단계: 품질 리포트 생성")
        quality_report = self.organizer.generate_quality_report(category_images)
        
        logger.info("=== 데이터 수집 파이프라인 완료 ===")
        
        return {
            'url_collection': url_results,
            'downloaded_images': category_images,
            'distribution_stats': distribution_stats,
            'metadata': metadata,
            'quality_report': quality_report
        }


if __name__ == "__main__":
    # 사용 예시
    pipeline = DataCollectionPipeline()
    results = pipeline.run_full_pipeline(max_images_per_category=120)
    
    print("데이터 수집 완료!")
    print(f"총 카테고리: {len(results['downloaded_images'])}")
    for category, images in results['downloaded_images'].items():
        print(f"- {category}: {len(images)}개 이미지")