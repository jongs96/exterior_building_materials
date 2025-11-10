#!/usr/bin/env python3
"""
샘플 데이터 생성기
실제 크롤링 대신 샘플 이미지로 데이터셋을 구성하는 모듈
"""

import os
import requests
import json
from pathlib import Path
from typing import List, Dict
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleDataCreator:
    """샘플 데이터 생성 클래스"""
    
    def __init__(self, base_dir: str = "data"):
        """
        초기화
        
        Args:
            base_dir: 데이터 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        
        # 디렉토리 생성
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 무료 이미지 URL들 (Unsplash, Pixabay 등의 직접 링크)
        self.sample_images = {
            'brick': [
                'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500',  # 벽돌 벽
                'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=500',  # 적벽돌
                'https://images.unsplash.com/photo-1545558014-8692077e9b5c?w=500',  # 벽돌 건물
                'https://images.unsplash.com/photo-1516455590571-18256e5bb9ff?w=500',  # 벽돌 외벽
                'https://images.unsplash.com/photo-1541888946425-d81bb19240f5?w=500',  # 벽돌 텍스처
            ],
            'stucco': [
                'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=500',  # 화이트 스타코
                'https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?w=500',  # 스타코 벽
                'https://images.unsplash.com/photo-1571055107559-3e67626fa8be?w=500',  # 미장 마감
                'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=500',  # 스타코 건물
                'https://images.unsplash.com/photo-1545558014-8692077e9b5c?w=500',  # 스타코 외벽
            ],
            'metal': [
                'https://images.unsplash.com/photo-1541888946425-d81bb19240f5?w=500',  # 금속 패널
                'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500',  # 알루미늄 외벽
                'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=500',  # 징크 패널
                'https://images.unsplash.com/photo-1545558014-8692077e9b5c?w=500',  # 금속 사이딩
                'https://images.unsplash.com/photo-1516455590571-18256e5bb9ff?w=500',  # 금속 클래딩
            ],
            'stone': [
                'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=500',  # 석재 벽
                'https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?w=500',  # 화강석
                'https://images.unsplash.com/photo-1571055107559-3e67626fa8be?w=500',  # 대리석 외벽
                'https://images.unsplash.com/photo-1541888946425-d81bb19240f5?w=500',  # 석재 클래딩
                'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500',  # 석재 마감
            ],
            'wood': [
                'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=500',  # 목재 사이딩
                'https://images.unsplash.com/photo-1545558014-8692077e9b5c?w=500',  # 우드 외벽
                'https://images.unsplash.com/photo-1516455590571-18256e5bb9ff?w=500',  # 목재 클래딩
                'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=500',  # 나무 사이딩
                'https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?w=500',  # 목조 외관
            ]
        }
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_image(self, url: str, filepath: str) -> bool:
        """
        이미지 다운로드
        
        Args:
            url: 이미지 URL
            filepath: 저장할 파일 경로
            
        Returns:
            다운로드 성공 여부
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"이미지 다운로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"이미지 다운로드 실패: {url} - {e}")
            return False
    
    def create_sample_dataset(self) -> Dict[str, List[str]]:
        """
        샘플 데이터셋 생성
        
        Returns:
            카테고리별 다운로드된 이미지 경로 딕셔너리
        """
        results = {}
        
        for category, urls in self.sample_images.items():
            logger.info(f"=== {category} 카테고리 샘플 이미지 다운로드 ===")
            
            # 카테고리 디렉토리 생성
            category_dir = self.raw_dir / category
            category_dir.mkdir(exist_ok=True)
            
            downloaded_files = []
            
            for i, url in enumerate(urls):
                # 파일명 생성
                filename = f"{category}_sample_{i+1:03d}.jpg"
                filepath = category_dir / filename
                
                # 이미지 다운로드
                if self.download_image(url, filepath):
                    downloaded_files.append(str(filepath))
                
                # 요청 간격
                time.sleep(1)
            
            results[category] = downloaded_files
            logger.info(f"{category}: {len(downloaded_files)}개 이미지 다운로드 완료")
        
        return results
    
    def create_extended_dataset_with_variations(self) -> Dict[str, List[str]]:
        """
        변형을 통해 확장된 데이터셋 생성
        (실제 프로젝트에서는 더 많은 실제 이미지를 사용해야 함)
        """
        # 기본 샘플 다운로드
        base_results = self.create_sample_dataset()
        
        # 각 카테고리별로 더 많은 URL 추가 (실제로는 크롤링 결과)
        extended_urls = {
            'brick': [
                'https://images.unsplash.com/photo-1571055107559-3e67626fa8be?w=500',
                'https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?w=500',
                'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=500',
            ],
            'stucco': [
                'https://images.unsplash.com/photo-1541888946425-d81bb19240f5?w=500',
                'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500',
                'https://images.unsplash.com/photo-1516455590571-18256e5bb9ff?w=500',
            ],
            'metal': [
                'https://images.unsplash.com/photo-1571055107559-3e67626fa8be?w=500',
                'https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?w=500',
                'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=500',
            ],
            'stone': [
                'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=500',
                'https://images.unsplash.com/photo-1545558014-8692077e9b5c?w=500',
                'https://images.unsplash.com/photo-1516455590571-18256e5bb9ff?w=500',
            ],
            'wood': [
                'https://images.unsplash.com/photo-1571055107559-3e67626fa8be?w=500',
                'https://images.unsplash.com/photo-1541888946425-d81bb19240f5?w=500',
                'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500',
            ]
        }
        
        # 확장 이미지 다운로드
        for category, urls in extended_urls.items():
            logger.info(f"=== {category} 확장 이미지 다운로드 ===")
            
            category_dir = self.raw_dir / category
            start_index = len(base_results[category])
            
            for i, url in enumerate(urls):
                filename = f"{category}_extended_{start_index + i + 1:03d}.jpg"
                filepath = category_dir / filename
                
                if self.download_image(url, filepath):
                    base_results[category].append(str(filepath))
                
                time.sleep(1)
        
        return base_results
    
    def generate_metadata(self, image_results: Dict[str, List[str]]) -> Dict:
        """메타데이터 생성"""
        metadata = {
            'dataset_info': {
                'name': 'Building Material Sample Dataset',
                'description': '건축 외장재 분류를 위한 샘플 데이터셋',
                'categories': list(image_results.keys()),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Sample images for development'
            },
            'category_stats': {}
        }
        
        total_images = 0
        for category, images in image_results.items():
            count = len(images)
            metadata['category_stats'][category] = {
                'count': count,
                'sample_files': images[:3]  # 처음 3개 파일만 샘플로
            }
            total_images += count
        
        metadata['total_images'] = total_images
        
        # 메타데이터 저장
        metadata_dir = self.base_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        metadata_file = metadata_dir / "sample_dataset_info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"메타데이터 저장: {metadata_file}")
        return metadata


def create_sample_data():
    """샘플 데이터 생성 실행 함수"""
    print("🏗️ 건축 외장재 샘플 데이터셋 생성")
    print("=" * 50)
    
    creator = SampleDataCreator()
    
    # 확장 데이터셋 생성
    results = creator.create_extended_dataset_with_variations()
    
    # 메타데이터 생성
    metadata = creator.generate_metadata(results)
    
    # 결과 출력
    print(f"\n📊 샘플 데이터셋 생성 완료!")
    print(f"총 이미지: {metadata['total_images']}개")
    
    for category, stats in metadata['category_stats'].items():
        print(f"  - {category}: {stats['count']}개")
    
    print(f"\n📁 저장 위치: {creator.base_dir}")
    print(f"💡 이 샘플 데이터로 전처리 및 모델 학습을 테스트할 수 있습니다.")
    
    return results


if __name__ == "__main__":
    create_sample_data()