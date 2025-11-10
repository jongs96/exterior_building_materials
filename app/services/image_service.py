"""
이미지 서비스
이미지 전처리 및 변환을 담당하는 서비스 클래스
"""

import io
import numpy as np
from PIL import Image, ImageOps
import cv2
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageService:
    """
    이미지 처리 서비스 클래스
    업로드된 이미지의 전처리를 담당합니다.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        이미지 서비스 초기화
        
        Args:
            target_size: 목표 이미지 크기 (width, height)
        """
        self.target_size = target_size
        
        # ImageNet 정규화 파라미터 (전이학습 모델용)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess_uploaded_image(self, file) -> Optional[np.ndarray]:
        """
        업로드된 파일을 모델 입력용으로 전처리
        
        Args:
            file: Flask 업로드 파일 객체
            
        Returns:
            전처리된 이미지 배열 (1, 224, 224, 3) 또는 None (실패 시)
        """
        try:
            # 파일을 PIL Image로 로드
            image = Image.open(file.stream)
            
            # 전처리 실행
            processed_image = self.preprocess_pil_image(image)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"업로드 이미지 전처리 실패: {e}")
            return None
    
    def preprocess_pil_image(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        """
        PIL Image를 모델 입력용으로 전처리
        
        Args:
            pil_image: PIL Image 객체
            
        Returns:
            전처리된 이미지 배열 (1, 224, 224, 3) 또는 None (실패 시)
        """
        try:
            # 1. 이미지 형식 확인 및 변환
            if pil_image.mode != 'RGB':
                if pil_image.mode == 'RGBA':
                    # RGBA를 RGB로 변환 (투명 배경을 흰색으로)
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[-1])  # 알파 채널을 마스크로 사용
                    pil_image = background
                else:
                    # 다른 모드를 RGB로 변환
                    pil_image = pil_image.convert('RGB')
            
            # 2. 이미지 크기 조정 (비율 유지하면서 리사이즈)
            resized_image = self._resize_with_padding(pil_image, self.target_size)
            
            # 3. NumPy 배열로 변환
            image_array = np.array(resized_image, dtype=np.float32)
            
            # 4. 정규화 (0-255 → 0-1)
            image_array = image_array / 255.0
            
            # 5. ImageNet 정규화 적용
            image_array = (image_array - self.mean) / self.std
            
            # 6. 배치 차원 추가 (1, 224, 224, 3)
            image_array = np.expand_dims(image_array, axis=0)
            
            logger.debug(f"이미지 전처리 완료: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"PIL 이미지 전처리 실패: {e}")
            return None
    
    def _resize_with_padding(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        비율을 유지하면서 이미지 크기 조정 (패딩 추가)
        
        Args:
            image: 원본 PIL Image
            target_size: 목표 크기 (width, height)
            
        Returns:
            리사이즈된 PIL Image
        """
        # 원본 이미지 크기
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # 비율 계산
        ratio = min(target_width / original_width, target_height / original_height)
        
        # 새로운 크기 계산
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # 이미지 리사이즈
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 패딩을 위한 새 이미지 생성 (흰색 배경)
        padded_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # 중앙에 리사이즈된 이미지 배치
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        return padded_image