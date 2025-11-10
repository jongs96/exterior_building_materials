"""
이미지 전처리 모듈

건축 외장재 이미지를 딥러닝 모델 학습 및 추론에 적합한 형태로 변환하는 모듈입니다.
주요 기능:
- 이미지 크기 조정 (224x224)
- 정규화 (0-1 범위)
- 텐서 변환
- 배치 처리
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """이미지 전처리를 담당하는 클래스"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        초기화 함수
        
        Args:
            target_size: 목표 이미지 크기 (높이, 너비)
        """
        self.target_size = target_size
        self.logger = logger
        
        # 이미지 정규화를 위한 평균과 표준편차 (ImageNet 기준)
        # 전이학습을 사용할 때 사전 훈련된 모델과 동일한 정규화 적용
        self.mean = np.array([0.485, 0.456, 0.406])  # RGB 채널별 평균
        self.std = np.array([0.229, 0.224, 0.225])   # RGB 채널별 표준편차
        
    def resize_image(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        이미지 크기를 조정하는 함수
        
        Args:
            image: 입력 이미지 (numpy 배열)
            size: 목표 크기 (높이, 너비). None이면 기본 크기 사용
            
        Returns:
            크기가 조정된 이미지
            
        초보자 주의사항:
        - cv2.resize는 (너비, 높이) 순서로 받지만, 우리는 (높이, 너비)로 통일
        - INTER_AREA는 축소 시 품질이 좋고, INTER_CUBIC은 확대 시 품질이 좋음
        """
        if size is None:
            size = self.target_size
            
        try:
            # OpenCV는 (너비, 높이) 순서이므로 순서를 바꿔줍니다
            resized = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            
            self.logger.debug(f"이미지 크기 조정: {image.shape} -> {resized.shape}")
            return resized
            
        except Exception as e:
            self.logger.error(f"이미지 크기 조정 실패: {e}")
            raise ValueError(f"이미지 크기 조정 중 오류 발생: {e}")
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지를 정규화하는 함수
        
        Args:
            image: 입력 이미지 (0-255 범위의 uint8)
            
        Returns:
            정규화된 이미지 (0-1 범위의 float32)
            
        왜 정규화가 필요한가?
        - 픽셀 값이 0-255 범위에 있으면 그래디언트가 너무 커져서 학습이 불안정해집니다
        - 0-1 범위로 조정하면 학습이 더 안정적이고 빨라집니다
        """
        try:
            # uint8 (0-255) -> float32 (0-1) 변환
            normalized = image.astype(np.float32) / 255.0
            
            # ImageNet 기준 정규화 적용 (전이학습 시 필수)
            # 각 채널별로 (픽셀값 - 평균) / 표준편차
            normalized = (normalized - self.mean) / self.std
            
            self.logger.debug(f"이미지 정규화 완료: {image.dtype} -> {normalized.dtype}")
            return normalized
            
        except Exception as e:
            self.logger.error(f"이미지 정규화 실패: {e}")
            raise ValueError(f"이미지 정규화 중 오류 발생: {e}")
    
    def preprocess_single_image(self, image_path: str) -> np.ndarray:
        """
        단일 이미지를 전처리하는 함수
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            전처리된 이미지 배열
            
        처리 순서:
        1. 이미지 로드
        2. RGB 변환 (OpenCV는 BGR로 로드하므로)
        3. 크기 조정
        4. 정규화
        """
        try:
            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
            # OpenCV로 이미지 로드 (BGR 형식)
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # BGR -> RGB 변환 (딥러닝 모델은 RGB를 기대)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 크기 조정
            image = self.resize_image(image)
            
            # 정규화
            image = self.normalize_image(image)
            
            self.logger.info(f"이미지 전처리 완료: {image_path}")
            return image
            
        except Exception as e:
            self.logger.error(f"이미지 전처리 실패 ({image_path}): {e}")
            raise
    
    def preprocess_batch_images(self, image_paths: List[str]) -> np.ndarray:
        """
        여러 이미지를 배치로 전처리하는 함수
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            
        Returns:
            전처리된 이미지 배치 (배치_크기, 높이, 너비, 채널)
            
        배치 처리의 장점:
        - GPU 메모리를 효율적으로 사용
        - 병렬 처리로 속도 향상
        - 메모리 사용량 예측 가능
        """
        processed_images = []
        failed_images = []
        
        for image_path in image_paths:
            try:
                processed_image = self.preprocess_single_image(image_path)
                processed_images.append(processed_image)
                
            except Exception as e:
                self.logger.warning(f"이미지 처리 실패, 건너뜀: {image_path} - {e}")
                failed_images.append(image_path)
                continue
        
        if not processed_images:
            raise ValueError("처리 가능한 이미지가 없습니다")
        
        # 리스트를 numpy 배열로 변환 (배치 차원 추가)
        batch_images = np.array(processed_images)
        
        self.logger.info(f"배치 처리 완료: {len(processed_images)}개 성공, {len(failed_images)}개 실패")
        if failed_images:
            self.logger.warning(f"실패한 이미지들: {failed_images}")
        
        return batch_images
    
    def create_data_generator(self, directory: str, batch_size: int = 32, 
                            class_mode: str = 'categorical', shuffle: bool = True) -> tf.keras.preprocessing.image.DirectoryIterator:
        """
        데이터 제너레이터를 생성하는 함수
        
        Args:
            directory: 이미지가 저장된 디렉토리 경로
            batch_size: 배치 크기
            class_mode: 라벨 형식 ('categorical', 'binary', 'sparse' 등)
            shuffle: 데이터 섞기 여부
            
        Returns:
            TensorFlow 데이터 제너레이터
            
        데이터 제너레이터란?
        - 메모리에 모든 이미지를 로드하지 않고 필요할 때마다 배치 단위로 로드
        - 대용량 데이터셋 처리 시 메모리 효율적
        - 실시간 데이터 증강 가능
        """
        try:
            # ImageDataGenerator 설정
            # rescale=1./255: 0-1 정규화 (우리는 별도로 정규화하므로 주석 처리)
            datagen = ImageDataGenerator(
                # rescale=1./255,  # 별도 정규화 함수 사용
                preprocessing_function=self._preprocess_function  # 커스텀 전처리 함수
            )
            
            # 디렉토리에서 데이터 제너레이터 생성
            generator = datagen.flow_from_directory(
                directory=directory,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                shuffle=shuffle,
                color_mode='rgb'  # RGB 컬러 모드
            )
            
            self.logger.info(f"데이터 제너레이터 생성 완료: {directory}")
            self.logger.info(f"클래스 수: {generator.num_classes}, 샘플 수: {generator.samples}")
            self.logger.info(f"클래스 인덱스: {generator.class_indices}")
            
            return generator
            
        except Exception as e:
            self.logger.error(f"데이터 제너레이터 생성 실패: {e}")
            raise
    
    def _preprocess_function(self, image: np.ndarray) -> np.ndarray:
        """
        ImageDataGenerator에서 사용할 전처리 함수
        
        Args:
            image: 입력 이미지 (0-255 범위)
            
        Returns:
            전처리된 이미지
        """
        # 0-255 -> 0-1 정규화
        image = image.astype(np.float32) / 255.0
        
        # ImageNet 정규화
        image = (image - self.mean) / self.std
        
        return image
    
    def get_image_statistics(self, image_paths: List[str]) -> dict:
        """
        이미지 데이터셋의 통계 정보를 계산하는 함수
        
        Args:
            image_paths: 분석할 이미지 경로 리스트
            
        Returns:
            통계 정보 딕셔너리
            
        통계 정보 활용:
        - 데이터셋 품질 확인
        - 전처리 파라미터 조정
        - 모델 성능 분석
        """
        stats = {
            'total_images': len(image_paths),
            'valid_images': 0,
            'invalid_images': 0,
            'image_sizes': [],
            'mean_pixel_values': [],
            'std_pixel_values': []
        }
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    stats['invalid_images'] += 1
                    continue
                
                # RGB 변환
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                stats['valid_images'] += 1
                stats['image_sizes'].append(image.shape[:2])  # (높이, 너비)
                
                # 픽셀 값 통계 (채널별)
                mean_vals = np.mean(image, axis=(0, 1))  # 각 채널의 평균
                std_vals = np.std(image, axis=(0, 1))    # 각 채널의 표준편차
                
                stats['mean_pixel_values'].append(mean_vals)
                stats['std_pixel_values'].append(std_vals)
                
            except Exception as e:
                self.logger.warning(f"통계 계산 실패: {image_path} - {e}")
                stats['invalid_images'] += 1
        
        # 전체 통계 계산
        if stats['mean_pixel_values']:
            stats['overall_mean'] = np.mean(stats['mean_pixel_values'], axis=0)
            stats['overall_std'] = np.mean(stats['std_pixel_values'], axis=0)
        
        # 이미지 크기 분포
        if stats['image_sizes']:
            unique_sizes = list(set(stats['image_sizes']))
            stats['unique_sizes'] = unique_sizes
            stats['most_common_size'] = max(set(stats['image_sizes']), 
                                          key=stats['image_sizes'].count)
        
        self.logger.info(f"이미지 통계 계산 완료: {stats['valid_images']}개 유효, {stats['invalid_images']}개 무효")
        
        return stats


# 편의 함수들
def preprocess_image_for_prediction(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    단일 이미지를 예측용으로 전처리하는 편의 함수
    
    Args:
        image_path: 이미지 파일 경로
        target_size: 목표 크기
        
    Returns:
        배치 차원이 포함된 전처리 이미지 (1, 높이, 너비, 채널)
    """
    preprocessor = ImagePreprocessor(target_size)
    processed_image = preprocessor.preprocess_single_image(image_path)
    
    # 배치 차원 추가 (모델은 배치 입력을 기대)
    return np.expand_dims(processed_image, axis=0)


def preprocess_pil_image(pil_image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    PIL 이미지를 전처리하는 편의 함수 (웹 업로드용)
    
    Args:
        pil_image: PIL Image 객체
        target_size: 목표 크기
        
    Returns:
        배치 차원이 포함된 전처리 이미지
    """
    # PIL -> numpy 변환
    image = np.array(pil_image)
    
    # RGB 확인 (RGBA인 경우 RGB로 변환)
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # 알파 채널 제거
    
    preprocessor = ImagePreprocessor(target_size)
    
    # 크기 조정 및 정규화
    image = preprocessor.resize_image(image, target_size)
    image = preprocessor.normalize_image(image)
    
    # 배치 차원 추가
    return np.expand_dims(image, axis=0)


if __name__ == "__main__":
    # 테스트 코드
    print("이미지 전처리 모듈 테스트")
    
    # 전처리기 초기화
    preprocessor = ImagePreprocessor()
    
    # 데이터셋 디렉토리가 있다면 테스트
    test_dir = "dataset/processed/train"
    if os.path.exists(test_dir):
        print(f"테스트 디렉토리: {test_dir}")
        
        # 샘플 이미지들 수집
        sample_images = []
        for category in ['brick', 'stucco', 'metal', 'stone', 'wood']:
            category_dir = os.path.join(test_dir, category)
            if os.path.exists(category_dir):
                images = [os.path.join(category_dir, f) for f in os.listdir(category_dir)[:5]]
                sample_images.extend(images)
        
        if sample_images:
            print(f"샘플 이미지 {len(sample_images)}개로 테스트 중...")
            
            # 배치 전처리 테스트
            try:
                batch = preprocessor.preprocess_batch_images(sample_images)
                print(f"배치 전처리 성공: {batch.shape}")
                
                # 통계 정보 계산
                stats = preprocessor.get_image_statistics(sample_images)
                print(f"데이터셋 통계: {stats}")
                
            except Exception as e:
                print(f"테스트 실패: {e}")
        else:
            print("테스트할 이미지가 없습니다.")
    else:
        print(f"테스트 디렉토리가 없습니다: {test_dir}")