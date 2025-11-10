# -*- coding: utf-8 -*-
"""
모델 서비스 - TensorFlow 모델 로딩 및 예측
건축 외장재 분류를 위한 AI 모델 서비스 (더미 모드 없음)
"""

import os
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import threading
import time

logger = logging.getLogger(__name__)

# TensorFlow 지연 로딩을 위한 전역 변수
_tf = None
_tf_lock = threading.Lock()


def get_tensorflow():
    """TensorFlow 모듈을 지연 로딩하는 함수"""
    global _tf
    if _tf is None:
        with _tf_lock:
            if _tf is None:
                try:
                    import tensorflow as tf
                    _tf = tf
                    logger.info("TensorFlow 로드 성공")
                except Exception as e:
                    logger.error(f"TensorFlow 로드 실패: {e}")
                    raise ImportError(f"TensorFlow 초기화 실패: {e}")
    return _tf


class ModelService:
    """건축 외장재 분류 모델 서비스 (실제 AI 모델만 사용)"""
    
    def __init__(self, model_path: str):
        """모델 서비스 초기화"""
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = ['brick', 'metal', 'stone', 'stucco', 'wood']
        self.is_loaded = False
        self._load_lock = threading.Lock()
        
        logger.info(f"ModelService 초기화: {self.model_path}")
    
    def load_model(self) -> bool:
        """실제 TensorFlow 모델을 메모리에 로드"""
        if self.is_loaded:
            return True
            
        with self._load_lock:
            if self.is_loaded:
                return True
                
            try:
                # 모델 파일 존재 확인
                if not self.model_path.exists():
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                
                # TensorFlow 로드
                tf = get_tensorflow()
                
                logger.info(f"실제 모델 로딩 시작: {self.model_path}")
                start_time = time.time()
                
                # 여러 방법으로 모델 로드 시도
                model_loaded = False
                
                # 방법 1: 기본 로딩
                try:
                    self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
                    model_loaded = True
                    logger.info("✅ 기본 모델 로딩 성공")
                except Exception as e1:
                    logger.warning(f"기본 로딩 실패: {e1}")
                    
                    # 방법 2: 호환성 모드
                    try:
                        # TensorFlow 2.13에서 batch_shape 문제 해결
                        import tensorflow.keras.utils as utils
                        
                        # 커스텀 객체 정의
                        custom_objects = {
                            'InputLayer': tf.keras.layers.InputLayer
                        }
                        
                        self.model = tf.keras.models.load_model(
                            str(self.model_path), 
                            compile=False,
                            custom_objects=custom_objects
                        )
                        model_loaded = True
                        logger.info("✅ 호환성 모드 로딩 성공")
                    except Exception as e2:
                        logger.error(f"호환성 모드도 실패: {e2}")
                        
                        # 방법 3: 가중치만 로드
                        try:
                            logger.info("가중치 기반 모델 재구성 시도...")
                            self.model = self._create_model_architecture()
                            
                            # 가중치 파일이 있는지 확인
                            if str(self.model_path).endswith('.h5'):
                                self.model.load_weights(str(self.model_path))
                                model_loaded = True
                                logger.info("✅ 가중치 기반 로딩 성공")
                        except Exception as e3:
                            logger.error(f"가중치 로딩도 실패: {e3}")
                
                if not model_loaded:
                    raise Exception("모든 모델 로딩 방법이 실패했습니다.")
                
                load_time = time.time() - start_time
                
                # 모델 워밍업
                self._warmup_model()
                
                self.is_loaded = True
                logger.info(f"🎉 실제 AI 모델 로드 완료! (소요시간: {load_time:.2f}초)")
                
                if hasattr(self.model, 'input_shape'):
                    logger.info(f"입력 형태: {self.model.input_shape}")
                if hasattr(self.model, 'output_shape'):
                    logger.info(f"출력 형태: {self.model.output_shape}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ 모델 로드 실패: {e}")
                self.is_loaded = False
                self.model = None
                raise Exception(f"실제 모델을 로드할 수 없습니다: {e}")
    
    def _create_model_architecture(self):
        """기본 CNN 아키텍처 생성 (가중치 로딩용)"""
        tf = get_tensorflow()
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Rescaling(1./255),
            
            # Conv 블록들
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # 분류기
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        return model
    
    def _warmup_model(self):
        """모델 워밍업"""
        try:
            if self.model is not None:
                dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
                logger.debug("모델 워밍업 완료")
        except Exception as e:
            logger.warning(f"모델 워밍업 실패: {e}")
    
    def is_model_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.is_loaded and self.model is not None
    
    def predict(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """실제 AI 모델로 이미지 분류 예측"""
        # 모델이 로드되지 않았다면 로드 시도
        if not self.is_model_loaded():
            logger.info("모델이 로드되지 않음. 로드를 시도합니다...")
            if not self.load_model():
                raise Exception("모델을 로드할 수 없습니다.")
        
        try:
            start_time = time.time()
            
            # 입력 검증
            if not isinstance(image, np.ndarray):
                raise ValueError("입력이 numpy 배열이 아닙니다")
                
            if len(image.shape) != 4:
                raise ValueError(f"잘못된 입력 형태: {image.shape}. (1, 224, 224, 3) 형태여야 합니다")
            
            if image.shape[1:] != (224, 224, 3):
                raise ValueError(f"잘못된 이미지 크기: {image.shape[1:]}. (224, 224, 3)이어야 합니다")
            
            # 실제 모델 예측 실행
            logger.debug(f"실제 AI 모델 예측 실행 - 입력 형태: {image.shape}")
            predictions = self.model.predict(image, verbose=0)
            probabilities = predictions[0]
            
            # 결과 처리
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # 모든 클래스별 확률
            class_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            }
            
            processing_time = time.time() - start_time
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': class_probabilities,
                'processing_time': processing_time,
                'model_type': 'real_ai'  # 실제 AI 모델임을 표시
            }
            
            logger.info(f"🤖 실제 AI 예측 완료: {predicted_class} (신뢰도: {confidence:.3f}, 처리시간: {processing_time:.3f}초)")
            
            return result
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            raise Exception(f"AI 모델 예측 실패: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'loaded': self.is_loaded,
            'model_path': str(self.model_path),
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'model_type': 'real_ai'
        }
        
        if self.is_loaded and self.model is not None:
            try:
                info.update({
                    'input_shape': self.model.input_shape,
                    'output_shape': self.model.output_shape,
                    'model_layers': len(self.model.layers),
                    'trainable_params': self.model.count_params(),
                })
            except Exception as e:
                logger.debug(f"모델 상세 정보 조회 실패: {e}")
        
        return info
    
    def unload_model(self):
        """메모리에서 모델 언로드"""
        with self._load_lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.is_loaded = False
                logger.info("모델이 메모리에서 언로드되었습니다")


# 편의 함수
def create_model_service(model_path: str) -> ModelService:
    """ModelService 인스턴스 생성"""
    return ModelService(model_path)


if __name__ == "__main__":
    # 테스트 코드
    print("실제 AI 모델 서비스 테스트")
    
    model_path = "exterior_material_cnn_v1.h5"
    if os.path.exists(model_path):
        service = create_model_service(model_path)
        print(f"ModelService 생성 완료: {service}")
        
        info = service.get_model_info()
        print(f"모델 정보: {info}")
    else:
        print(f"모델 파일이 없습니다: {model_path}")