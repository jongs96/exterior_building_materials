# -*- coding: utf-8 -*-
"""
PyTorch 모델 서비스 - 건축 외장재 분류
"""

import os
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import threading
import time

logger = logging.getLogger(__name__)

# PyTorch 지연 로딩을 위한 전역 변수
_torch = None
_torch_lock = threading.Lock()


def get_torch():
    """PyTorch 모듈을 지연 로딩하는 함수"""
    global _torch
    if _torch is None:
        with _torch_lock:
            if _torch is None:
                try:
                    import torch
                    _torch = torch
                    logger.info("PyTorch 로드 성공")
                except Exception as e:
                    logger.error(f"PyTorch 로드 실패: {e}")
                    raise ImportError(f"PyTorch 초기화 실패: {e}")
    return _torch


class ModelService:
    """건축 외장재 분류 PyTorch 모델 서비스"""
    
    def __init__(self, model_path: str):
        """모델 서비스 초기화"""
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = ['brick', 'metal', 'stone', 'stucco', 'wood']
        self.is_loaded = False
        self.device = None
        self._load_lock = threading.Lock()
        
        logger.info(f"PyTorch ModelService 초기화: {self.model_path}")
    
    def load_model(self) -> bool:
        """PyTorch 모델을 메모리에 로드"""
        if self.is_loaded:
            return True
            
        with self._load_lock:
            if self.is_loaded:
                return True
                
            try:
                # 모델 파일 존재 확인
                if not self.model_path.exists():
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                
                # PyTorch 로드
                torch = get_torch()
                import torch.nn as nn
                from torchvision import models
                
                logger.info(f"PyTorch 모델 로딩 시작: {self.model_path}")
                start_time = time.time()
                
                # 디바이스 설정
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"디바이스: {self.device}")
                
                # 체크포인트 로드
                checkpoint = torch.load(str(self.model_path), map_location=self.device)
                
                # 체크포인트 구조 확인
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if 'class_names' in checkpoint:
                        self.class_names = checkpoint['class_names']
                        logger.info(f"저장된 클래스: {self.class_names}")
                else:
                    state_dict = checkpoint
                
                # 모델 구조 추론 (layer3의 블록 수로 ResNet 타입 확인)
                layer3_blocks = [k for k in state_dict.keys() if k.startswith('layer3.')]
                max_block = max([int(k.split('.')[1]) for k in layer3_blocks if k.split('.')[1].isdigit()], default=1)
                
                if max_block >= 5:  # ResNet50
                    logger.info("모델 타입: ResNet50")
                    self.model = models.resnet50(weights=None)
                    
                    # FC 레이어 구조 확인
                    if 'fc.1.weight' in state_dict:
                        logger.info("FC 레이어: Sequential (Dropout + Linear)")
                        num_features = self.model.fc.in_features
                        self.model.fc = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(num_features, 512),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(512, len(self.class_names))
                        )
                    else:
                        logger.info("FC 레이어: Simple Linear")
                        num_features = self.model.fc.in_features
                        self.model.fc = nn.Linear(num_features, len(self.class_names))
                else:  # ResNet18
                    logger.info("모델 타입: ResNet18")
                    self.model = models.resnet18(weights=None)
                    num_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(num_features, len(self.class_names))
                
                # 가중치 로드
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                load_time = time.time() - start_time
                
                # 모델 워밍업
                self._warmup_model()
                
                self.is_loaded = True
                logger.info(f"🎉 PyTorch 모델 로드 완료! (소요시간: {load_time:.2f}초)")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ 모델 로드 실패: {e}")
                import traceback
                traceback.print_exc()
                self.is_loaded = False
                self.model = None
                raise Exception(f"PyTorch 모델을 로드할 수 없습니다: {e}")
    
    def _warmup_model(self):
        """모델 워밍업"""
        try:
            if self.model is not None:
                torch = get_torch()
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                logger.debug("모델 워밍업 완료")
        except Exception as e:
            logger.warning(f"모델 워밍업 실패: {e}")
    
    def is_model_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.is_loaded and self.model is not None
    
    def predict(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """PyTorch 모델로 이미지 분류 예측"""
        # 모델이 로드되지 않았다면 로드 시도
        if not self.is_model_loaded():
            logger.info("모델이 로드되지 않음. 로드를 시도합니다...")
            if not self.load_model():
                raise Exception("모델을 로드할 수 없습니다.")
        
        try:
            torch = get_torch()
            start_time = time.time()
            
            # 입력 검증
            if not isinstance(image, np.ndarray):
                raise ValueError("입력이 numpy 배열이 아닙니다")
                
            if len(image.shape) != 4:
                raise ValueError(f"잘못된 입력 형태: {image.shape}. (1, 224, 224, 3) 형태여야 합니다")
            
            if image.shape[1:] != (224, 224, 3):
                raise ValueError(f"잘못된 이미지 크기: {image.shape[1:]}. (224, 224, 3)이어야 합니다")
            
            # NumPy (H, W, C) -> PyTorch (C, H, W) 변환
            # 입력: (1, 224, 224, 3) -> (1, 3, 224, 224)
            # 주의: image_service에서 이미 정규화가 완료된 상태
            image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).float()
            image_tensor = image_tensor.to(self.device)
            
            # 예측 실행
            logger.debug(f"PyTorch 모델 예측 실행 - 입력 형태: {image_tensor.shape}")
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
            
            # CPU로 이동 및 NumPy 변환
            probabilities = probabilities.cpu().numpy()
            
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
                'model_type': 'pytorch_resnet'
            }
            
            logger.info(f"🤖 PyTorch 예측 완료: {predicted_class} (신뢰도: {confidence:.3f}, 처리시간: {processing_time:.3f}초)")
            
            return result
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"PyTorch 모델 예측 실패: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'loaded': self.is_loaded,
            'model_path': str(self.model_path),
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'model_type': 'pytorch_resnet',
            'device': str(self.device) if self.device else 'not_set'
        }
        
        if self.is_loaded and self.model is not None:
            try:
                # 파라미터 수 계산
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                info.update({
                    'total_params': total_params,
                    'trainable_params': trainable_params,
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
    print("PyTorch 모델 서비스 테스트")
    
    model_path = "building_material_classifier_pytorch.pth"
    if os.path.exists(model_path):
        service = create_model_service(model_path)
        print(f"ModelService 생성 완료: {service}")
        
        try:
            service.load_model()
            info = service.get_model_info()
            print(f"모델 정보: {info}")
        except Exception as e:
            print(f"오류: {e}")
    else:
        print(f"모델 파일이 없습니다: {model_path}")
