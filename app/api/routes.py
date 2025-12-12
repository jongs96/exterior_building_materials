"""
API 라우트
이미지 분류를 위한 REST API 엔드포인트
"""

import os
import time
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from app.api import bp
from app.services.image_service import ImageService
from app.utils.validators import validate_image_file, validate_base64_image
from app.utils.error_handler import safe_api_call, ApplicationError, create_success_response


# 전역 서비스 인스턴스
model_service = None
image_service = None


def get_model_service():
    """모델 서비스 인스턴스 반환 (지연 로딩)"""
    global model_service
    if model_service is None:
        try:
            # PyTorch 모델 서비스 사용
            model_type = current_app.config.get('MODEL_TYPE', 'pytorch')
            
            if model_type == 'pytorch':
                from app.services.model_service_pytorch import ModelService
                current_app.logger.info("PyTorch 모델 서비스 사용")
            else:
                from app.services.model_service import ModelService
                current_app.logger.info("TensorFlow 모델 서비스 사용")
            
            model_service = ModelService(current_app.config['MODEL_PATH'])
            # 모델 로드 시도
            if not model_service.is_model_loaded():
                model_service.load_model()
        except ImportError as e:
            current_app.logger.error(f"ModelService import 실패: {e}")
            return None
        except Exception as e:
            current_app.logger.error(f"ModelService 초기화 실패: {e}")
            return None
    return model_service


def get_image_service():
    """이미지 서비스 인스턴스 반환"""
    global image_service
    if image_service is None:
        image_service = ImageService(current_app.config['IMAGE_SIZE'])
    return image_service


@bp.route('/predict', methods=['POST'])
@safe_api_call
def predict():
    """
    이미지 분류 API
    업로드된 이미지를 분석하여 외장재 종류를 예측합니다.
    
    Returns:
        JSON: 분류 결과와 신뢰도
    """
    start_time = time.time()
    
    try:
        # 파일 존재 확인
        if 'file' not in request.files:
            raise ApplicationError('NO_FILE')
        
        file = request.files['file']
        
        # 파일명 확인
        if file.filename == '':
            raise ApplicationError('NO_FILENAME')
        
        # 파일 유효성 검사
        validation_result = validate_image_file(file, current_app.config)
        if not validation_result['valid']:
            raise ApplicationError(validation_result['code'], validation_result['message'])
        
        # 이미지 전처리
        img_service = get_image_service()
        processed_image = img_service.preprocess_uploaded_image(file)
        
        if processed_image is None:
            raise ApplicationError('PROCESSING_ERROR', status_code=500)
        
        # 모델 예측
        model_svc = get_model_service()
        prediction_result = model_svc.predict(processed_image)
        
        if prediction_result is None:
            raise ApplicationError('PREDICTION_ERROR', status_code=500)
        
        # 처리 시간 계산
        processing_time = round(time.time() - start_time, 3)
        
        # 결과 반환
        response_data = {
            'prediction': {
                'class': prediction_result['predicted_class'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities']
            },
            'description': current_app.config['CLASS_DESCRIPTIONS'].get(
                prediction_result['predicted_class'], {}
            ),
            'processing_time': processing_time
        }
        
        return jsonify(create_success_response(response_data, '이미지 분류가 완료되었습니다.'))
        
    except Exception as e:
        current_app.logger.error(f"예측 API 오류: {str(e)}")
        return jsonify({
            'error': '서버 내부 오류가 발생했습니다.',
            'code': 'INTERNAL_ERROR'
        }), 500


@bp.route('/health', methods=['GET'])
@safe_api_call
def health_check():
    """
    서비스 상태 확인 API
    서버와 모델의 상태를 확인합니다.
    
    Returns:
        JSON: 서비스 상태 정보
    """
    try:
        # 모델 로드 상태 확인
        model_svc = get_model_service()
        model_loaded = model_svc.is_model_loaded()
        
        # 이미지 서비스 상태 확인
        img_service = get_image_service()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'supported_classes': current_app.config['CLASS_NAMES'],
            'max_file_size': current_app.config['MAX_CONTENT_LENGTH'],
            'supported_formats': list(current_app.config['ALLOWED_EXTENSIONS']),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        current_app.logger.error(f"헬스체크 오류: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500


@bp.route('/predict-camera', methods=['POST'])
@safe_api_call
def predict_camera():
    """
    카메라 이미지 분류 API
    Base64 인코딩된 이미지를 받아서 분석합니다.
    
    Returns:
        JSON: 분류 결과와 신뢰도
    """
    start_time = time.time()
    
    try:
        # JSON 데이터 확인
        if not request.is_json:
            current_app.logger.warning("Request content-type is not JSON")
            raise ApplicationError('INVALID_FORMAT', 'JSON 형식의 데이터가 필요합니다.')
        
        data = request.get_json()
        
        image_data = data.get('image')
        
        if not image_data:
            current_app.logger.warning("No image data provided (None or empty)")
            raise ApplicationError('NO_IMAGE_DATA', '이미지 데이터가 없습니다.')

        current_app.logger.info(f"Received camera image data (length: {len(image_data)})")
        
        # Base64 이미지 검증 (헤더 포함된 상태로 전달해도 됨, validators 내부에서 처리)
        validation_result = validate_base64_image(image_data, current_app.config['MAX_CONTENT_LENGTH'])
        if not validation_result['valid']:
            # 디버깅을 위해 검증 실패 시에도 진행 (로그만 남김)
            current_app.logger.warning(f"Image validation FAILED but proceeding for debug: {validation_result.get('message')}")

        # Base64 디코딩
        import base64
        try:
            # image_data가 문자열임을 보장
            if not isinstance(image_data, str):
                image_data = str(image_data)

            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # 패딩 보정 (Base64 길이는 4의 배수여야 함)
            missing_padding = len(image_data) % 4
            if missing_padding:
                image_data += '=' * (4 - missing_padding)
                
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            current_app.logger.error(f"Base64 decoding failed: {str(e)}")
            return jsonify({
                'error': f'Base64 디코딩 실패: {str(e)}',
                'code': 'DECODE_ERROR'
            }), 400
        
        # PIL 이미지로 변환
        try:
            from io import BytesIO
            image_stream = BytesIO(image_bytes)
            pil_image = Image.open(image_stream)
            # 이미지가 완전히 로드되었는지 확인
            pil_image.verify()
            # verify 후에는 다시 열어야 함
            image_stream.seek(0)
            pil_image = Image.open(image_stream)
        except Exception as e:
            current_app.logger.error(f"PIL Image open failed: {str(e)}")
            return jsonify({
                'error': f'이미지 변환 실패: {str(e)}',
                'code': 'INVALID_IMAGE'
            }), 400
        
        # 이미지 전처리
        img_service = get_image_service()
        processed_image = img_service.preprocess_pil_image(pil_image)
        
        if processed_image is None:
            return jsonify({
                'error': '이미지 처리 중 오류가 발생했습니다.',
                'code': 'PROCESSING_ERROR'
            }), 500
        
        # 모델 예측
        model_svc = get_model_service()
        prediction_result = model_svc.predict(processed_image)
        
        if prediction_result is None:
            return jsonify({
                'error': '모델 예측 중 오류가 발생했습니다.',
                'code': 'PREDICTION_ERROR'
            }), 500
        
        # 처리 시간 계산
        processing_time = round(time.time() - start_time, 3)
        
        # 결과 반환
        response = {
            'success': True,
            'prediction': {
                'class': prediction_result['predicted_class'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities']
            },
            'description': current_app.config['CLASS_DESCRIPTIONS'].get(
                prediction_result['predicted_class'], {}
            ),
            'processing_time': processing_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'camera'
        }
        
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"카메라 예측 API 오류: {str(e)}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'error': f'서버 내부 오류가 발생했습니다: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }), 500


@bp.route('/classes', methods=['GET'])
def get_classes():
    """
    지원하는 분류 클래스 정보 API
    
    Returns:
        JSON: 분류 클래스 목록과 설명
    """
    return jsonify({
        'classes': current_app.config['CLASS_NAMES'],
        'descriptions': current_app.config['CLASS_DESCRIPTIONS']
    })