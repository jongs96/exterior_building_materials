# -*- coding: utf-8 -*-
"""
종합적 에러 처리 시스템
사용자 친화적 에러 메시지 및 복구 가이드 제공
"""

import logging
import traceback
from functools import wraps
from flask import jsonify, request, current_app, g
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 에러 코드별 사용자 친화적 메시지
ERROR_MESSAGES = {
    # 파일 업로드 관련
    'NO_FILE': {
        'message': '파일이 업로드되지 않았습니다.',
        'solution': '이미지 파일을 선택하거나 드래그해서 업로드해주세요.',
        'severity': 'warning'
    },
    'NO_FILENAME': {
        'message': '파일명이 없습니다.',
        'solution': '유효한 파일명을 가진 이미지를 선택해주세요.',
        'severity': 'warning'
    },
    'INVALID_FORMAT': {
        'message': '지원하지 않는 파일 형식입니다.',
        'solution': 'JPG, PNG, WEBP 형식의 이미지 파일을 사용해주세요.',
        'severity': 'warning'
    },
    'FILE_TOO_LARGE': {
        'message': '파일 크기가 너무 큽니다.',
        'solution': '10MB 이하의 이미지를 사용하거나 이미지를 압축해주세요.',
        'severity': 'warning'
    },
    'FILE_TOO_SMALL': {
        'message': '파일이 너무 작습니다.',
        'solution': '유효한 이미지 파일인지 확인해주세요.',
        'severity': 'warning'
    },
    'INVALID_IMAGE': {
        'message': '유효하지 않은 이미지 파일입니다.',
        'solution': '손상되지 않은 이미지 파일을 사용해주세요.',
        'severity': 'error'
    },
    'IMAGE_TOO_SMALL': {
        'message': '이미지 해상도가 너무 낮습니다.',
        'solution': '더 높은 해상도의 이미지를 사용해주세요.',
        'severity': 'warning'
    },
    'RESOLUTION_TOO_HIGH': {
        'message': '이미지 해상도가 너무 높습니다.',
        'solution': '4K(4096x4096) 이하의 해상도로 이미지를 조정해주세요.',
        'severity': 'warning'
    },
    
    # 보안 관련
    'MALICIOUS_CONTENT': {
        'message': '보안상 위험한 파일입니다.',
        'solution': '안전한 이미지 파일만 업로드해주세요.',
        'severity': 'error'
    },
    'INVALID_SIGNATURE': {
        'message': '파일 형식이 올바르지 않습니다.',
        'solution': '정상적인 이미지 파일을 사용해주세요.',
        'severity': 'error'
    },
    'RATE_LIMIT_EXCEEDED': {
        'message': '요청 한도를 초과했습니다.',
        'solution': '잠시 후 다시 시도해주세요.',
        'severity': 'warning'
    },
    'IP_BLOCKED': {
        'message': '접근이 차단되었습니다.',
        'solution': '관리자에게 문의해주세요.',
        'severity': 'error'
    },
    
    # 처리 관련
    'PROCESSING_ERROR': {
        'message': '이미지 처리 중 오류가 발생했습니다.',
        'solution': '다른 이미지를 시도하거나 잠시 후 다시 시도해주세요.',
        'severity': 'error'
    },
    'PREDICTION_ERROR': {
        'message': 'AI 분석 중 오류가 발생했습니다.',
        'solution': '잠시 후 다시 시도해주세요. 문제가 지속되면 관리자에게 문의해주세요.',
        'severity': 'error'
    },
    'MODEL_NOT_LOADED': {
        'message': 'AI 모델이 로드되지 않았습니다.',
        'solution': '잠시 후 다시 시도해주세요. 서버가 시작 중일 수 있습니다.',
        'severity': 'error'
    },
    
    # 카메라 관련
    'CAMERA_ERROR': {
        'message': '카메라 접근 오류가 발생했습니다.',
        'solution': '브라우저에서 카메라 권한을 허용하고 다시 시도해주세요.',
        'severity': 'warning'
    },
    'DECODE_ERROR': {
        'message': '이미지 데이터 처리 오류입니다.',
        'solution': '다시 촬영하거나 파일 업로드를 시도해주세요.',
        'severity': 'error'
    },
    
    # 일반 오류
    'INTERNAL_ERROR': {
        'message': '서버 내부 오류가 발생했습니다.',
        'solution': '잠시 후 다시 시도해주세요. 문제가 지속되면 관리자에게 문의해주세요.',
        'severity': 'error'
    },
    'VALIDATION_ERROR': {
        'message': '데이터 검증 중 오류가 발생했습니다.',
        'solution': '입력 데이터를 확인하고 다시 시도해주세요.',
        'severity': 'error'
    },
    'NETWORK_ERROR': {
        'message': '네트워크 연결 오류입니다.',
        'solution': '인터넷 연결을 확인하고 다시 시도해주세요.',
        'severity': 'warning'
    }
}


class ApplicationError(Exception):
    """애플리케이션 커스텀 예외"""
    
    def __init__(self, code: str, message: str = None, details: Dict = None, status_code: int = 400):
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, {}).get('message', '알 수 없는 오류입니다.')
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


def handle_application_error(error: ApplicationError) -> Tuple[Dict, int]:
    """애플리케이션 에러 처리"""
    error_info = ERROR_MESSAGES.get(error.code, {})
    
    response = {
        'success': False,
        'error': error.message,
        'code': error.code,
        'solution': error_info.get('solution', '다시 시도해주세요.'),
        'severity': error_info.get('severity', 'error'),
        'timestamp': get_timestamp()
    }
    
    # 개발 환경에서는 상세 정보 추가
    if current_app.debug and error.details:
        response['details'] = error.details
    
    # 로깅
    log_error(error.code, error.message, error.details, error_info.get('severity', 'error'))
    
    return response, error.status_code


def safe_api_call(f):
    """API 호출 안전 래퍼 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        
        except ApplicationError as e:
            return jsonify(*handle_application_error(e))
        
        except Exception as e:
            # 예상치 못한 오류 처리
            error_id = log_unexpected_error(e)
            
            response = {
                'success': False,
                'error': '예상치 못한 오류가 발생했습니다.',
                'code': 'UNEXPECTED_ERROR',
                'solution': '잠시 후 다시 시도해주세요. 문제가 지속되면 관리자에게 문의해주세요.',
                'severity': 'error',
                'timestamp': get_timestamp()
            }
            
            # 개발 환경에서는 상세 오류 정보 추가
            if current_app.debug:
                response['debug_info'] = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'error_id': error_id
                }
            
            return jsonify(response), 500
    
    return decorated_function


def log_error(code: str, message: str, details: Dict = None, severity: str = 'error'):
    """에러 로깅"""
    client_ip = getattr(g, 'client_ip', 'unknown')
    user_agent = request.headers.get('User-Agent', 'unknown')
    
    log_data = {
        'error_code': code,
        'message': message,
        'client_ip': client_ip,
        'user_agent': user_agent,
        'endpoint': request.endpoint,
        'method': request.method,
        'url': request.url,
        'timestamp': get_timestamp()
    }
    
    if details:
        log_data['details'] = details
    
    if severity == 'error':
        logger.error(f"Application Error: {log_data}")
    elif severity == 'warning':
        logger.warning(f"Application Warning: {log_data}")
    else:
        logger.info(f"Application Info: {log_data}")


def log_unexpected_error(error: Exception) -> str:
    """예상치 못한 오류 로깅"""
    import uuid
    
    error_id = str(uuid.uuid4())[:8]
    client_ip = getattr(g, 'client_ip', 'unknown')
    
    log_data = {
        'error_id': error_id,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'client_ip': client_ip,
        'endpoint': request.endpoint,
        'method': request.method,
        'url': request.url,
        'timestamp': get_timestamp(),
        'traceback': traceback.format_exc()
    }
    
    logger.error(f"Unexpected Error: {log_data}")
    
    return error_id


def get_timestamp() -> str:
    """현재 타임스탬프 반환"""
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S')


def validate_request_data(required_fields: list, data: Dict) -> None:
    """요청 데이터 검증"""
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ApplicationError(
            'MISSING_FIELDS',
            f'필수 필드가 누락되었습니다: {", ".join(missing_fields)}',
            {'missing_fields': missing_fields}
        )


def create_success_response(data: Any, message: str = '성공적으로 처리되었습니다.') -> Dict:
    """성공 응답 생성"""
    response = {
        'success': True,
        'message': message,
        'timestamp': get_timestamp()
    }
    
    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response['data'] = data
    
    return response


def create_error_response(code: str, message: str = None, details: Dict = None) -> Tuple[Dict, int]:
    """에러 응답 생성"""
    error = ApplicationError(code, message, details)
    return handle_application_error(error)


# 에러 복구 가이드
RECOVERY_GUIDES = {
    'file_upload': {
        'title': '파일 업로드 문제 해결',
        'steps': [
            '1. 이미지 파일 형식을 확인하세요 (JPG, PNG, WEBP)',
            '2. 파일 크기가 10MB 이하인지 확인하세요',
            '3. 이미지가 손상되지 않았는지 확인하세요',
            '4. 브라우저를 새로고침하고 다시 시도하세요'
        ]
    },
    'camera': {
        'title': '카메라 문제 해결',
        'steps': [
            '1. 브라우저에서 카메라 권한을 허용하세요',
            '2. 다른 앱에서 카메라를 사용 중이지 않은지 확인하세요',
            '3. 브라우저를 새로고침하고 다시 시도하세요',
            '4. 파일 업로드 방식을 대신 사용해보세요'
        ]
    },
    'network': {
        'title': '네트워크 문제 해결',
        'steps': [
            '1. 인터넷 연결 상태를 확인하세요',
            '2. 잠시 후 다시 시도하세요',
            '3. 브라우저 캐시를 삭제하고 다시 시도하세요',
            '4. 다른 브라우저에서 시도해보세요'
        ]
    },
    'server': {
        'title': '서버 문제 해결',
        'steps': [
            '1. 잠시 후 다시 시도하세요',
            '2. 페이지를 새로고침하세요',
            '3. 문제가 지속되면 관리자에게 문의하세요'
        ]
    }
}


def get_recovery_guide(error_code: str) -> Optional[Dict]:
    """에러 코드에 따른 복구 가이드 반환"""
    if error_code in ['NO_FILE', 'INVALID_FORMAT', 'FILE_TOO_LARGE', 'INVALID_IMAGE']:
        return RECOVERY_GUIDES['file_upload']
    elif error_code in ['CAMERA_ERROR', 'DECODE_ERROR']:
        return RECOVERY_GUIDES['camera']
    elif error_code in ['NETWORK_ERROR']:
        return RECOVERY_GUIDES['network']
    elif error_code in ['INTERNAL_ERROR', 'PREDICTION_ERROR']:
        return RECOVERY_GUIDES['server']
    
    return None