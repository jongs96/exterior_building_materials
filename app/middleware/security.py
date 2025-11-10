# -*- coding: utf-8 -*-
"""
보안 미들웨어
API 보안, Rate Limiting, CORS 등을 처리
"""

import time
import hashlib
from collections import defaultdict, deque
from functools import wraps
from flask import request, jsonify, current_app, g
import logging

logger = logging.getLogger(__name__)

# Rate Limiting 저장소 (프로덕션에서는 Redis 사용 권장)
rate_limit_storage = defaultdict(lambda: deque())
blocked_ips = set()

class SecurityMiddleware:
    """보안 미들웨어 클래스"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Flask 앱에 보안 미들웨어 등록"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """요청 전 보안 검사"""
        # IP 주소 추출
        client_ip = self.get_client_ip()
        g.client_ip = client_ip
        
        # 차단된 IP 확인
        if client_ip in blocked_ips:
            logger.warning(f"차단된 IP에서 접근 시도: {client_ip}")
            return jsonify({
                'error': '접근이 차단되었습니다.',
                'code': 'IP_BLOCKED'
            }), 403
        
        # Rate Limiting 검사
        if not self.check_rate_limit(client_ip):
            logger.warning(f"Rate limit 초과: {client_ip}")
            return jsonify({
                'error': '요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.',
                'code': 'RATE_LIMIT_EXCEEDED'
            }), 429
        
        # 요청 크기 검사
        if request.content_length and request.content_length > current_app.config.get('MAX_CONTENT_LENGTH', 16*1024*1024):
            logger.warning(f"요청 크기 초과: {request.content_length} bytes from {client_ip}")
            return jsonify({
                'error': '요청 크기가 너무 큽니다.',
                'code': 'REQUEST_TOO_LARGE'
            }), 413
        
        # 의심스러운 User-Agent 검사
        user_agent = request.headers.get('User-Agent', '')
        if self.is_suspicious_user_agent(user_agent):
            logger.warning(f"의심스러운 User-Agent: {user_agent} from {client_ip}")
            # 경고만 로그하고 차단하지는 않음
        
        # 요청 로깅
        logger.info(f"Request: {request.method} {request.path} from {client_ip}")
    
    def after_request(self, response):
        """응답 후 보안 헤더 추가"""
        # 보안 헤더 설정 (CSP 완전 제거)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # CSP 헤더 완전 제거 (개발 환경)
        if 'Content-Security-Policy' in response.headers:
            del response.headers['Content-Security-Policy']
        
        # HTTPS 강제 (프로덕션에서)
        if current_app.config.get('FORCE_HTTPS', False):
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response
    
    def get_client_ip(self):
        """클라이언트 IP 주소 추출"""
        # 프록시 헤더들 확인
        forwarded_ips = request.headers.getlist("X-Forwarded-For")
        if forwarded_ips:
            return forwarded_ips[0].split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        return request.remote_addr or 'unknown'
    
    def check_rate_limit(self, client_ip):
        """Rate Limiting 검사"""
        now = time.time()
        window = 60  # 1분 윈도우
        max_requests = 30  # 분당 최대 30회 요청
        
        # 현재 IP의 요청 기록
        requests = rate_limit_storage[client_ip]
        
        # 윈도우 밖의 오래된 요청들 제거
        while requests and requests[0] < now - window:
            requests.popleft()
        
        # 현재 요청 수 확인
        if len(requests) >= max_requests:
            # 너무 많은 요청 시 일시적으로 IP 차단
            if len(requests) >= max_requests * 2:
                blocked_ips.add(client_ip)
                logger.error(f"IP 차단: {client_ip} (과도한 요청)")
            return False
        
        # 현재 요청 기록
        requests.append(now)
        return True
    
    def is_suspicious_user_agent(self, user_agent):
        """의심스러운 User-Agent 검사"""
        suspicious_patterns = [
            'bot', 'crawler', 'spider', 'scraper',
            'curl', 'wget', 'python-requests',
            'scanner', 'exploit', 'hack'
        ]
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)


def require_api_key(f):
    """API 키 인증 데코레이터 (선택적)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # API 키가 설정된 경우에만 검증
        api_key = current_app.config.get('API_KEY')
        if api_key:
            provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            if not provided_key or provided_key != api_key:
                return jsonify({
                    'error': '유효하지 않은 API 키입니다.',
                    'code': 'INVALID_API_KEY'
                }), 401
        
        return f(*args, **kwargs)
    return decorated_function


def validate_content_type(allowed_types):
    """Content-Type 검증 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            content_type = request.content_type
            if content_type not in allowed_types:
                return jsonify({
                    'error': f'지원하지 않는 Content-Type입니다. 허용된 타입: {", ".join(allowed_types)}',
                    'code': 'INVALID_CONTENT_TYPE'
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def log_security_event(event_type, details, severity='INFO'):
    """보안 이벤트 로깅"""
    client_ip = getattr(g, 'client_ip', 'unknown')
    user_agent = request.headers.get('User-Agent', 'unknown')
    
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'event_type': event_type,
        'client_ip': client_ip,
        'user_agent': user_agent,
        'details': details,
        'severity': severity
    }
    
    if severity == 'ERROR':
        logger.error(f"Security Event: {log_entry}")
    elif severity == 'WARNING':
        logger.warning(f"Security Event: {log_entry}")
    else:
        logger.info(f"Security Event: {log_entry}")


def sanitize_filename(filename):
    """파일명 안전화"""
    import re
    from werkzeug.utils import secure_filename
    
    # 기본 안전화
    safe_name = secure_filename(filename)
    
    # 추가 정리
    safe_name = re.sub(r'[^\w\-_\.]', '', safe_name)
    safe_name = safe_name[:100]  # 길이 제한
    
    return safe_name


def generate_csrf_token():
    """CSRF 토큰 생성"""
    import secrets
    return secrets.token_urlsafe(32)


def validate_csrf_token(token):
    """CSRF 토큰 검증"""
    # 간단한 구현 - 실제로는 세션에 저장된 토큰과 비교
    return token and len(token) == 43  # base64url 32바이트 = 43문자


# 보안 설정 검증
def validate_security_config(app):
    """보안 설정 검증"""
    warnings = []
    
    # SECRET_KEY 확인
    if not app.config.get('SECRET_KEY') or app.config['SECRET_KEY'] == 'dev':
        warnings.append("SECRET_KEY가 설정되지 않았거나 기본값을 사용하고 있습니다.")
    
    # HTTPS 설정 확인
    if not app.config.get('FORCE_HTTPS') and not app.debug:
        warnings.append("프로덕션 환경에서 HTTPS가 강제되지 않습니다.")
    
    # 파일 크기 제한 확인
    if app.config.get('MAX_CONTENT_LENGTH', 0) > 50*1024*1024:
        warnings.append("파일 크기 제한이 너무 큽니다 (50MB 초과).")
    
    # 경고 출력
    for warning in warnings:
        logger.warning(f"보안 설정 경고: {warning}")
    
    return len(warnings) == 0