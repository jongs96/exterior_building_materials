"""
Flask 애플리케이션 팩토리
애플리케이션 인스턴스를 생성하고 설정하는 모듈입니다.
"""

import os
import logging
from pathlib import Path
from flask import Flask
from flask_cors import CORS

from .settings import config


def create_app(config_name=None):
    """
    Flask 애플리케이션 팩토리 함수
    
    Args:
        config_name: 설정 환경명 ('development', 'production', 'testing')
        
    Returns:
        Flask 애플리케이션 인스턴스
    """
    # 설정 환경 결정
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Flask 앱 생성 (template_folder를 프로젝트 루트 기준으로 설정)
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app = Flask(__name__, 
                template_folder=os.path.join(base_dir, 'templates'),
                static_folder=os.path.join(base_dir, 'static'))
    
    # 설정 로드
    app.config.from_object(config[config_name])
    
    # 업로드 폴더 생성
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    upload_folder.mkdir(exist_ok=True)
    
    # CORS 설정
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # CSP 완전 제거를 위한 after_request 핸들러
    @app.after_request
    def remove_csp_header(response):
        # CSP 헤더 완전 제거
        if 'Content-Security-Policy' in response.headers:
            del response.headers['Content-Security-Policy']
        return response
    
    # 로깅 설정
    setup_logging(app)
    
    # 블루프린트 등록
    register_blueprints(app)
    
    # 에러 핸들러 등록
    register_error_handlers(app)
    
    return app


def setup_logging(app):
    """로깅 설정"""
    if not app.debug and not app.testing:
        # 프로덕션 환경에서의 로깅 설정
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Building Material Classifier startup')


def register_blueprints(app):
    """블루프린트 등록"""
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')


def setup_security(app):
    """보안 설정"""
    from app.middleware.security import SecurityMiddleware, validate_security_config
    
    # 보안 미들웨어 초기화
    security = SecurityMiddleware(app)
    
    # 보안 설정 검증
    validate_security_config(app)
    
    app.logger.info('보안 미들웨어가 활성화되었습니다.')


def register_error_handlers(app):
    """에러 핸들러 등록"""
    from flask import jsonify
    
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Page not found', 'code': 'NOT_FOUND'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'서버 오류: {error}')
        return jsonify({'error': 'Internal server error', 'code': 'INTERNAL_ERROR'}), 500
    
    @app.errorhandler(413)
    def too_large(error):
        return jsonify({
            'error': 'File too large. Maximum size is 10MB.',
            'code': 'FILE_TOO_LARGE'
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            'error': 'Too many requests. Please try again later.',
            'code': 'RATE_LIMIT_EXCEEDED'
        }), 429
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'error': 'Access forbidden',
            'code': 'FORBIDDEN'
        }), 403