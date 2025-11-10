"""
API 블루프린트
REST API 엔드포인트를 담당하는 블루프린트입니다.
"""

from flask import Blueprint

bp = Blueprint('api', __name__)

from app.api import routes