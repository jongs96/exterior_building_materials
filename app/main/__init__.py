"""
메인 블루프린트
웹 페이지 라우팅을 담당하는 블루프린트입니다.
"""

from flask import Blueprint

bp = Blueprint('main', __name__)

from app.main import routes