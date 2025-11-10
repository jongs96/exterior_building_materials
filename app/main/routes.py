"""
메인 웹 페이지 라우트
사용자 인터페이스를 위한 HTML 페이지 라우팅
"""

from flask import render_template, current_app
from app.main import bp


@bp.route('/')
@bp.route('/index')
def index():
    """
    메인 페이지
    이미지 업로드 및 분류 인터페이스를 제공합니다.
    """
    return render_template('index.html', title='메인')


@bp.route('/camera')
def camera():
    """
    카메라 촬영 페이지
    실시간 카메라를 통한 이미지 분류 인터페이스를 제공합니다.
    """
    return render_template('camera.html',
                         title='실시간 카메라 분류',
                         class_names=current_app.config['CLASS_NAMES'])


@bp.route('/about')
def about():
    """
    서비스 소개 페이지
    건축 외장재 분류 서비스에 대한 설명을 제공합니다.
    """
    return render_template('about.html',
                         title='서비스 소개',
                         class_descriptions=current_app.config['CLASS_DESCRIPTIONS'])


@bp.route('/help')
def help():
    """
    도움말 페이지
    서비스 사용법과 FAQ를 제공합니다.
    """
    return render_template('help.html', title='도움말')