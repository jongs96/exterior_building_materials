#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask 애플리케이션 실행 스크립트
"""

import os
from app import create_app

# 환경 설정
env = os.environ.get('FLASK_ENV', 'development')

# 앱 생성
app = create_app(env)

if __name__ == '__main__':
    # 개발 서버 실행
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=(env == 'development')
    )
