# -*- coding: utf-8 -*-
"""
Gunicorn 설정 파일
프로덕션 환경에서 Flask 앱을 실행하기 위한 설정
"""

import multiprocessing

# 서버 소켓
bind = "0.0.0.0:8000"
backlog = 2048

# 워커 프로세스
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# 로깅
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 프로세스 이름
proc_name = "building_material_classifier"

# 서버 메커니즘
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (필요시)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
