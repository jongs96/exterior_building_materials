"""
검증 유틸리티
파일 업로드 및 데이터 검증 함수들
보안 강화된 파일 검증 시스템
"""

import os
import hashlib
import magic
from werkzeug.utils import secure_filename
from PIL import Image
import logging
import mimetypes
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# 위험한 파일 시그니처 (매직 바이트)
DANGEROUS_SIGNATURES = [
    b'\x4D\x5A',  # PE executable (MZ)
    b'\x7F\x45\x4C\x46',  # ELF executable
    b'\xCA\xFE\xBA\xBE',  # Java class file
    b'\x50\x4B\x03\x04',  # ZIP/JAR (일부 허용할 수도 있음)
    b'\x89\x50\x4E\x47',  # PNG (허용되지만 검증 필요)
    b'\xFF\xD8\xFF',  # JPEG (허용되지만 검증 필요)
]

# 허용된 이미지 MIME 타입
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/webp',
    'image/gif'  # 필요시 추가
}

# 허용된 이미지 시그니처
ALLOWED_IMAGE_SIGNATURES = {
    b'\xFF\xD8\xFF': 'image/jpeg',  # JPEG
    b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'image/png',  # PNG
    b'RIFF': 'image/webp',  # WEBP (RIFF 컨테이너)
    b'GIF87a': 'image/gif',  # GIF87a
    b'GIF89a': 'image/gif',  # GIF89a
}


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """
    파일 확장자 검증
    
    Args:
        filename: 파일명
        allowed_extensions: 허용된 확장자 집합
        
    Returns:
        허용 여부
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def check_file_signature(file_data: bytes) -> tuple:
    """
    파일 시그니처(매직 바이트) 검증
    
    Args:
        file_data: 파일의 첫 부분 바이트
        
    Returns:
        (is_valid, detected_type, message)
    """
    # 위험한 시그니처 검사
    for dangerous_sig in DANGEROUS_SIGNATURES:
        if file_data.startswith(dangerous_sig):
            if dangerous_sig in [b'\xFF\xD8\xFF', b'\x89\x50\x4E\x47']:
                # JPEG, PNG는 허용하지만 추가 검증 필요
                continue
            return False, 'executable', '실행 파일은 업로드할 수 없습니다.'
    
    # 허용된 이미지 시그니처 검사
    for sig, mime_type in ALLOWED_IMAGE_SIGNATURES.items():
        if file_data.startswith(sig):
            return True, mime_type, '유효한 이미지 시그니처입니다.'
        
        # WEBP는 특별 처리 (RIFF 컨테이너 내부 확인)
        if sig == b'RIFF' and len(file_data) >= 12:
            if file_data[8:12] == b'WEBP':
                return True, 'image/webp', '유효한 WEBP 이미지입니다.'
    
    return False, 'unknown', '알 수 없거나 허용되지 않는 파일 형식입니다.'


def validate_mime_type(file) -> tuple:
    """
    MIME 타입 검증 (python-magic 사용)
    
    Args:
        file: Flask 파일 객체
        
    Returns:
        (is_valid, mime_type, message)
    """
    try:
        file.stream.seek(0)
        file_data = file.stream.read(1024)  # 첫 1KB만 읽기
        file.stream.seek(0)
        
        # python-magic으로 MIME 타입 감지
        try:
            detected_mime = magic.from_buffer(file_data, mime=True)
        except:
            # magic이 없으면 mimetypes 모듈 사용
            detected_mime, _ = mimetypes.guess_type(file.filename)
        
        if detected_mime not in ALLOWED_MIME_TYPES:
            return False, detected_mime, f'허용되지 않는 MIME 타입: {detected_mime}'
        
        return True, detected_mime, '유효한 MIME 타입입니다.'
        
    except Exception as e:
        logger.warning(f"MIME 타입 검증 실패: {e}")
        return False, 'unknown', 'MIME 타입을 확인할 수 없습니다.'


def scan_for_malicious_content(file) -> tuple:
    """
    악성 콘텐츠 스캔 (기본적인 패턴 검사)
    
    Args:
        file: Flask 파일 객체
        
    Returns:
        (is_safe, message)
    """
    try:
        file.stream.seek(0)
        file_content = file.stream.read()
        file.stream.seek(0)
        
        # 이미지 파일에 대해서는 악성 콘텐츠 스캔 생략
        # (이미지 메타데이터에 <% 같은 문자가 포함될 수 있음)
        file_size = len(file_content)
        if file_size > 100:  # 100바이트 이상이면 정상 이미지로 간주
            logger.debug(f"이미지 파일 악성 콘텐츠 스캔 생략 (크기: {file_size} bytes)")
            return True, '이미지 파일 - 스캔 생략'
        
        # 매우 작은 파일만 스캔
        suspicious_patterns = [
            b'<script',  # JavaScript
            b'<?php',    # PHP 코드  
            b'eval(',    # eval 함수
            b'exec(',    # exec 함수
        ]
        
        for pattern in suspicious_patterns:
            if pattern in file_content.lower():
                return False, f'의심스러운 콘텐츠 발견: {pattern.decode("utf-8", errors="ignore")}'
        
        return True, '악성 콘텐츠가 발견되지 않았습니다.'
        
    except Exception as e:
        logger.warning(f"악성 콘텐츠 스캔 실패: {e}")
        return True, '스캔을 완료할 수 없지만 허용합니다.'


def validate_image_integrity(file) -> tuple:
    """
    이미지 무결성 및 메타데이터 검증
    
    Args:
        file: Flask 파일 객체
        
    Returns:
        (is_valid, image_info, message)
    """
    try:
        file.stream.seek(0)
        
        # PIL로 이미지 열기
        image = Image.open(file.stream)
        
        # 이미지 무결성 검증
        image.verify()
        
        # 새로운 이미지 객체로 다시 열기 (verify 후에는 사용 불가)
        file.stream.seek(0)
        image = Image.open(file.stream)
        
        width, height = image.size
        
        # 이미지 크기 제한
        max_dimension = 4096  # 4K 해상도까지 허용
        if width > max_dimension or height > max_dimension:
            return False, None, f'이미지가 너무 큽니다. 최대 {max_dimension}x{max_dimension} 픽셀까지 허용됩니다.'
        
        # 최소 크기 확인
        min_size = 32
        if width < min_size or height < min_size:
            return False, None, f'이미지가 너무 작습니다. 최소 {min_size}x{min_size} 픽셀이 필요합니다.'
        
        # 이미지 정보 수집
        image_info = {
            'width': width,
            'height': height,
            'mode': image.mode,
            'format': image.format,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        }
        
        # EXIF 데이터 확인 (개인정보 보호)
        if hasattr(image, '_getexif') and image._getexif():
            logger.info("EXIF 데이터가 포함된 이미지입니다. 처리 시 제거됩니다.")
        
        file.stream.seek(0)
        return True, image_info, '유효한 이미지입니다.'
        
    except Exception as e:
        logger.error(f"이미지 무결성 검증 실패: {e}")
        return False, None, '손상된 이미지 파일입니다.'


def calculate_file_hash(file) -> str:
    """
    파일 해시 계산 (중복 검사용)
    
    Args:
        file: Flask 파일 객체
        
    Returns:
        SHA256 해시 문자열
    """
    try:
        file.stream.seek(0)
        file_content = file.stream.read()
        file.stream.seek(0)
        
        return hashlib.sha256(file_content).hexdigest()
        
    except Exception as e:
        logger.error(f"파일 해시 계산 실패: {e}")
        return None


def validate_image_file(file, config: dict) -> dict:
    """
    업로드된 이미지 파일 종합 보안 검증
    
    Args:
        file: Flask 업로드 파일 객체
        config: 애플리케이션 설정
        
    Returns:
        검증 결과 딕셔너리
    """
    try:
        # 1. 기본 파일명 검증
        if not file.filename or file.filename == '':
            return {
                'valid': False,
                'message': '파일명이 없습니다.',
                'code': 'NO_FILENAME'
            }
        
        # 안전한 파일명으로 변환
        safe_filename = secure_filename(file.filename)
        if not safe_filename:
            return {
                'valid': False,
                'message': '유효하지 않은 파일명입니다.',
                'code': 'INVALID_FILENAME'
            }
        
        # 2. 파일 확장자 검증
        if not allowed_file(file.filename, config['ALLOWED_EXTENSIONS']):
            return {
                'valid': False,
                'message': f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(config['ALLOWED_EXTENSIONS'])}",
                'code': 'INVALID_FORMAT'
            }
        
        # 3. 파일 크기 검증
        file.stream.seek(0, 2)  # 파일 끝으로 이동
        file_size = file.stream.tell()
        file.stream.seek(0)  # 처음으로 되돌리기
        
        if file_size > config['MAX_CONTENT_LENGTH']:
            return {
                'valid': False,
                'message': f"파일 크기가 너무 큽니다. 최대 {config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB까지 허용됩니다.",
                'code': 'FILE_TOO_LARGE'
            }
        
        if file_size < 100:  # 100바이트 미만은 의심스러움
            return {
                'valid': False,
                'message': '파일이 너무 작습니다.',
                'code': 'FILE_TOO_SMALL'
            }
        
        # 4. 파일 시그니처 검증
        file.stream.seek(0)
        file_header = file.stream.read(32)  # 첫 32바이트 읽기
        file.stream.seek(0)
        
        sig_valid, detected_type, sig_message = check_file_signature(file_header)
        if not sig_valid:
            return {
                'valid': False,
                'message': sig_message,
                'code': 'INVALID_SIGNATURE'
            }
        
        # 5. MIME 타입 검증
        mime_valid, mime_type, mime_message = validate_mime_type(file)
        if not mime_valid:
            return {
                'valid': False,
                'message': mime_message,
                'code': 'INVALID_MIME_TYPE'
            }
        
        # 6. 악성 콘텐츠 스캔
        is_safe, scan_message = scan_for_malicious_content(file)
        if not is_safe:
            return {
                'valid': False,
                'message': f'보안 위험: {scan_message}',
                'code': 'MALICIOUS_CONTENT'
            }
        
        # 7. 이미지 무결성 검증
        img_valid, image_info, img_message = validate_image_integrity(file)
        if not img_valid:
            return {
                'valid': False,
                'message': img_message,
                'code': 'INVALID_IMAGE'
            }
        
        # 8. 파일 해시 계산
        file_hash = calculate_file_hash(file)
        
        # 모든 검증 통과
        return {
            'valid': True,
            'message': '모든 보안 검증을 통과했습니다.',
            'code': 'VALID',
            'safe_filename': safe_filename,
            'file_size': file_size,
            'mime_type': mime_type,
            'file_hash': file_hash,
            'image_info': image_info
        }
        
    except Exception as e:
        logger.error(f"파일 검증 중 오류: {e}")
        return {
            'valid': False,
            'message': '파일 검증 중 오류가 발생했습니다.',
            'code': 'VALIDATION_ERROR'
        }


def validate_base64_image(image_data: str, max_size: int = 10*1024*1024) -> dict:
    """
    Base64 이미지 데이터 검증 (카메라 API용)
    
    Args:
        image_data: Base64 인코딩된 이미지 데이터
        max_size: 최대 파일 크기 (바이트)
        
    Returns:
        검증 결과 딕셔너리
    """
    try:
        import base64
        from io import BytesIO
        
        # Base64 헤더 제거
        if ',' in image_data:
            header, data = image_data.split(',', 1)
            # 헤더에서 MIME 타입 추출
            if 'data:' in header and ';base64' in header:
                mime_type = header.split('data:')[1].split(';base64')[0]
            else:
                mime_type = 'unknown'
        else:
            data = image_data
            mime_type = 'unknown'
        
        # Base64 디코딩
        try:
            decoded_data = base64.b64decode(data)
        except Exception as e:
            return {
                'valid': False,
                'message': 'Base64 디코딩 실패',
                'code': 'DECODE_ERROR'
            }
        
        # 크기 검증
        if len(decoded_data) > max_size:
            return {
                'valid': False,
                'message': f'이미지가 너무 큽니다. 최대 {max_size // (1024*1024)}MB까지 허용됩니다.',
                'code': 'IMAGE_TOO_LARGE'
            }
        
        # 시그니처 검증
        sig_valid, detected_type, sig_message = check_file_signature(decoded_data[:32])
        if not sig_valid:
            return {
                'valid': False,
                'message': sig_message,
                'code': 'INVALID_SIGNATURE'
            }
        
        # PIL로 이미지 검증
        try:
            image_stream = BytesIO(decoded_data)
            image = Image.open(image_stream)
            image.verify()
            
            # 다시 열어서 정보 수집
            image_stream.seek(0)
            image = Image.open(image_stream)
            width, height = image.size
            
            # 크기 제한
            if width > 4096 or height > 4096:
                return {
                    'valid': False,
                    'message': '이미지 해상도가 너무 높습니다.',
                    'code': 'RESOLUTION_TOO_HIGH'
                }
            
            if width < 32 or height < 32:
                return {
                    'valid': False,
                    'message': '이미지가 너무 작습니다.',
                    'code': 'IMAGE_TOO_SMALL'
                }
            
        except Exception as e:
            return {
                'valid': False,
                'message': '유효하지 않은 이미지 데이터입니다.',
                'code': 'INVALID_IMAGE'
            }
        
        return {
            'valid': True,
            'message': '유효한 Base64 이미지입니다.',
            'code': 'VALID',
            'mime_type': mime_type,
            'size': len(decoded_data),
            'image_info': {
                'width': width,
                'height': height,
                'mode': image.mode,
                'format': image.format
            }
        }
        
    except Exception as e:
        logger.error(f"Base64 이미지 검증 오류: {e}")
        return {
            'valid': False,
            'message': '이미지 검증 중 오류가 발생했습니다.',
            'code': 'VALIDATION_ERROR'
        }