/**
 * 건축 외장재 분류기 메인 JavaScript
 * 공통 기능 및 유틸리티 함수들
 */

// 전역 설정
const CONFIG = {
    API_BASE_URL: '/api',
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    SUPPORTED_FORMATS: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
    ANIMATION_DURATION: 300
};

// 유틸리티 함수들
const Utils = {
    /**
     * 파일 크기를 읽기 쉬운 형태로 변환
     */
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * 파일 유효성 검사
     */
    validateFile: function(file) {
        const errors = [];

        // 파일 타입 검사
        if (!CONFIG.SUPPORTED_FORMATS.includes(file.type)) {
            errors.push('지원하지 않는 파일 형식입니다. (JPG, PNG, WEBP만 지원)');
        }

        // 파일 크기 검사
        if (file.size > CONFIG.MAX_FILE_SIZE) {
            errors.push(`파일 크기가 너무 큽니다. (최대 ${Utils.formatFileSize(CONFIG.MAX_FILE_SIZE)})`);
        }

        // 파일 크기가 너무 작은 경우
        if (file.size < 1024) {
            errors.push('파일 크기가 너무 작습니다.');
        }

        return {
            isValid: errors.length === 0,
            errors: errors
        };
    },

    /**
     * API 호출 래퍼
     */
    apiCall: async function(endpoint, options = {}) {
        try {
            const response = await fetch(CONFIG.API_BASE_URL + endpoint, {
                ...options,
                headers: {
                    ...options.headers
                }
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error('API 호출 오류:', error);
            throw error;
        }
    },

    /**
     * 로딩 스피너 표시/숨김
     */
    showLoading: function(element, show = true) {
        if (show) {
            element.innerHTML = `
                <div class="d-flex align-items-center justify-content-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status">
                        <span class="visually-hidden">로딩 중...</span>
                    </div>
                    처리 중...
                </div>
            `;
            element.disabled = true;
        } else {
            element.disabled = false;
        }
    },

    /**
     * 토스트 알림 표시
     */
    showToast: function(message, type = 'info') {
        // Bootstrap 토스트가 있다면 사용, 없으면 alert
        if (typeof bootstrap !== 'undefined' && bootstrap.Toast) {
            const toastHtml = `
                <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
                    <div class="d-flex">
                        <div class="toast-body">${message}</div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>
            `;
            
            // 토스트 컨테이너가 없으면 생성
            let toastContainer = document.getElementById('toast-container');
            if (!toastContainer) {
                toastContainer = document.createElement('div');
                toastContainer.id = 'toast-container';
                toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
                document.body.appendChild(toastContainer);
            }
            
            toastContainer.insertAdjacentHTML('beforeend', toastHtml);
            const toastElement = toastContainer.lastElementChild;
            const toast = new bootstrap.Toast(toastElement);
            toast.show();
            
            // 토스트가 숨겨진 후 DOM에서 제거
            toastElement.addEventListener('hidden.bs.toast', () => {
                toastElement.remove();
            });
        } else {
            alert(message);
        }
    },

    /**
     * 요소 애니메이션
     */
    animateElement: function(element, animationClass, duration = CONFIG.ANIMATION_DURATION) {
        return new Promise((resolve) => {
            element.classList.add(animationClass);
            setTimeout(() => {
                element.classList.remove(animationClass);
                resolve();
            }, duration);
        });
    },

    /**
     * 이미지 압축 (선택사항)
     */
    compressImage: function(file, maxWidth = 1024, quality = 0.8) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = function() {
                // 비율 유지하면서 크기 조정
                const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
                canvas.width = img.width * ratio;
                canvas.height = img.height * ratio;

                // 이미지 그리기
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                // Blob으로 변환
                canvas.toBlob(resolve, 'image/jpeg', quality);
            };

            img.src = URL.createObjectURL(file);
        });
    },

    /**
     * 디바운스 함수
     */
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * 로컬 스토리지 헬퍼
     */
    storage: {
        set: function(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
            } catch (e) {
                console.warn('로컬 스토리지 저장 실패:', e);
            }
        },
        
        get: function(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.warn('로컬 스토리지 읽기 실패:', e);
                return defaultValue;
            }
        },
        
        remove: function(key) {
            try {
                localStorage.removeItem(key);
            } catch (e) {
                console.warn('로컬 스토리지 삭제 실패:', e);
            }
        }
    }
};

// 전역 이벤트 리스너
document.addEventListener('DOMContentLoaded', function() {
    // 네트워크 상태 모니터링
    window.addEventListener('online', () => {
        Utils.showToast('인터넷 연결이 복구되었습니다.', 'success');
    });

    window.addEventListener('offline', () => {
        Utils.showToast('인터넷 연결이 끊어졌습니다.', 'warning');
    });

    // 전역 에러 핸들러
    window.addEventListener('error', (event) => {
        console.error('전역 에러:', event.error);
        Utils.showToast('예상치 못한 오류가 발생했습니다.', 'danger');
    });

    // API 상태 확인
    checkApiHealth();
});

/**
 * API 상태 확인
 */
async function checkApiHealth() {
    try {
        const health = await Utils.apiCall('/health');
        console.log('API 상태:', health);
        
        if (!health.model_loaded) {
            Utils.showToast('AI 모델 로딩 중입니다. 잠시 후 다시 시도해주세요.', 'warning');
        }
    } catch (error) {
        console.error('API 상태 확인 실패:', error);
        Utils.showToast('서버 연결에 문제가 있습니다.', 'danger');
    }
}

/**
 * 페이지 성능 모니터링
 */
if ('performance' in window) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log('페이지 로드 시간:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
        }, 0);
    });
}

// 전역 객체로 노출
window.Utils = Utils;
window.CONFIG = CONFIG;