# -*- coding: utf-8 -*-
"""
PyTorch 모델이 연결된 Flask API 테스트
"""

import requests
from pathlib import Path
import json

def test_api_with_images():
    """실제 이미지로 API 테스트"""
    print("🧪 PyTorch 모델 API 테스트")
    print("=" * 60)
    
    # API 엔드포인트
    url = "http://127.0.0.1:5000/api/predict"
    
    # 테스트할 이미지 찾기
    data_dir = Path("data/raw")
    
    test_images = []
    for class_name in ['brick', 'metal', 'stone', 'stucco', 'wood']:
        class_dir = data_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg"))[:2]  # 각 클래스에서 2개씩
            test_images.extend([(img, class_name) for img in images])
    
    print(f"테스트 이미지: {len(test_images)}개\n")
    
    # 각 이미지 테스트
    correct = 0
    total = 0
    
    for img_path, true_class in test_images:
        try:
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                # 응답 구조 확인
                if 'data' in result:
                    predicted = result['data']['prediction']['class']
                    confidence = result['data']['prediction']['confidence']
                elif 'prediction' in result:
                    predicted = result['prediction']['class']
                    confidence = result['prediction']['confidence']
                else:
                    print(f"응답 구조: {result}")
                    continue
                
                is_correct = predicted == true_class
                if is_correct:
                    correct += 1
                total += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {img_path.name[:40]:40} | 실제: {true_class:8} | 예측: {predicted:8} | 신뢰도: {confidence:.3f}")
            else:
                print(f"❌ API 오류: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    # 결과 요약
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"정확도: {correct}/{total} = {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_api_with_images()
