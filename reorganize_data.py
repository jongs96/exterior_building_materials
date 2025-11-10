# -*- coding: utf-8 -*-
"""
데이터 재분류 스크립트
파일 이름을 기준으로 올바른 폴더로 이동
"""

import shutil
from pathlib import Path

def reorganize_data():
    """파일 이름 기준으로 데이터 재분류"""
    print("🔄 데이터 재분류 시작")
    print("=" * 50)
    
    data_dir = Path("data/raw")
    
    # 클래스별 키워드 매핑
    class_keywords = {
        'brick': ['벽돌', '조적', 'brick'],
        'metal': ['금속', '패널', 'metal'],
        'stone': ['석재', '돌', 'stone'],
        'stucco': ['스타코', '미장', 'stucco'],
        'wood': ['목재', '사이딩', 'wood']
    }
    
    moved_count = {class_name: 0 for class_name in class_keywords.keys()}
    error_count = 0
    
    # 각 폴더의 파일들을 확인
    for current_class in class_keywords.keys():
        current_dir = data_dir / current_class
        
        if not current_dir.exists():
            continue
        
        print(f"\n📁 {current_class} 폴더 확인 중...")
        
        # 이미지 파일 찾기
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
        image_files = [f for f in current_dir.iterdir() if f.is_file() and f.suffix in image_extensions]
        
        for img_file in image_files:
            file_name = img_file.name
            
            # 파일 이름에서 올바른 클래스 찾기
            correct_class = None
            for class_name, keywords in class_keywords.items():
                if any(keyword in file_name for keyword in keywords):
                    correct_class = class_name
                    break
            
            # 올바른 클래스를 찾지 못한 경우
            if correct_class is None:
                print(f"  ⚠️ 분류 불가: {file_name}")
                error_count += 1
                continue
            
            # 현재 폴더가 올바른 폴더가 아닌 경우 이동
            if correct_class != current_class:
                target_dir = data_dir / correct_class
                target_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = target_dir / file_name
                
                # 파일 이동
                try:
                    shutil.move(str(img_file), str(target_path))
                    print(f"  ✅ {file_name[:40]:40} -> {correct_class}")
                    moved_count[correct_class] += 1
                except Exception as e:
                    print(f"  ❌ 이동 실패: {file_name} - {e}")
                    error_count += 1
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("📊 재분류 결과:")
    print(f"{'='*50}")
    
    total_moved = sum(moved_count.values())
    
    for class_name, count in moved_count.items():
        if count > 0:
            print(f"{class_name:8}: {count}개 파일 이동")
    
    print(f"\n총 이동: {total_moved}개")
    print(f"오류: {error_count}개")
    
    # 재분류 후 각 폴더의 파일 수 확인
    print(f"\n{'='*50}")
    print("📊 재분류 후 파일 수:")
    print(f"{'='*50}")
    
    for class_name in class_keywords.keys():
        class_dir = data_dir / class_name
        if class_dir.exists():
            image_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix in {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}]
            print(f"{class_name:8}: {len(image_files)}개")

if __name__ == "__main__":
    reorganize_data()
    print("\n🎉 데이터 재분류 완료!")