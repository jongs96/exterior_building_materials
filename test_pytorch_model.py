# -*- coding: utf-8 -*-
"""
저장된 PyTorch 모델 테스트
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import numpy as np

def test_saved_model():
    """저장된 PyTorch 모델 테스트"""
    print("🔍 저장된 PyTorch 모델 테스트")
    print("=" * 60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 모델 파일 확인
    model_path = 'building_material_classifier_pytorch.pth'
    if not Path(model_path).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    print(f"✅ 모델 파일 발견: {model_path}")
    
    # 데이터 변환
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 로딩
    print("\n📂 데이터셋 로딩...")
    data_dir = Path("data/raw")
    
    full_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    
    # 훈련/검증 분할 (동일한 시드 사용)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(123)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    class_names = full_dataset.classes
    print(f"클래스: {class_names}")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 모델 로드
    print("\n🏗️ 모델 로딩...")
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 체크포인트 구조 확인
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            print("✅ 체크포인트 형식 (state_dict 포함)")
            state_dict = checkpoint['model_state_dict']
            if 'class_names' in checkpoint:
                saved_class_names = checkpoint['class_names']
                print(f"저장된 클래스: {saved_class_names}")
        else:
            print("✅ 직접 state_dict 형식")
            state_dict = checkpoint
    else:
        print("❌ 알 수 없는 체크포인트 형식")
        return
    
    # 모델 구조 추론 (state_dict의 키로부터)
    # layer3의 블록 수로 ResNet 타입 확인
    layer3_blocks = [k for k in state_dict.keys() if k.startswith('layer3.')]
    max_block = max([int(k.split('.')[1]) for k in layer3_blocks if k.split('.')[1].isdigit()], default=1)
    
    if max_block >= 5:  # ResNet50은 layer3에 6개 블록 (0-5)
        print("모델 타입: ResNet50")
        model = models.resnet50(pretrained=False)
        
        # FC 레이어 구조 확인
        if 'fc.1.weight' in state_dict:
            print("FC 레이어: Sequential (Dropout + Linear)")
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, len(class_names))
            )
        else:
            print("FC 레이어: Simple Linear")
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(class_names))
    else:  # ResNet18
        print("모델 타입: ResNet18")
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(class_names))
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("✅ 모델 로드 완료")
    
    # 평가
    print("\n📊 모델 평가 중...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 전체 정확도
    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
    total = len(all_labels)
    overall_acc = correct / total
    
    print(f"\n{'='*60}")
    print(f"전체 정확도: {overall_acc:.3f} ({overall_acc*100:.1f}%) - {correct}/{total}")
    print(f"{'='*60}")
    
    # 클래스별 정확도
    print("\n📊 클래스별 성능:")
    print("-" * 60)
    
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            print(f"{class_name:8}: {accuracy:.3f} ({accuracy*100:.1f}%) - {class_correct[i]:3d}/{class_total[i]:3d}")
    
    # 혼동 행렬
    print("\n📊 혼동 행렬 (실제 → 예측):")
    print("-" * 60)
    
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label][pred] += 1
    
    # 헤더 출력
    header = "실제\\예측"
    print(f"{header:>10}", end="")
    for name in class_names:
        print(f"{name:>8}", end="")
    print()
    print("-" * 60)
    
    # 행렬 출력
    for i, name in enumerate(class_names):
        print(f"{name:>10}", end="")
        for j in range(len(class_names)):
            print(f"{confusion_matrix[i][j]:>8}", end="")
        print()
    
    # 가장 많이 틀린 예측 찾기
    print("\n⚠️ 주요 오분류:")
    print("-" * 60)
    
    misclassifications = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and confusion_matrix[i][j] > 0:
                misclassifications.append((confusion_matrix[i][j], class_names[i], class_names[j]))
    
    misclassifications.sort(reverse=True)
    
    for count, true_class, pred_class in misclassifications[:5]:
        print(f"{true_class:8} → {pred_class:8}: {count}회")
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    try:
        test_saved_model()
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
