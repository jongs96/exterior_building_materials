# -*- coding: utf-8 -*-
"""
빠른 PyTorch 모델 학습 (최적화 버전)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time

def train_fast_model():
    """최적화된 빠른 모델 학습"""
    print("🚀 빠른 PyTorch 모델 학습")
    print("=" * 60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 간단한 데이터 변환 (속도 최적화)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 로딩
    print("\n📂 데이터셋 로딩...")
    data_dir = Path("data/raw")
    
    # 전체 데이터셋 로드
    full_dataset = datasets.ImageFolder(data_dir)
    
    # 훈련/검증 분할 (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(123)
    )
    
    # Transform 적용
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # DataLoader 생성 (배치 크기 증가, num_workers=0으로 Windows 호환)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    
    class_names = full_dataset.classes
    print(f"클래스: {class_names}")
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    print(f"배치 크기: 64")
    
    # 가벼운 모델 사용 (ResNet18)
    print("\n🏗️ ResNet18 모델 생성 (빠른 학습용)...")
    model = models.resnet18(pretrained=True)
    
    # 마지막 FC 레이어만 교체
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    
    model = model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 학습
    print("\n🎯 모델 학습 시작...")
    print("=" * 60)
    
    best_val_acc = 0.0
    epochs = 15  # 에포크 수 감소
    
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 훈련 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # 학습률 조정
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{epochs} ({epoch_time:.1f}s) - "
              f"Train: {train_acc:.3f} ({train_acc*100:.1f}%) - "
              f"Val: {val_acc:.3f} ({val_acc*100:.1f}%)")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, 'building_material_classifier_pytorch.pth')
            print(f"  ✅ 최고 성능 모델 저장!")
        
        # 조기 종료 (75% 달성 시)
        if val_acc >= 0.75:
            print(f"\n🎉 목표 달성! 75% 이상 정확도!")
            break
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"총 학습 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    print(f"최고 검증 정확도: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    
    # 최고 성능 모델 로드
    checkpoint = torch.load('building_material_classifier_pytorch.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 최종 평가
    print("\n📊 최종 평가...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 클래스별 정확도
    print("\n📊 클래스별 성능:")
    print("-" * 40)
    
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            print(f"{class_name:8}: {accuracy:.3f} ({accuracy*100:.1f}%) - {class_correct[i]}/{class_total[i]}")
    
    overall_acc = sum(class_correct) / sum(class_total)
    print("-" * 40)
    print(f"전체    : {overall_acc:.3f} ({overall_acc*100:.1f}%) - {sum(class_correct)}/{sum(class_total)}")
    
    print(f"\n💾 최종 모델 저장: building_material_classifier_pytorch.pth")
    
    return model, class_names

if __name__ == "__main__":
    try:
        model, class_names = train_fast_model()
        print("\n🎉 PyTorch 모델 학습 완료!")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
