# -*- coding: utf-8 -*-
"""
PyTorch 기반 건축 외장재 분류 모델 학습
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time

def train_pytorch_model():
    """PyTorch 모델 학습"""
    print("🚀 PyTorch 기반 건축 외장재 분류 모델 학습")
    print("=" * 60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 데이터 변환 정의
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    class_names = full_dataset.classes
    print(f"클래스: {class_names}")
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 모델 생성
    print("\n🏗️ ResNet50 모델 생성...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # 마지막 FC 레이어 교체
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(class_names))
    )
    
    model = model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 학습
    print("\n🎯 모델 학습 시작...")
    
    best_val_acc = 0.0
    epochs = 25
    
    for epoch in range(epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
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
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'building_material_classifier_pytorch.pth')
            print(f"  ✅ 최고 성능 모델 저장! (검증 정확도: {val_acc:.4f})")
        
        # 조기 종료 (80% 달성 시)
        if val_acc >= 0.80:
            print(f"🎉 목표 달성! 80% 이상 정확도!")
            break
    
    # 최고 성능 모델 로드
    model.load_state_dict(torch.load('building_material_classifier_pytorch.pth'))
    
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
    print(f"최고 검증 정확도: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    
    return model, class_names

if __name__ == "__main__":
    try:
        model, class_names = train_pytorch_model()
        print("\n🎉 PyTorch 모델 학습 완료!")
        print(f"클래스: {class_names}")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()