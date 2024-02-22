[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/3DbKuh4a)
# Document Type Classification Competitions
## Team

|![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9233ab6e-25d5-4c16-8dd4-97a7b8535baf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/e7394268-0f94-4468-8cf5-3cf67e4edd07) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9c75cbd9-f409-4fdd-a5c3-dec082ade3bf) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/388eac05-7cd9-4688-8a87-5b6b742715cf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/48dd674c-ab93-48d1-9e05-e7e8e402597c) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/0a524747-a854-4eee-95b6-108c84514df8) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최장원](https://github.com/UpstageAILab)             |            [김영천](https://github.com/UpstageAILab)             |            [배창현](https://github.com/UpstageAILab)             |            [박성우](https://github.com/UpstageAILab)             |            [조예람](https://github.com/huB-ram)             |            [이소영B](https://github.com/UpstageAILab)             |
|                            팀장                            |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |

## 1. Competitions Info
$\color{red}{\textsf{해당 대회의 Train/Test 데이터는 저작권 문제로 공개가 불가능해 설명을 위해 비슷한 이미지로 대체되어있음을 알려드립니다.}}$<br>
대회에서 실제 활용된 Train/Test에 가까운 데이터는 https://www.content.upstage.ai/ocr-pack/insurance 링크에서 확인해보실 수 있습니다.

### Overview
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/4e685524-05b9-4f48-b980-d24460bb43cb)
해당 대회는 **Upstage AI Lab** 과정에서 비공개로 진행된 내부 대회이며 **다양한 문서를 분류하는 것이 주된 목적**입니다. 하지만 **자동차 계기판, 자동차 번호판** 같은 문서가 아닌 데이터, 그리고 **주민등록증, 여권**과 같은 일반적인 문서와는 조금 이질적인 데이터까지 포함돼 있는 특징이 있습니다. 또한 Train 데이터는 비교적 깨끗한 이미지가 주어지고 **Test 데이터에는 실데이터를 반영해** 다양한 노이즈가 포함된 이미지가 주어졌으며 이에 대응 가능한 모델을 학습시키는 것이 핵심인 대회입니다.

### Environment
Vscode, ssh server(RTX 3090/Ubuntu 20.04.6), pytorch

### Timeline(2 weeks)
- February 05, 2024 - Start Date
- February 19, 2024 - Final submission deadline

### Evaluation
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/1c2bc659-2d35-4678-9a54-6a6671e002c8)
평가지표는 macro f1 스코어로 **클래스 불균형**에도 신뢰성있는 점수를 제공하는 평가지표입니다.

## 2. Components

### Directory

```
├── code
    ├── EDA
    ├── Augmentation
    ├── Modelling
    └── Ensemble
```

## 3. Data descrption

### Dataset overview

이번 대회는 computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회입니다.

이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/65c0f5ec-08f0-4fa6-b176-e57c2cc8868d)

그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

### EDA & Augmentaion
![스크린샷 2024-02-21 151743](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/42354230/1a27b195-36bb-4402-acd1-8a7aa145f0b5)

### Augmentation
![스크린샷 2024-02-21 151956](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/42354230/9ccc0776-d786-4a31-aabc-09f0ff8e50a7)


## 4. Modeling

### Model descrition
![197601575-6a19ed8c-7bc2-433b-895b-e5363358ea77](https://github.com/dudcjs2779/kr-document-type-classification-upstage-competition/assets/42354230/82ff7195-2714-4839-83bc-60f3ffed4b29)
**Used Model:** caformer_s18_384([github](https://github.com/sail-sg/metaformer))

최근에는 Vision Task에서도 **Transforemr 아키텍쳐 기반의 Vision 모델**들이 많이 등장하면서 기존의 Conv 기반의 모델보다 훨씬 좋은 성능을 내고 있고 그에 따라 Vision 모델들도 LLM과 같이 사이즈를 키우는 움직임을 보이며 성능도 비례해서 올라가고 있어 ImageNet Task에서의 상위권을 사이즈가 큰 모델들이 차지하고 있습니다. 하지만 이번 대회에서 제공받은 **GPU의 한계와 2주라는 짧은 시간**으로 인해 사이즈가 큰 모델을 사용하기 힘들다고 판단했고 **MetaFormer 아키텍쳐를 활용한 해당 모델**이 다른 Transformer 기반의 모델에 비해서 **성능도 1~2% 높으며 사이즈도 훨씬 작고 빠르게** 동작한다는 것을 발견했고 해당 모델을 선택하게 되었습니다.

<img src="image/CAFormer Models compare.png/" alt="model compare" width="460" height="400">

### Modeling Process


## Ensemble & TTA
https://github.com/qubvel/ttach?tab=readme-ov-file
https://github.com/qubvel/ttach/blob/master/ttach/wrappers.py #L52

## 5. Result

### Leader Board
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/5e233a57-3e37-4141-885c-220fd5273e59)  
최종 리더보드  
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/4df686f9-9b6d-484c-9c6c-c3c115f3dbfa)


### Presentation
ppt 폴더참조

### Reference
https://paperswithcode.com/  
https://github.com/qubvel/ttach?tab=readme-ov-file  
https://github.com/sail-sg/metaformer  
Focal Loss: https://github.com/mathiaszinnen/focal_loss_torch  
CAFormer: https://arxiv.org/abs/2210.13452  
RotNet: https://github.com/d4nst/RotNet  

대회 데이터 참고용 링크: https://www.content.upstage.ai/ocr-pack/insurance
