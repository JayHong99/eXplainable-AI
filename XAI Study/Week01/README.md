# XAI 스터디

|주차|일자|내용|Study URL or Content|정리자료|
|---|---|---|---|---|
|Week1|2022.02.06|XAI의 정의 등|https://www.youtube.com/watch?v=Grc7egfZP84|
|Week1|2022.02.06|Lime & SHAP|https://www.youtube.com/watch?v=BQSkV95Dy4s||

# 1. XAI Overview

## Introduction & Motivation
- XAI에 대한 관심도가 늘어나는 추세
    - AI의 결과만을 단순히 신뢰하기에는 risk가 크다.
    - 의사결정론적인 측면에서 설명이 필요하다.
- XAI의 필요성이 늘어남
    - 모델의 판단 근거를 바탕으로 피드백 가능
    - 자율주행 자동차 등 원인을 찾을 수 있음
- XAI의 정의
    - 인간의 Explanation이 아닌 AI가 Explanation을 도출하며, 사람이 AI의 동작과 최종 결과를 이애하고 올바르게 해석할 수 있고, 결과물이 생성되는 과정을 설명 가능하도록 해주는 기술
- 기대효과
    - 투명성 : 인간이 이해할 수 있다.
    - 신뢰도 : 왜 그런 결과를 냈는지 알 수 있다
    - Bias에 대한 이해

## XAI의 종류
### 1. 분류기준 : Scope
- Local
    - 하나의 input x에 대한 결과 해석
    - example ) 환자 1명에 대한 의료 진료 근거
    - 방법론
        - Actiavtion Maximization
        - Saliency Map Visualization
        - LRP
- Global
    - 모델 전체를 하나로 인식하여, x집합 X에 대한 결과 해석
    - 복잡한 deep model을 linear counter parts로 축소하여 해석하기 쉽게함
    - Non-linear하면 tree based로 설명하기도 함
    - 최종 도출된 explanation map g는 "HOW"에 대한 설명이 주를 이룸
        - 얼마나 내 모델이 정형화 되어있는가
        - 어떻게 내 모델이 perform 하는가
    - 방법론
        - Class Model Visualization
        - CAVs
        - TCAVs
        - SpRAy
### 2. 분류기준 : Methodology
- Gradient Based XAI
    - 역전파 gradient를 통해 influence나 relevence를 계산
    - 방법론
        - Saliency Maps
        - Grad CAM
        - Salinet Relevance Maps
        - Attribution Map
- Perturbation based XAI   
    - Input 데이터의 feature간의 변화가 있었을때의 정도 측정
    - 방법론
        - Deconvolution Nets for Convolution Visualizations
        - Rise
### 3. 분류기준 : Useage
- Intrinsic
    - 다른 모델에 적용 불가능
    - Decision Tree 등
- Post-Hoc
    - 모델 구조에 구애받지 않음
    - 범용적으로 적용 가능
    - Deconvolution Net, Saliency maps 등

## XAI 평가
- System Causability Scale (SCS)
    - XAI에서 중요한 지표 10가지에 대한 점수 체계화
- Benchmarking Attribution Methods
    - Feature Attribution과 relative importance에 대한 측정 가능
    - 3가지 점수가 있음
        - Model Contrast Scores
        - Input Dependence Rate
        - Input Independence rate
- Faithfulness and Monotonicity
    - Important score와 performance effect 간의 상관관계
    - Monotonicity : feature를 점점 추가하면서 나타나는 feature 성능 변화
- Human-grounded Evaluation Benchmark
    - 사람의 평가 기반
    - 다양한 의견을 수집해 human-bias를 제거하는 연구
## 한계점
- Explanation map 해석의 어려움
    - 이미지에 약간의 shift를 주었을 때, acc에는 변화가 없지만 attention map은 다른 정보로 판단함.
- XAI에 대한 정량적 평가가 어려움

## 2. Lime & SHAP
### 1) Lime
- Local Interpretable Model - agnostic Explanations
- 특정 관측치 x에 대한 설명
- 모델과 정확도 사이의 trade-off를 이용하여, 특정 모델을 예측하는데 있어서의 변수 중요도를 예측
    - 복잡한 모델 f가 예측하는 값과 해석 모델 g가 예측하는 값 사이의 차이를 사용
- 특징
    - 모든 모델에 적용 가능하며, 다양한 데이터에 유연하게 적용 가능
    - 특정 관측치 하나의 deicision boundary에 대한 해석 제공

### 2) SHAP
- Shapley Values를 기반으로 Feature 기여도 예측
- 게임이론 중, 상금을 분배하기 위한 player의 기여도를 판단하는 방법
    - Player A가 있을 때와 없을 때의 상금을 바탕으로 기여도를 측정
    - Player의 Marginal Contribution에 가중평균을 곱하여 계산
    - Marginal Contribution
        - A가 없는 부분집합에 A를 추가했을때의 상금
    - 가중평균
        - 플레이어 수의 따른 A의 기여도의 크기
    - 플레이어 수가 적지만, A가 있을 때의 상금이 커진다면 기여도가 커짐
- 단점
    - 모든 부분집합과 경우의 수를 판단하기에 시간이 오래 걸림
    - 이를 위하여 모델별 적합한 SHAP을 만드는 추세