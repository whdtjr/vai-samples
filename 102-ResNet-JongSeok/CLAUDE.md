

# Vulkan GPU 프레임워크를 이용한 ResNet34 구현
이 프로젝트는 11-mnist-refactor의 vai-samples 프레임워크를 기반으로, GPU 가속을 위해 Vulkan 컴퓨트 셰이더를 사용하여 ResNet34 신경망을 구현합니다.

## 프로젝트 구조
main.cpp - ResNet34 그래프 구성 및 추론을 위한 메인 진입점

neuralNodes.h/cpp - 신경망 노드 구현 (Conv, BatchNorm, ReLU, Add 등)

neuralNet.h - 신경망 그래프 관리

tensor.h - 텐서(Tensor) 자료 구조

vulkanApp.h/cpp - Vulkan 애플리케이션 프레임워크

convert_npz_to_json.py - NPZ 가중치를 JSON 형식으로 변환하는 유틸리티

## 구현된 기능
신경망 노드
ConvolutionNode: 스트라이드(stride)를 지원하는 2D 컨볼루션

BatchNormalizationNode: 배치 정규화

ReluNode: ReLU 활성화 함수

MaxPoolingNode: 최대 풀링

GlobalAvgPoolNode: 전역 평균 풀링

AddNode: 스킵 커넥션(skip connection)을 위한 요소별 덧셈

FullyConnectedNode: 완전 연결 레이어

FlattenNode: 다차원 텐서를 1차원으로 평탄화

BasicBlock: 스킵 커넥션을 포함하는 ResNet 기본 블록 (NodeGroup)

ResNet34 아키텍처
초기 레이어: Conv7x7 (stride=2) → BatchNorm → ReLU → MaxPool (2x2)

레이어 1: 3× BasicBlock(64 채널)

레이어 2: 4× BasicBlock(128 채널, 첫 번째 블록 stride=2)

레이어 3: 6× BasicBlock(256 채널, 첫 번째 블록 stride=2)

레이어 4: 3× BasicBlock(512 채널, 첫 번째 블록 stride=2)

최종 레이어: Global Average Pool → FC(1000 클래스)

## 빌드 및 실행
사전 요구 사항
- Vulkan SDK

- CMake 3.16+

- C++23 컴파일러

- Python 3 (가중치 변환용)

## 빌드 단계
프로젝트 빌드:

```Bash

cd /home/js/workspace/vai-samples/102-ResNet-JongSeok
mkdir -p build
cd build
cmake ..
make

```
```bash
# execute
./102-ResNet-JongSeok
```
참고: 이 프로젝트에는 이제 커스텀 NPZ 로더가 포함되어 있어, JSON으로 변환할 필요 없이 resnet34_weights.npz 파일을 직접 사용할 수 있습니다!

### 입력 데이터
이미지: dog.jpg (어떤 크기든 224x224로 리사이즈됨)

가중치: resnet34_weights.npz (커스텀 NPZ 로더를 통해 직접 로드)

입력 전처리: ImageNet 정규화 (평균=[0.485, 0.456, 0.406], 표준편차=[0.229, 0.224, 0.225])

### NPZ 로더
이 프로젝트는 .npz 파일을 직접 읽을 수 있는 커스텀 NPZ (NumPy ZIP) 로더를 포함합니다. 이 로더는:

ZIP 아카이브 구조를 파싱합니다.

NPY 형식 헤더를 읽습니다.

배열 데이터 및 형태(shape)를 추출합니다.

비압축(stored) 및 deflate 압축 NPZ 파일을 모두 지원합니다.

zlib를 사용하여 압축을 해제합니다.

ResNet34의 182개 가중치 배열을 모두 성공적으로 로드합니다.

## 구현 세부 사항
스트라이드를 이용한 컨볼루션
컨볼루션 연산은 스트라이드를 지원하는 im2col 변환을 사용합니다:

입력 이미지 패치를 컬럼(column)으로 변환합니다.

GEMM (행렬 곱) 연산을 수행합니다.

임의의 스트라이드 값을 지원합니다.

배치 정규화
표준 배치 정규화 공식을 구현합니다:

y = gamma * (x - mean) / sqrt(var + eps) + beta
스킵 커넥션
BasicBlock은 잔차 연결(residual connection)을 구현합니다:

메인 경로: Conv3x3 → BN → ReLU → Conv3x3 → BN

스킵 경로: Identity (항등) 또는 Conv1x1+BN (stride≠1 또는 채널 변경 시)

Add → ReLU

GPU 컴퓨트 셰이더
모든 연산은 Vulkan 컴퓨트 셰이더로 구현됩니다:

공유 메모리를 사용한 최적화된 GEMM 커널

병렬 배치 정규화

효율적인 풀링 연산


## 프로그램은 다음을 출력합니다:

이미지 로딩 정보

그래프 구조 확인

Top-5 예측 클래스 및 신뢰도 점수

##  참고 사항
⚠️ 가중치 로딩: 현재 구현은 기본 구조만 보여줍니다. 모든 ResNet34 레이어의 전체 가중치를 로드하려면 NPZ 가중치 이름을 각 레이어의 파라미터에 매핑하는 작업이 완료되어야 합니다.

⚠️ 성능: 이것은 데모용 구현입니다. 실제 상용 환경에서 사용하려면 추가 최적화가 이루어질 수 있습니다.

📚 참고 자료
원본 ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)

vai-samples 프레임워크 기반: 11-mnist-refactor

GPU 가속을 위해 Vulkan 컴퓨트 셰이더 사용