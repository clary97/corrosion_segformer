# Corrosion SegFormer

SegFormer 기반 **부식(corrosion) Semantic Segmentation** 학습 코드입니다.
강재 표면 이미지를 픽셀 단위로 분류하여 부식 등급을 예측합니다.

- **4-class 모드**: `Good` / `Fair` / `Poor` / `Severe`
- **이진 모드**: `Good` / `Corrosion` (Fair·Poor·Severe를 하나로 통합)

백본은 HuggingFace `nvidia/mit-b2` (SegFormer-B2)를 사용합니다.

---

## 디렉토리 구조

```
corrosion_segformer/
├── training/
│   ├── main.py          # CLI 진입점 (인자 파싱, 모델/프로세서 로드)
│   ├── datahandler.py   # 데이터셋·데이터로더, 마스크→라벨 변환, 증강
│   └── trainer.py       # 학습 루프, F1/IoU 로깅, best 체크포인트 저장
├── configs/             # 설정 파일
├── preprocessing/       # 전처리 스크립트
├── figures/             # 결과 시각화
├── requirements.txt
└── README.md
```

데이터셋은 저장소 밖의 별도 경로에 둡니다(`--data_dir`로 지정).

---

## 데이터셋 형식

```
<data_dir>/
├── Train/
│   ├── Images/   # 입력 이미지 (512×512, RGB)
│   └── Masks/    # 라벨 마스크 (512×512, PNG, BGR 색상 인코딩)
└── Test/
    ├── Images/
    └── Masks/
```

- 이미지와 마스크는 **확장자가 달라도** 파일명(stem) 기준으로 매칭됩니다 (예: `img_0.jpeg` ↔ `img_0.png`).
- 마스크는 BGR 색상으로 클래스를 인코딩합니다 (`datahandler.py`):

  | BGR 색상      | 클래스 인덱스 | 라벨   |
  |---------------|---------------|--------|
  | `(0, 0, 0)`   | 0             | Good   |
  | `(0, 0, 128)` | 1             | Fair   |
  | `(0, 128, 0)` | 2             | Poor   |
  | `(0, 128, 128)`| 3            | Severe |

  > 마스크는 무손실 PNG여야 합니다. JPEG처럼 압축 색번짐이 생기면, 매핑에 없는 색상은 조용히 `Good(0)`으로 처리되어 부식이 누락될 수 있습니다.

- 이진 모드(`--num_classes 2`)에서는 1·2·3 → `Corrosion(1)`로 자동 통합됩니다.

---

## 환경 설정

```bash
conda create -n segformer python=3.10
conda activate segformer

# PyTorch (CUDA 버전에 맞게)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

pip install -r requirements.txt
```

검증된 버전: PyTorch 2.12.1 (+cu132), transformers 5.12.1, albumentations 2.0.8, OpenCV 4.13, scikit-learn 1.7.

---

## 학습

```bash
cd training
conda activate segformer
```

### 4-class

```bash
python main.py \
  --data_dir   /path/to/unified_corrosion \
  --exp_dir    ./stored_weights/segformer_b2_4class \
  --model_name nvidia/mit-b2 \
  --num_classes 4 \
  --epochs 40 --batch_size 4 --lr 6e-5 \
  --class_weights 0.1 0.3 0.3 0.3
```

### 이진 (Good / Corrosion)

```bash
python main.py \
  --data_dir   /path/to/unified_corrosion \
  --exp_dir    ./stored_weights/segformer_b2_binary \
  --model_name nvidia/mit-b2 \
  --num_classes 2 \
  --epochs 40 --batch_size 4 --lr 6e-5 \
  --class_weights 0.1 0.5
```

### 주요 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--data_dir` | `Train/`, `Test/`를 포함한 데이터 루트 | (필수) |
| `--exp_dir` | 체크포인트·로그 저장 경로 | (필수) |
| `--model_name` | HuggingFace 백본 | `nvidia/mit-b2` |
| `--num_classes` | `4` 또는 `2` | `4` |
| `--epochs` | 학습 에폭 | `40` |
| `--batch_size` | 배치 크기 | `4` |
| `--lr` | AdamW 학습률 | `6e-5` |
| `--class_weights` | 클래스별 손실 가중치 (**개수 = num_classes**) | `None` |

> `--class_weights` 개수가 `--num_classes`와 다르면 학습 시작 전에 에러로 알려줍니다.
> 클래스 불균형(픽셀 기준 Good ≈ 73%, Severe ≈ 2%)이 크므로 배경 클래스 가중치를 낮춰 시작하길 권합니다.

---

## 출력물

`--exp_dir` 아래에 생성됩니다:

- `log.csv` — 에폭별 Train/Test의 loss·F1·IoU
- `weights_<epoch>.pt` — Test F1 갱신 시점의 체크포인트
- `weights_best.pt` — 최종 best 모델

체크포인트는 **state_dict** 형태로 저장됩니다(모델 객체 통째 저장 시 transformers hook 직렬화 오류 방지).

### 추론 시 로드 예시

```python
import torch
from transformers import SegformerForSemanticSegmentation

id2label = {0: 'Good', 1: 'Fair', 2: 'Poor', 3: 'Severe'}   # 이진: {0:'Good', 1:'Corrosion'}
model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/mit-b2',
    num_labels=len(id2label),
    id2label=id2label,
    label2id={v: k for k, v in id2label.items()},
    ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load('stored_weights/segformer_b2_4class/weights_best.pt'))
model.eval()
```

---

## 구현 메모

- 입력은 `SegformerImageProcessor`로 ImageNet 정규화(`do_resize=False`, 이미 512×512).
- 학습 시 증강: 수평 뒤집기 + 소폭 Affine(회전/이동/전단/스케일).
- SegFormer logits는 입력의 1/4 해상도이므로, 손실 계산 전에 `F.interpolate`(bilinear)로 라벨 해상도까지 업샘플.
- 평가 지표는 픽셀 단위 **weighted** F1·IoU. Good 클래스 비중이 커서 전체 점수가 높게 나오는 경향이 있으니, 부식 클래스 성능은 클래스별로 따로 보는 것을 권장합니다.
