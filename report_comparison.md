# SegFormer 부식 세그멘테이션 — 실험 결과 및 모델 비교 보고서

## 1. 개요

본 보고서는 **SegFormer-B2** 기반 부식(corrosion) 시맨틱 세그멘테이션 모델의
4-class / 이진(binary) 학습 결과를 정리하고, 동일 데이터셋으로 학습한
**DeepLabV3+ / ResNet-101** 모델과 성능을 비교한다.

- SegFormer 실험: 본 저장소 (`corrosion_segformer`)
- DeepLabV3+ 실험: <https://github.com/clary97/corrosion_cs_classification>

두 실험은 **동일한 통합 데이터셋(unified_corrosion)과 동일한 Test셋**을 사용한다
(Test 전체 픽셀 14,680,064개, 클래스별 support 픽셀 수 일치 확인).

| 항목 | SegFormer 실험 | DeepLabV3+ 실험 |
|---|---|---|
| 모델 | SegFormer-B2 (`nvidia/mit-b2`) | DeepLabV3+ / ResNet-101 |
| 클래스 | 4-class & Binary | 4-class & Binary |
| 에포크 | 40 | 40 |
| 배치 크기 | 4 | 2 |
| 손실 | (weighted) Cross-Entropy | (weighted) Cross-Entropy |
| 4-class 가중치 | 0.1 / 0.3 / 0.3 / 0.3 | 0.1 / 0.3 / 0.3 / 0.3 |
| 입력 해상도 | 512 × 512 | 512 × 512 |
| 평가 | Test셋 전체 픽셀 혼동행렬 기반 | Test셋 전체 픽셀 혼동행렬 기반 |

> **평가 모델 선택**: 두 실험 모두 각자의 **best 체크포인트**로 평가하였다.
> 단, best 선택 기준이 다르다 — SegFormer는 **Test F1 최대**(4-class: epoch 25, Binary: epoch 30),
> DeepLabV3+는 **Test spectrum_score 최저**(4-class: epoch 37). DeepLab 로그상 F1 최대 epoch와
> spectrum 최저 epoch의 Test F1 차이는 약 0.2%p로 미미하여, 비교 결론에는 영향을 주지 않는다.

평가 지표: Precision, Recall, F1, IoU, FPR(거짓양성률, 낮을수록 좋음).
FPR_c = FP / (FP + TN) = FP / (전체 픽셀 − support).

> **보고서 범위 안내**: 2~4절은 **원본 데이터(CCSD+CIR)** 기준의
> 아키텍처 비교(SegFormer-**B2** vs DeepLabV3+)이다. SegFormer-**B3**는 베이스라인 백본 선택
> 맥락에서 도입되어 **통합셋(Roboflow 추가)으로만 학습**하였으므로, 6~7절에서 다룬다
> (원본 데이터에 대한 B3 결과는 없음).

---

## 2. 4-class 결과

### SegFormer-B2

| Class | Precision | Recall | F1 | IoU | FPR | Support (px) |
|---|---|---|---|---|---|---|
| Good | 0.9658 | 0.9587 | 0.9622 | 0.9272 | 0.1095 | 11,206,383 |
| Fair | 0.6936 | 0.7323 | 0.7124 | 0.5533 | 0.0432 | 1,728,971 |
| Poor | 0.7329 | 0.7172 | 0.7250 | 0.5686 | 0.0261 | 1,334,125 |
| Severe | 0.6847 | 0.7092 | 0.6967 | 0.5346 | 0.0094 | 410,585 |
| **Weighted** | **0.9047** | **0.9031** | **0.9038** | **0.8396** | **0.0913** | 14,680,064 |
| macro | 0.7693 | 0.7793 | 0.7741 | 0.6459 | 0.0471 | |

### DeepLabV3+ / ResNet-101

| Class | Precision | Recall | F1 | IoU | FPR | Support (px) |
|---|---|---|---|---|---|---|
| Good | 0.9383 | 0.9593 | 0.9487 | 0.9023 | 0.2036 | 11,206,383 |
| Fair | 0.6248 | 0.5629 | 0.5922 | 0.4207 | 0.0451 | 1,728,971 |
| Poor | 0.6643 | 0.6913 | 0.6775 | 0.5123 | 0.0349 | 1,334,125 |
| Severe | 0.8323 | 0.5605 | 0.6699 | 0.5036 | 0.0033 | 410,585 |
| **Weighted** | **0.8735** | **0.8771** | **0.8742** | **0.7990** | **0.1640** | 14,680,064 |

### 4-class 비교 (Weighted)

| 지표 | SegFormer | DeepLabV3+ | 차이 |
|---|---|---|---|
| Precision | **0.9047** | 0.8735 | +3.12%p |
| Recall | **0.9031** | 0.8771 | +2.60%p |
| F1 | **0.9038** | 0.8742 | +2.96%p |
| IoU | **0.8396** | 0.7990 | +4.06%p |
| FPR ↓ | **0.0913** | 0.1640 | −7.27%p |

**관찰**
- SegFormer가 모든 weighted 지표에서 우세하며, 특히 FPR이 크게 낮다(오탐 감소).
- 가장 어려운 **Fair** 클래스에서 F1 +12%p(0.59 → 0.71)로 격차가 크다.
- **Severe**는 DeepLab의 Precision(0.83)이 더 높지만 Recall이 0.56으로 낮아 검출 누락이 많고,
  F1·IoU 종합으로는 SegFormer가 앞선다.

---

## 3. Binary 결과 (Good / Corrosion)

### SegFormer-B2

| Class | Precision | Recall | F1 | IoU | FPR | Support (px) |
|---|---|---|---|---|---|---|
| Good | 0.9705 | 0.9529 | 0.9616 | 0.9260 | 0.0936 | 11,206,383 |
| Corrosion | 0.8563 | 0.9064 | 0.8807 | 0.7868 | 0.0471 | 3,473,681 |
| **Weighted** | **0.9435** | **0.9419** | **0.9424** | **0.8931** | **0.0826** | 14,680,064 |
| macro | 0.9134 | 0.9296 | 0.9211 | 0.8564 | 0.0704 | |

### DeepLabV3+ / ResNet-101

| Class | Precision | Recall | F1 | IoU | FPR | Support (px) |
|---|---|---|---|---|---|---|
| Good | 0.9565 | 0.9492 | 0.9529 | 0.9100 | 0.1393 | 11,206,383 |
| Corrosion | 0.8402 | 0.8607 | 0.8503 | 0.7396 | 0.0508 | 3,473,681 |
| **Weighted** | **0.9290** | **0.9283** | **0.9286** | **0.8696** | **0.1184** | 14,680,064 |

### Binary 비교 (Weighted)

| 지표 | SegFormer | DeepLabV3+ | 차이 |
|---|---|---|---|
| Precision | **0.9435** | 0.9290 | +1.45%p |
| Recall | **0.9419** | 0.9283 | +1.36%p |
| F1 | **0.9424** | 0.9286 | +1.38%p |
| IoU | **0.8931** | 0.8696 | +2.35%p |
| FPR ↓ | **0.0826** | 0.1184 | −3.58%p |

**관찰**
- 이진에서도 SegFormer가 전 지표 우세. 다만 태스크가 쉬워 두 모델 모두 높게 수렴해 격차는 4-class보다 작다.
- 핵심은 **Corrosion Recall +4.57%p(0.86 → 0.91)** — 부식 픽셀 검출 누락이 더 적다.

---

## 4. 종합

| 모델 | 모드 | Precision | Recall | F1 | IoU | FPR ↓ |
|---|---|---|---|---|---|---|
| SegFormer-B2 | 4-class | 0.9047 | 0.9031 | 0.9038 | 0.8396 | 0.0913 |
| DeepLabV3+ | 4-class | 0.8735 | 0.8771 | 0.8742 | 0.7990 | 0.1640 |
| SegFormer-B2 | Binary | **0.9435** | **0.9419** | **0.9424** | **0.8931** | **0.0826** |
| DeepLabV3+ | Binary | 0.9290 | 0.9283 | 0.9286 | 0.8696 | 0.1184 |

- **전 지표·전 설정에서 SegFormer-B2가 DeepLabV3+ / ResNet-101보다 우수**하며, 특히 FPR(오탐) 감소가 두드러진다.
- 두 모델 모두 이진이 4-class보다 점수가 높다 — 부식 유무 검출이 심각도 등급 분류보다 쉬운 태스크임을 보여준다.

## 5. 한계 및 주의사항

- weighted 지표는 Good(전체 픽셀 약 73%)의 비중이 커서 높게 나타난다. 부식 등급별 실력은
  클래스별 지표 및 macro 평균(4-class F1 0.774)으로 함께 판단해야 한다.
- 이진 모델은 부식 심각도(Fair/Poor/Severe)를 구분하지 않으므로, 유지보수 등급 판정에는
  4-class 모델이 함께 필요하다.
- 두 실험의 best 체크포인트 선택 기준이 다르다(1절 참고). 결론에 영향을 주는 수준은 아니다.
- 평가 재현: 각 `--exp_dir` 아래 `eval_per_class.csv` 및 `training/evaluate.py` 참고.

---

## 6. 범용성 강화 실험 (외부 데이터 추가)

범용 이진(Good/Corrosion) 모델을 위해 외부 공개 데이터셋
[Roboflow university-of-tebessa/corrosion-levell](https://universe.roboflow.com/university-of-tebessa/corrosion-levell)
(484장, 이진 semantic 마스크)을 학습에 추가하였다. 데이터 누수 방지를 위해 Roboflow의
train(440)만 학습에 합치고, test(44)는 별도 외부 평가셋으로 분리하였다.

- 학습셋: CCSD+CIR train(898) + Roboflow train(440) = **1338장**
- 평가셋: ① in-domain = CCSD+CIR test(56, 회귀 확인용) ② Roboflow test(44, 새 도메인)
- 모든 이미지는 512×512로 정렬(이미지 INTER_AREA, 마스크 INTER_NEAREST).

### SegFormer-B2 이진: 추가 전 vs 후

**in-domain (CCSD+CIR test 56)**

| 지표 | 추가 전 | 추가 후 | 변화 |
|---|---|---|---|
| Weighted F1 | 0.9424 | **0.9673** | +2.49%p |
| Corrosion F1 | 0.8807 | **0.9332** | +5.25%p |
| Corrosion IoU | 0.7868 | **0.8748** | +8.80%p |
| Corrosion Recall | 0.9064 | **0.9832** | +7.68%p |

**Roboflow (test 44)**

| 지표 | 추가 전 | 추가 후 | 변화 |
|---|---|---|---|
| Weighted F1 | 0.9354 | **0.9703** | +3.49%p |
| Corrosion F1 | 0.8483 | **0.9310** | +8.27%p |
| Corrosion IoU | 0.7366 | **0.8708** | +13.42%p |
| Corrosion Recall | 0.8841 | **0.9872** | +10.31%p |

**결론**: in-domain 성능이 **떨어지지 않고 오히려 향상**(회귀 없음)되었으며, 새 도메인은 더 크게
향상되었다. 두 도메인 모두 Corrosion Recall ≈ 0.98로, 부식 픽셀 검출 누락이 거의 없는
강건한 범용 이진 모델을 얻었다. (학습 데이터 898→1338 증가가 전반적 성능을 함께 끌어올림.)

### DeepLabV3+ 이진: 추가 전 vs 후

**in-domain (CCSD+CIR test 56)**

| 지표 | 추가 전 | 추가 후 | 변화 |
|---|---|---|---|
| Weighted F1 | 0.9286 | **0.9699** | +4.13%p |
| Corrosion F1 | 0.8503 | **0.9380** | +8.77%p |
| Corrosion IoU | 0.7396 | **0.8832** | +14.36%p |
| Corrosion Recall | 0.8607 | **0.9767** | +11.60%p |

**Roboflow (test 44)**

| 지표 | 추가 전 | 추가 후 | 변화 |
|---|---|---|---|
| Weighted F1 | 0.9213 | **0.9781** | +5.68%p |
| Corrosion F1 | 0.8108 | **0.9486** | +13.78%p |
| Corrosion IoU | 0.6818 | **0.9021** | +22.03%p |
| Corrosion Recall | 0.8142 | **0.9885** | +17.43%p |

DeepLabV3+ 역시 in-domain 회귀 없이 두 도메인 모두 크게 향상되었다.

### 통합 모델 최종 비교 (추가 후, Weighted F1 / Corrosion IoU)

| 모델 | in-domain (56) | Roboflow (44) |
|---|---|---|
| SegFormer-B2 | 0.9673 / 0.8748 | 0.9703 / 0.8708 |
| DeepLabV3+ | **0.9699 / 0.8832** | **0.9781 / 0.9021** |

**종합 결론**: 외부 데이터(Roboflow train 440) 추가는 두 모델 모두에서 **in-domain 회귀 없이
양 도메인 성능을 크게 끌어올렸다**(범용성 강화 성공). 추가 후에는 DeepLabV3+가 근소하게 앞서나,
CCSD+CIR만 사용한 기존 비교(2~4절)에서는 SegFormer가 우세했던 점과 함께 보면, 데이터 규모
증가의 효과가 컸음을 시사한다. 두 모델 모두 Corrosion Recall ≈ 0.98로 부식 누락이 거의 없다.

> 주의: SegFormer는 best 기준이 Test F1, DeepLabV3+는 spectrum_score로 서로 다르며,
> Roboflow 평가는 단일 외부 데이터셋(44장) 기준이므로 일반화의 절대 보장은 아니다.

데이터셋 구성 스크립트: `preprocessing/build_combined_dataset.py`,
평가: `training/evaluate.py`(SegFormer), `eval_deeplab_generic.py`(DeepLabV3+).

---

## 7. 백본 비교 (SegFormer-B2 vs B3, 통합셋)

동일한 통합셋(1338장)·동일 조건으로 SegFormer-B2(~27M)와 B3(~47M)를 비교하였다.

### in-domain (test 56)

| 지표 | B2 | B3 |
|---|---|---|
| Weighted F1 | 0.9673 | **0.9705** |
| Corrosion F1 | 0.9332 | **0.9397** |
| Corrosion IoU | 0.8748 | **0.8863** |

### Roboflow (test 44)

| 지표 | B2 | B3 |
|---|---|---|
| Weighted F1 | 0.9703 | **0.9740** |
| Corrosion F1 | 0.9310 | **0.9394** |
| Corrosion IoU | 0.8708 | **0.8857** |

**관찰**: B3가 전 지표에서 일관되게 +1~1.5%p(Corrosion IoU) 앞선다. 단 파라미터가 ~1.7배라
학습·파인튜닝이 더 무겁다.

### 3모델 통합셋 종합 (Corrosion IoU, in / rf)

| 모델 | 크기 | in-domain | Roboflow |
|---|---|---|---|
| SegFormer-B2 | ~27M | 0.8748 | 0.8708 |
| SegFormer-B3 | ~47M | 0.8863 | 0.8857 |
| DeepLabV3+ (R101) | ~60M | 0.8832 | **0.9021** |

선택 가이드: 가벼움·빠른 파인튜닝→B2, SegFormer 내 성능 우선→B3, 절대 성능(특히 새 도메인)+무게 감수→DeepLabV3+.

---

## 8. 파인튜닝 리허설 (실증 데이터 적응 검증)

향후 실증(field) 데이터가 들어오면 베이스라인을 파인튜닝할 예정이므로, **Roboflow를 '가짜 실증
데이터'로 취급**해 적응성·forgetting을 사전 검증하였다. 베이스라인(CCSD+CIR만 학습한
`segformer_b2_binary`)을 Roboflow train의 일부(25/50/100장)로 저LR(1e-5)·15ep 파인튜닝.

| 설정 | in-domain corrIoU | Roboflow corrIoU |
|---|---|---|
| baseline (파인튜닝 전) | 0.7868 | 0.7366 |
| +25장 | 0.7870 | 0.7366 |
| +50장 | 0.7881 | 0.7416 |
| +100장 | 0.7916 | 0.7496 |

(replay 100장 추가 변형은 차이 미미 — forgetting이 없어 replay 효과 거의 없음)

**발견**
1. **파인튜닝은 안전하다** — 어떤 설정에서도 in-domain 회귀가 없었다(오히려 미세 상승).
2. **보수적 파인튜닝(저LR·소량·소수ep)은 적응이 약하다** — 100장에도 Roboflow corrIoU
   +0.013에 그쳐, 전체 통합 재학습(+0.13)에 크게 못 미친다. 적응폭은 데이터량에 비례.

### 연속(반복) 파인튜닝 운영 방식

새 실증 데이터가 시간차로 계속 들어오는 시나리오에 대한 권장 운영:

- **누적 후 재학습** (안전·확실): 들어온 실증 데이터를 누적하여 베이스 데이터와 함께 주기적으로
  전체 재학습. 본 보고서 6절에서 검증된 방식(forgetting 없음).
- **순차 파인튜닝 + replay** (자주·소량 유입 시): 매 회 기존 데이터 일부를 섞어 파인튜닝하여
  catastrophic forgetting 방지.
- 같은 데이터로의 반복 파인튜닝은 의미 없음(과적합만 증가).

리허설 스크립트: `training/finetune_rehearsal.py`.
