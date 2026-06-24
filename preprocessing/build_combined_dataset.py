"""
범용 이진 부식 모델용 통합 데이터셋 구성 (데이터 누수 방지: Roboflow의 train/test 분할 유지).

생성물:
  unified_corrosion_plus_roboflow/        # 학습용 통합 데이터셋
    Train/  = CCSD+CIR train(898, symlink) + Roboflow train(440, 변환)
    Test/   = CCSD+CIR test(56, symlink)   # in-domain 회귀 확인용
  corrosion_levell_roboflow_testsplit/    # Roboflow 도메인 평가용
    Test/   = Roboflow test(44, 변환)

Roboflow 변환: 이미지 512 리사이즈(INTER_AREA), 마스크 512(INTER_NEAREST),
corrosion(>0) → BGR(0,0,128), bg → (0,0,0)  ← 우리 num_classes=2 파이프라인과 호환.
"""
import glob, os, cv2, numpy as np

HOME      = "/home/ldh/minkyung"
UNIFIED   = os.path.join(HOME, "unified_corrosion")          # 4-class BGR, 512
ROBOFLOW  = os.path.join(HOME, "corrosion_levell_roboflow")  # 0/1 인덱스, 가변크기
COMBINED  = os.path.join(HOME, "unified_corrosion_plus_roboflow")
RF_TEST   = os.path.join(HOME, "corrosion_levell_roboflow_testsplit")
SIZE = 512
CORR_BGR = (0, 0, 128)


def mkdirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def symlink_split(src_split, dst_img_dir, dst_msk_dir):
    """기존 unified 데이터(이미 우리 포맷)를 심볼릭 링크로 연결."""
    n = 0
    for img in glob.glob(os.path.join(src_split, "Images", "*")):
        dst = os.path.join(dst_img_dir, os.path.basename(img))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(img), dst)
        n += 1
    for msk in glob.glob(os.path.join(src_split, "Masks", "*")):
        dst = os.path.join(dst_msk_dir, os.path.basename(msk))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(msk), dst)
    return n


def convert_roboflow_split(rf_split, dst_img_dir, dst_msk_dir):
    """Roboflow split(train/test)을 우리 포맷(512, BGR 마스크)으로 변환."""
    n = 0
    for img_path in sorted(glob.glob(os.path.join(ROBOFLOW, rf_split, "*.jpg"))):
        stem = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(ROBOFLOW, rf_split, stem + "_mask.png")
        if not os.path.exists(mask_path):
            continue
        name = "rf_" + stem  # 기존 파일명과 충돌 방지

        img = cv2.imread(img_path)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m.ndim == 3:
            m = m[..., 0]
        m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        bgr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        bgr[m > 0] = CORR_BGR

        cv2.imwrite(os.path.join(dst_img_dir, name + ".jpg"), img)
        cv2.imwrite(os.path.join(dst_msk_dir, name + ".png"), bgr)
        n += 1
    return n


# ── 통합 학습 데이터셋 ────────────────────────────────────────────────────
c_tr_img = os.path.join(COMBINED, "Train", "Images")
c_tr_msk = os.path.join(COMBINED, "Train", "Masks")
c_te_img = os.path.join(COMBINED, "Test", "Images")
c_te_msk = os.path.join(COMBINED, "Test", "Masks")
mkdirs(c_tr_img, c_tr_msk, c_te_img, c_te_msk)

n_unified_tr = symlink_split(os.path.join(UNIFIED, "Train"), c_tr_img, c_tr_msk)
n_unified_te = symlink_split(os.path.join(UNIFIED, "Test"),  c_te_img, c_te_msk)
n_rf_tr      = convert_roboflow_split("train", c_tr_img, c_tr_msk)

# ── Roboflow 평가셋 (test split) ──────────────────────────────────────────
rf_te_img = os.path.join(RF_TEST, "Test", "Images")
rf_te_msk = os.path.join(RF_TEST, "Test", "Masks")
mkdirs(rf_te_img, rf_te_msk)
n_rf_te = convert_roboflow_split("test", rf_te_img, rf_te_msk)

print("===== 구성 완료 =====")
print(f"통합 학습셋 ({COMBINED}):")
print(f"  Train = unified {n_unified_tr} + roboflow {n_rf_tr} = {n_unified_tr + n_rf_tr}")
print(f"  Test  = unified {n_unified_te} (in-domain 회귀 확인용)")
print(f"Roboflow 평가셋 ({RF_TEST}):")
print(f"  Test  = roboflow {n_rf_te}")
