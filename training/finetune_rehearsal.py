"""
파인튜닝 리허설: CCSD+CIR 베이스라인을 'Roboflow를 가짜 실증 데이터로' 소량 파인튜닝하여
(1) 적응(새 도메인 성능 향상), (2) forgetting(기존 도메인 회귀), (3) replay 효과를 측정.

베이스라인:  segformer_b2_binary/weights_best.pt (CCSD+CIR만 학습)
field pool:  Roboflow train(440, rf_ 변환본)  → subset 25/50/100 사용
replay pool: CCSD+CIR train (4색 마스크, num_classes=2가 이진 통합)
평가셋:      in-domain = unified_corrosion/Test(56), roboflow = corrosion_levell_roboflow_testsplit/Test(44)
"""
import os, glob, random, csv, shutil
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datahandler import CorrosionDataset

random.seed(0); torch.manual_seed(0)
DEV = 'cuda:0'
HOME = '/home/ldh/minkyung'
BASE_CKPT = './stored_weights/segformer_b2_binary/weights_best.pt'
PROC = SegformerImageProcessor(do_resize=False, do_normalize=True)
TMP = '/tmp/claude-1000/-home-ldh-minkyung-corrosion-segformer/3e0ea3a8-3217-4bdb-8c43-228caa17cd44/scratchpad/rehearsal'
OUT_CSV = './stored_weights/segformer_b2_binary/finetune_rehearsal.csv'

FIELD_IMG_DIR = f'{HOME}/unified_corrosion_plus_roboflow/Train/Images'   # rf_* 들
FIELD_MSK_DIR = f'{HOME}/unified_corrosion_plus_roboflow/Train/Masks'
BASE_IMG_DIR  = f'{HOME}/unified_corrosion/Train/Images'
BASE_MSK_DIR  = f'{HOME}/unified_corrosion/Train/Masks'
TEST_IN  = f'{HOME}/unified_corrosion/Test'
TEST_RF  = f'{HOME}/corrosion_levell_roboflow_testsplit/Test'

EPOCHS, LR, BS = 15, 1e-5, 4
WEIGHT = torch.tensor([0.1, 0.5]).to(DEV)


def pairs_from(img_dir, msk_dir, prefix=None):
    out = []
    for img in sorted(glob.glob(os.path.join(img_dir, '*'))):
        b = os.path.basename(img)
        if prefix and not b.startswith(prefix):
            continue
        stem = os.path.splitext(b)[0]
        cand = glob.glob(os.path.join(msk_dir, stem + '.*'))
        if cand:
            out.append((img, cand[0]))
    return out


FIELD = pairs_from(FIELD_IMG_DIR, FIELD_MSK_DIR, prefix='rf_')   # 440
BASE  = pairs_from(BASE_IMG_DIR, BASE_MSK_DIR)                    # 898
random.shuffle(BASE)


def build_folder(pairs, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, 'Images'))
    os.makedirs(os.path.join(dst, 'Masks'))
    for img, msk in pairs:
        os.symlink(os.path.abspath(img), os.path.join(dst, 'Images', os.path.basename(img)))
        os.symlink(os.path.abspath(msk), os.path.join(dst, 'Masks', os.path.basename(msk)))
    return dst


def load_baseline():
    m = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b2', num_labels=2,
        id2label={0: 'Good', 1: 'Corrosion'}, label2id={'Good': 0, 'Corrosion': 1},
        ignore_mismatched_sizes=True)
    m.load_state_dict(torch.load(BASE_CKPT, map_location='cpu'))
    return m.to(DEV)


def finetune(train_dir):
    model = load_baseline()
    ds = CorrosionDataset(train_dir, PROC, augment=True, num_classes=2)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    for ep in range(EPOCHS):
        for batch in dl:
            pv = batch['pixel_values'].to(DEV); lb = batch['labels'].to(DEV)
            opt.zero_grad()
            logits = model(pixel_values=pv).logits
            up = F.interpolate(logits, size=lb.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.cross_entropy(up, lb, weight=WEIGHT)
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, test_dir):
    model.eval()
    ds = CorrosionDataset(test_dir, PROC, augment=False, num_classes=2)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    cm = np.zeros((2, 2), dtype=np.int64)
    for batch in dl:
        pv = batch['pixel_values'].to(DEV); lb = batch['labels'].to(DEV)
        up = F.interpolate(model(pixel_values=pv).logits, size=lb.shape[-2:],
                           mode='bilinear', align_corners=False)
        pred = up.argmax(1).view(-1).cpu().numpy(); true = lb.view(-1).cpu().numpy()
        cm += np.bincount(true * 2 + pred, minlength=4).reshape(2, 2)
    tp = np.diag(cm).astype(float); sup = cm.sum(1).astype(float); ps = cm.sum(0).astype(float)
    fp = ps - tp; fn = sup - tp; eps = 1e-12
    f1 = 2 * tp / (2 * tp + fp + fn + eps); iou = tp / (tp + fp + fn + eps)
    rec = tp / (tp + fn + eps)
    w = sup / sup.sum()
    return {'wF1': float((f1 * w).sum()), 'corr_IoU': float(iou[1]),
            'corr_Rec': float(rec[1])}


configs = [(s, r) for s in (25, 50, 100) for r in (0, 100)]
rows = []
print(f"베이스라인 평가 (파인튜닝 전)")
base = load_baseline()
b_in = evaluate(base, TEST_IN); b_rf = evaluate(base, TEST_RF)
print(f"  baseline | in:{b_in['wF1']:.4f}/{b_in['corr_IoU']:.4f}  rf:{b_rf['wF1']:.4f}/{b_rf['corr_IoU']:.4f}")
rows.append(['baseline(no-ft)', 0, 0, b_in['wF1'], b_in['corr_IoU'], b_rf['wF1'], b_rf['corr_IoU'], b_rf['corr_Rec']])
del base; torch.cuda.empty_cache()

for s, r in configs:
    field = FIELD[:s]
    rep = BASE[:r]
    train_dir = build_folder(field + rep, os.path.join(TMP, f's{s}_r{r}'))
    model = finetune(train_dir)
    ein = evaluate(model, TEST_IN); erf = evaluate(model, TEST_RF)
    print(f"  subset={s} replay={r} | in:{ein['wF1']:.4f}/{ein['corr_IoU']:.4f}  "
          f"rf:{erf['wF1']:.4f}/{erf['corr_IoU']:.4f} (rec {erf['corr_Rec']:.4f})")
    rows.append([f's{s}_r{r}', s, r, ein['wF1'], ein['corr_IoU'], erf['wF1'], erf['corr_IoU'], erf['corr_Rec']])
    del model; torch.cuda.empty_cache()

with open(OUT_CSV, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['config', 'field_n', 'replay_n', 'indomain_wF1', 'indomain_corrIoU',
                 'roboflow_wF1', 'roboflow_corrIoU', 'roboflow_corrRecall'])
    wr.writerows(rows)
print(f"\n저장: {OUT_CSV}")
print("REHEARSAL_DONE")
