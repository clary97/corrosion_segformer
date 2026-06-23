"""
SegFormer 부식 세그멘테이션 학습 스크립트

Usage:
    python main.py \
        --data_dir   /home/ldh/minkyung/unified_corrosion \
        --exp_dir    ./stored_weights/segformer_b2_4class \
        --model_name nvidia/mit-b2 \
        --num_classes 4 \
        --epochs 40 \
        --batch_size 4 \
        --lr 6e-5 \
        --class_weights 0.1 0.3 0.3 0.3
"""
import argparse
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datahandler import get_dataloaders
from trainer import train_model

ID2LABEL_4 = {0: 'Good', 1: 'Fair', 2: 'Poor',  3: 'Severe'}
ID2LABEL_2 = {0: 'Good', 1: 'Corrosion'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     required=True)
    parser.add_argument('--exp_dir',      required=True)
    parser.add_argument('--model_name',   default='nvidia/mit-b2')
    parser.add_argument('--num_classes',  type=int,   default=4)
    parser.add_argument('--epochs',       type=int,   default=40)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--lr',           type=float, default=6e-5)
    parser.add_argument('--class_weights', nargs='+', type=float, default=None)
    args = parser.parse_args()

    id2label = ID2LABEL_4 if args.num_classes == 4 else ID2LABEL_2
    label2id = {v: k for k, v in id2label.items()}

    print(f'Model  : {args.model_name}')
    print(f'Classes: {args.num_classes} → {list(id2label.values())}')
    print(f'Data   : {args.data_dir}')
    print(f'ExpDir : {args.exp_dir}')

    # ── 이미지 프로세서 & 모델 로드 ──────────────────────────────────────
    processor = SegformerImageProcessor(
        do_resize=False,       # 이미 512×512
        do_normalize=True,     # ImageNet 정규화
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── 데이터로더 ────────────────────────────────────────────────────────
    dataloaders = get_dataloaders(args.data_dir, processor, args.batch_size)

    # ── 옵티마이저 ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)

    # ── 학습 ─────────────────────────────────────────────────────────────
    train_model(
        model       = model,
        dataloaders = dataloaders,
        optimizer   = optimizer,
        bpath       = args.exp_dir,
        class_weights = args.class_weights,
        num_epochs  = args.epochs,
        num_classes = args.num_classes,
    )


if __name__ == '__main__':
    main()
