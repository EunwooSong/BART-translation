# BART-translation
BART-base 이용한 en-kr translator 

## Reference
- [KoBART-translation](https://github.com/seujung/KoBART-translation) 의 파일을 수정해서 활용

## Requirements
- python 3.11 사용
- cuda 11.8 사용
```
pytorch==2.0.1+cu118
transformers==4.31.0
pytorch-lightning==1.1.0
```

## How to Train
```
prepare.sh
python train.py  --gradient_clip_val 1.0 --max_epochs 50 --default_root_dir logs  --gpus 1 --batch_size 4
```