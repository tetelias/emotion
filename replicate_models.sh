python src/train.py --train-model --evaluate-devset --base-model rubert --epochs  35 --scheduler cosine --dataset combo --create-onnx
python src/train.py --train-model --evaluate-devset --base-model rubert --epochs 100 --scheduler cosine --dataset cedr
python src/train.py --train-model --evaluate-devset --base-model xlm    --epochs  16 --scheduler cosine --dataset combo --create-onnx
