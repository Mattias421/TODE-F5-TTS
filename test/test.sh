accelerate launch --debug --mixed_precision=fp16 ../src/f5_tts/train/train.py --config-name F5TTS_Small_train.yaml --datasets.max_samples=8 --optim.epochs=2 --num_warmup_updates=20 
