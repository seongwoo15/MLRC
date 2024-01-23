call conda activate tiemerge
cd D:\Source\patching
set count=1
set dataset_list=MNIST Cars DTD EuroSAT GTSRB RESISC45 SVHN
set model_list=ViT-L/14 ViT-B/32
(for %%a in (%model_list%) do (
	(for %%b in (%dataset_list%) do (
		set model_str=%%a
		set model_str=%model_str:\/=_%
		python src\patch.py --train-dataset=%%b --epochs=5 --lr=0.00001 --batch-size=4 --model=%%a --eval-datasets=%%b  --results-db=models/patch/%model_str%_%%b_%count%/results_%model_str%_%%b_%count%.jsonl --save=models/patch/%model_str%_%%b_%count% --data-location=data --warmup_length=200 --eval-every-epoch --wd=0.1 --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
	   
	))
))

conda deactivate

pause
ViT-B/32
MNIST Cars DTD EuroSAT GTSRB RESISC45 SUN397 SVHN