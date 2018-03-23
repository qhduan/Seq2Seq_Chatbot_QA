::python s2s.py --size 512 --num_layers 1 --num_epoch 20 --batch_size 64 --num_per_epoch 1000000 --test false --bleu -1 --model_dir ./model/model3
@echo off

set arg1=%1
set arg2=%2

if "%arg1%" == "" goto train
if "%arg1%" == "test" goto test
if "%arg1%" == "bleu" goto bleu

:test
echo ------- test -------
python s2s.py --test true --size 512 --num_layers 1 --num_epoch 20 --batch_size 64 --num_per_epoch 1000000 --bleu -1 --model_dir ./model/model3
goto end

:bleu
echo ------- bleu -------
if "%arg2%" == "" ( echo bleu sample is NULL! ) else ( python s2s.py --bleu %arg2% --size 512 --num_layers 1 --num_epoch 20 --batch_size 64 --num_per_epoch 1000000 --test false --model_dir ./model/model3 )
goto end

:train
echo ------- train -------
python s2s.py --size 512 --num_layers 1 --num_epoch 20 --batch_size 64 --num_per_epoch 1000000 --test false --bleu -1 --model_dir ./model/model3
goto end

:end
echo ------- end -------
