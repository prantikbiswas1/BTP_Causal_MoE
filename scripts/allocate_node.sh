salloc --nodes=1 --time=12:00:00 --partition=gpu --nodelist=ragpu004

squeue --me

scancel id

ssh ragpu004

tmux new -s moe_train
bash 09_train_moe.sh
Ctrl + B, then D
tmux attach -t moe_train
tmux ls

torchrun --nproc_per_node=2 /home/rupeshk_iitp/Prantik/BTP/BTP_Causal_MoE/src/training/train_peft_moe.py

https://gemini.google.com/share/0bb1cd43f499