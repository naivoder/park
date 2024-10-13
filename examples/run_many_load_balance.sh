mkdir ../../data/logs

env_name=load_balance
algo=a2c
gpu=2
n_steps=10000
total_timesteps=100000000
for distribution in Pareto # Uniform
do
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u run_agent_sb3.py --env_name=${env_name} --algo=${algo} --distribution=${distribution} --n_steps=${n_steps} --total_timesteps=${total_timesteps} > ../../data/logs/${algo}_sb3_${env_name}_${distribution}_${n_steps}.log &
    gpu=$((gpu+1))
done 
