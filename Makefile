.DEFAULT_GOAL := all_local
.PHONY: build push pull pull_local run run_local clean all

server = harbor.dev-hs.de
project = crcl_os
image_name = rl_agent
username = tp100@hsg.privat

build:
	docker build -t ${server}/${project}/${image_name} .
	
push:
	docker push ${server}/${project}/${image_name}
	
pull_local:
	docker pull ${server}/${project}/${image_name}
	
pull:
	ssh ${username}@10.122.193.58 docker pull ${server}/${project}/${image_name}
	
run_local:
	docker run \
	-e PYTHONUNBUFFERED=1 \
	--mount type=bind,src=/home/ki-user/mountdir/gdrive/06\ \ Projekte/72\ KI/31172009\ CRLC_OS/10\ Experimentelle\ Daten/agent_training_runs,target=/home/results/,readonly=0 \
	${server}/${project}/${image_name} simplified/masked_ppo.py \
	--num_episodes 10 \
	--num_nodes 32 \
	--num_envs 32 \
	--policy_arch 128 128 \
	--lr 0.00003 \
	--entropy_c 0.01 \
	--output_path /home/results \
	--verbose 1 \
	# --model_path bla \
	
run:
	ssh ${username}@10.122.193.58 \
	docker run \
	-e PYTHONUNBUFFERED=1 \
	--cpus=8 \
	--mount type=bind,src="/home/tp100@hsg.privat/g/06\ \ Projekte/72\ KI/31172009\ CRLC_OS/10\ Experimentelle\ Daten/agent_training_runs",target=/home/results/,readonly=0 \
	${server}/${project}/${image_name} simplified/masked_ppo.py \
	--num_episodes 10 \
	--num_nodes 32 \
	--num_envs 32 \
	--policy_arch 128 128 \
	--lr 0.00003 \
	--entropy_c 0.01 \
	--output_path /home/results \
	--verbose 1 \
	#--gpus=3 \
	

	
all: build push pull run

all_local: build push pull_local run_local

clean:
	docker image prune
	docker container prune
	
