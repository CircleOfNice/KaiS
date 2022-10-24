from algorithm.GPG import OrchestrateAgent


episodes = 5000

for episode in episodes:
    - initialise q estimator(we have just one because we have one edge cluster) (CmmAC) and OrchestrateAgent 
    - Declaration of nodes, deploy_states, etc
    - Create Docker containers for cloud computing
    - for episode_length:
        if slot time% Orchestration_cycle ==0:
            - get state info e.g. (task info, task queue, deployment state etc)
            
            - get orchestration decision (Change of nodes, change of service, experience)
            # In this decision based on loss it is trained as well
            - Execute the orchestration action, (deploy nodes appropriately over two eAPs)
        else:
            - get current tasks in queue, cpu, memory etc for
            - generate a state based on these info
            - Get a dispatch decision via q_estimator
            - Put tasks on queue
            - update state of task and nodes 
            - Calculate tasks and rewards
            - assign current state as previous
            
        - Train q estimator policy and value net 