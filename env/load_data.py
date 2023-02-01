from dataclasses import dataclass, field
import pandas as pd
from typing import List, Tuple
import json

@dataclass
class AgentList:
    task_type : List = field(default_factory=list)
    cpu_req : List = field(default_factory=list)
    cpu_limit : List = field(default_factory=list)
    mem_req : List = field(default_factory=list)
    mem_limit : List = field(default_factory=list)
    task_duration: List = field(default_factory=list)

    def to_list(self) -> List:
        return [self.task_type, self.cpu_req, self.cpu_limit, self.mem_req, self.mem_limit, self.task_duration]

def get_all_task(path: str) -> List:

    agent_list = AgentList()
    data = pd.read_csv(path, sep=";")
    for _, row in enumerate(data.values):
        task_data = json.loads(row[1])
        if task_data['task_type'] == 'Pod':
            agent_list.task_type.append("Deployment")
            agent_list.task_duration.append(0)
        else:
            agent_list.task_type.append(task_data['task_type'])
            agent_list.task_duration.append(task_data['duration'])

        agent_list.cpu_limit.append(task_data['cpu_lim'])
        agent_list.cpu_req.append(task_data['cpu_req'])
        agent_list.mem_limit.append(task_data['mem_lim'])
        agent_list.mem_req.append(task_data['mem_req'])

    return agent_list

if __name__ == '__main__':
    ret = get_all_task('data.json')
    f = open('data.list', 'a')
    f.write(str(ret))
    f.close()

    print(ret)
