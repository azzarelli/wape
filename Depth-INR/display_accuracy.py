import copy

import matplotlib.pyplot as plt
import tensordict
import torch

fig, (axA) = plt.subplots(1,1)
axA.title.set_text(' Accuracy')

col = ['#1E5631', '#0700C4', '#910000','#FE036A', 
        '#A4DE02', '#0000FF', '#6B0000', '#F5347F', 
        '#76BA1B','#0052FF', '#D32431', '#FC72A5',
        '#4C9A2A', '#007AFF', '#FF3F3D','#F99DBC', 
        '#ACDF87', '#00A3FF','#FF6B6B', '#FEC2D6'
]

experiment_title = 'ReLU_Six'
trackers = torch.load(f'results/history_{experiment_title}.pt').to_dict()

t = None
for i, key in enumerate(trackers.keys()):
    if t == None: t = trackers[key]["accuracy"]
    else: t = t + trackers[key]["accuracy"]
    # axA.plot(trackers[key]["accuracy"], color=col[i], label=f'{key}', alpha=.5)
axA.plot(t/6., color=col[0], label=f'{experiment_title}', alpha=.5)

experiment_title = 'WIRE_Six'
trackers = torch.load(f'results/history_{experiment_title}.pt').to_dict()

t = None
for i, key in enumerate(trackers.keys()):
    if t == None: t = trackers[key]["accuracy"]
    else: t = t + trackers[key]["accuracy"]
    # axA.plot(trackers[key]["accuracy"], color=col[i], label=f'{key}', alpha=.5)
axA.plot(t/6., color=col[1], label=f'{experiment_title}', alpha=.5)


experiment_title = 'SIREN_Six'
trackers = torch.load(f'results/history_{experiment_title}.pt').to_dict()

t = None
for i, key in enumerate(trackers.keys()):
    if t == None: t = trackers[key]["accuracy"]
    else: t = t + trackers[key]["accuracy"]
    # axA.plot(trackers[key]["accuracy"], color=col[i], label=f'{key}', alpha=.5)
axA.plot(t/6., color=col[2], label=f'{experiment_title}', alpha=.5)

experiment_title = 'Gauss_Six'
trackers = torch.load(f'results/history_{experiment_title}.pt').to_dict()

t = None
for i, key in enumerate(trackers.keys()):
    if t == None: t = trackers[key]["accuracy"]
    else: t = t + trackers[key]["accuracy"]
    # axA.plot(trackers[key]["accuracy"], color=col[i], label=f'{key}', alpha=.5)
axA.plot(t/6., color=col[3], label=f'{experiment_title}', alpha=.5)


axA.set_xlabel("Epochs")
axA.set_ylabel("Avg. Pred Difference")
axA.legend()


plt.show()

