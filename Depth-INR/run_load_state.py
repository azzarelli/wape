import copy

import matplotlib.pyplot as plt
import tensordict
import torch
from modules.gauss import Gauss_INR
from modules.relu_posenc import RELU_INR
from modules.siren import SIREN_INR
from modules.wire import Wire_INR
from scene import Scene
from trainer import Trainer


experiment_title = 'Put Your Title Here'
net_names = ['SIREN', 'WIRE', 'RELU', 'Gauss']

# Eh these things...
lr = 1e-5
epochs = 20
print_frequency = 50
test_frequency = 20

batch_data = {'train':256,'test':512} 
test_train_split = 0.8

hidden_layers = 5   # Number of hidden layers in the mlp
hidden_features = 256

omega_wire = 1.4
omega_siren = 2.6
sigma_wire = 1.0
sigma_gauss = 1.2

paramgroups = {
    'WIRE':{
        "title":'WIRE',
        "net":Wire_INR(in_features=6,
                        out_features=1, 
                        hidden_features=hidden_features,
                        hidden_layers=hidden_layers,
                        first_omega_0=omega_wire,
                        hidden_omega_0=omega_wire,
                        scale=sigma_wire,
                        ),
        "optim": torch.optim.Adam,
        "looser": torch.nn.MSELoss(),

        "epochs":epochs, 
        "test_frequency":test_frequency,
        "print_frequency":print_frequency,
        "lr":lr,
        "batch_data":batch_data,
        "test_train_split":test_train_split
}
,
    'Gauss':{"title":'Gauss',
        "net":Gauss_INR(in_features=6,
                        out_features=1, 
                        hidden_features=hidden_features,
                        hidden_layers=hidden_layers,
                        scale=sigma_gauss
                        ),
        "optim": torch.optim.Adam,
        "looser": torch.nn.MSELoss(),

        "epochs":epochs, 
        "test_frequency":test_frequency,
        "print_frequency":print_frequency,
        "lr":lr,
        "batch_data":batch_data,
        "test_train_split":test_train_split
    }
,
    'SIREN':{
        "title":'SIREN',
        "net":SIREN_INR(in_features=6,
                        out_features=1, 
                        hidden_features=hidden_features,
                        hidden_layers=hidden_layers,
                        first_omega_0=omega_siren,
                        hidden_omega_0=omega_siren
                        ),
        "optim": torch.optim.Adam,
        "looser": torch.nn.MSELoss(),

        "epochs":epochs, 
        "test_frequency":test_frequency,
        "print_frequency":print_frequency,
        "lr":lr,
        "batch_data":batch_data,
        "test_train_split":test_train_split
}
,
    'RELU' :{
        "title":'RELU',
        "net":RELU_INR(in_features=6,
                        out_features=1, 
                        hidden_features=hidden_features,
                        hidden_layers=8
                        ),
        "optim": torch.optim.Adam,
        "looser": torch.nn.MSELoss(),

        "epochs":epochs, 
        "test_frequency":test_frequency,
        "print_frequency":print_frequency,
        "lr":lr,
        "batch_data":batch_data,
        "test_train_split":test_train_split
    }
}

res_ = []
for net_name in net_names:
    scene = Scene()
    scene.generate_data(experiment_title)
    # scene.load_data(experiment_title)
    scene.set_trainer(Trainer(title=experiment_title))


    params = paramgroups[net_name]

    scene.initialiser_new_trainer( (
                net_name,
                copy.deepcopy(params['net']),
                params['optim'],
                params['looser'],
                params['epochs'], 
                params["test_frequency"],
                params["print_frequency"],
                params["lr"],
                params["batch_data"],
                params["test_train_split"]
                ) )

    scene.load_trainer(experiment_title+net_name)

    
    scene.display_pred_scene(title=experiment_title, view_scale=(8,8,8), show_rays=False)

    """Rendering Images for each Model
    """
    res = scene.disp_heatmap(title=experiment_title)

    res_.append(res[:3]) # appendd to results
    gt = res[3:] # ground truth

fig, axs = plt.subplots(3, 5, subplot_kw={'xticks': [], 'yticks': []}, figsize=(12, 9))

for j, r in enumerate(res_):
    i = j+1
    axs[0,i].imshow(r[0], cmap='hot')#, interpolation='spline36')
    axs[1,i].imshow(r[1], cmap='hot')#, interpolation='spline36')
    axs[2,i].imshow(r[2], cmap='hot')#, interpolation='spline36')
    print(i,j)
    axs[0,i].set_title(net_names[j])

axs[0,0].imshow(gt[0], cmap='hot')#,interpolation='spline36')
axs[1,0].imshow(gt[1], cmap='hot')#, interpolation='spline36')
axs[2,0].imshow(gt[2], cmap='hot')#, interpolation='spline36')
axs[0,0].set_title('GT')

plt.tight_layout()
plt.show()

    
