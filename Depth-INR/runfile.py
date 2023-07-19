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

# Set the Experiment Title
experiment_title = 'Put Your Title Here'

# Scene Handler
scene = Scene()
scene.generate_data(experiment_title) # choose to generate data (saves using experiment titled) or load data
# scene.load_data()

# Training & Network Handler 
scene.set_trainer(Trainer(title=experiment_title))

# Display Scene Prior to Training
# scene.display_scene(title=experiment_title, show_GT_intersections=False, view_scale=(15,15,10))


# Training Params
lr = 1e-5
epochs = 1500
print_frequency = 200
test_frequency = 20

batch_data = {'train':256,'test':512} 
test_train_split = 0.8

hidden_layers = 5   # Number of hidden layers in the mlp
hidden_features = 256

# Activation Params for Gauss, Wire and Siren
omega_wire = 1.4
omega_siren = 2.6
sigma_wire = 1.0
sigma_gauss = 1.2

# Param Dicts, For network initialisation during run
WIRE_params = {
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

Gauss_params = {
    "title":'Gauss',
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

SIREN_params = {
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

RELU_params = {
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

# Run Params
model_params = [SIREN_params, WIRE_params, Gauss_params, RELU_params]
things = [1] # subcomponent to vary -> can be used to configure tests over different params e.g. lr
trackers = {}

for thing in things:
    for idx, model in enumerate(model_params):
        name = experiment_title+model['title'] # +'_'+(str(thing))
        print(name)

        # Initialise and Run New trainer
        scene.initialiser_new_trainer( (
            name,
            copy.deepcopy(model['net']),
            model['optim'],
            model['looser'],
            model['epochs'], 
            model["test_frequency"],
            model["print_frequency"],
            model["lr"],
            model["batch_data"],
            model["test_train_split"]
            ) )
        
        trackers[name] = scene.run(title=experiment_title, load_data=True)
        scene.save_trainer()


# Plot Training Results (and Save)
print(f'{experiment_title} Plotting ...')
     
col = ['#1E5631', '#0700C4', '#910000','#FE036A', 
        '#A4DE02', '#0000FF', '#6B0000', '#F5347F', 
        '#76BA1B','#0052FF', '#D32431', '#FC72A5',
        '#4C9A2A', '#007AFF', '#FF3F3D','#F99DBC', 
        '#ACDF87', '#00A3FF','#FF6B6B', '#FEC2D6'
    ]

# Save Training Results
targs = tensordict.TensorDict(trackers, [])
torch.save(targs, f'results/history_{experiment_title}.pt')

# Load Training Results
trackers = torch.load(f'results/history_{experiment_title}.pt').to_dict() # Trakced accuracy, loss and PSNR

# Display Preidcted Scene
# scene.display_pred_scene(title=experiment_title, view_scale=(15,15,10), show_rays=False, show_GT_intersections=False)
# scene.display_pred_scene(title=experiment_title, view_scale=(15,15,10), show_rays=False, alpha=.1)

fig, (axA, axP) = plt.subplots(1,2)
# axA.title.set_text(experiment_title+' WSAPE')
# axP.title.set_text(experiment_title+' PSNR')

for i, key in enumerate(trackers.keys()):
    axP.plot(trackers[key]["psnr"], color=col[i], label=f'{key}', alpha=.5)
    axA.plot(trackers[key]["accuracy"], color=col[i], label=f'{key}', alpha=.5)


axA.set_xlabel("Epochs")
axA.set_ylabel("WSAPE")
axP.set_xlabel("Epochs")
axP.set_ylabel("PSNR")
axA.legend()
fig.set_size_inches(18.5, 8.5)

fig.savefig(f'results/{experiment_title}_psnr_wsape.png')

plt.show()


# Useful Commands to run! -----------------------------------------------------------------------------

# Commands for Generating (and saving) and Loading data
# scene.generate_data(experiment_title)
# scene.load_data(experiment_title)

# Commands for Displaying Scene
# scene.display_scene(title=experiment_title, show_rays=False)
# scene.display_pred_scene(title=experiment_title, view_scale=(15,15,10), show_rays=False)
