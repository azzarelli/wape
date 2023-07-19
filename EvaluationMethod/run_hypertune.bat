@echo off
setlocal

rem Define the list of batch sizes to iterate over
set BATCH_SIZES=512 1024 2048 4096

rem Define the list of learning rates to iterate over
set LR_VALUES=1e-2 1e-3 1e-4 1e-5

rem Define the list of loss functions to iterate over for --pipeline.datamanager.loss-function-rgb
set RGB_LOSS_FUNCTIONS=MSELoss

rem Define the list of loss functions to iterate over for --pipeline.datamanager.loss-function-density
set DENSITY_LOSS_FUNCTIONS=MSELoss L1Loss

set SEED=42

rem Loop over each batch size, learning rate, and loss function combination and run the script
for %%d in (%DENSITY_LOSS_FUNCTIONS%) do (
    for %%r in (%RGB_LOSS_FUNCTIONS%) do (
        for %%s in (%BATCH_SIZES%) do (
            for %%l in (%LR_VALUES%) do (
                echo Running script with batch size %%s, learning rate %%l, rgb loss function %%r, and density loss function %%d...
                ns-train mipnerf --pipeline.datamanager.train-batch-size=%%s --optimizers.fields.optimizer.lr=%%l --pipeline.datamanager.loss-function-rgb=%%r --pipeline.datamanager.loss-function-density=%%d --machine.seed SEED
                
                ns-train nerfacto --pipeline.datamanager.train-batch-size=%%s --optimizers.fields.optimizer.lr=%%l --pipeline.datamanager.loss-function-rgb=%%r --pipeline.datamanager.loss-function-density=%%d --machine.seed SEED

                ns-train instant-ngp --pipeline.datamanager.train-batch-size=%%s --optimizers.fields.optimizer.lr=%%l --pipeline.datamanager.loss-function-rgb=%%r --pipeline.datamanager.loss-function-density=%%d --machine.seed SEED
            )
        )
    )
)

endlocal
