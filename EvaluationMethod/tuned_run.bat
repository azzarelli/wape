@echo off
setlocal

rem Define the list of loss functions to iterate over for --pipeline.datamanager.loss-function-density
set ITERATIONS=0 1 2 3

rem Loop over each dataset
set PTHCHANGE=data/GT_reflection data/GT_no_reflection

rem Loop over each batch size, learning rate, and loss function combination and run the script
for %%d in (%PTHCHANGE%) do (
    for %%i in (%ITERATIONS%) do (
        echo Running iteration %%i 
        ns-train instant-ngp --pipeline.datamanager.path=%%d --pipeline.datamanager.train-batch-size=256 --optimizers.fields.optimizer.lr=1e-4 --pipeline.datamanager.loss-function-rgb=MSELoss --pipeline.datamanager.loss-function-density=L1Loss
    )
)
endlocal
