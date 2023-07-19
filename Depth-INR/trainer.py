from dataclasses import dataclass

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import nn
from utils_ import (
    PSNR,
    Accuracy,
    Accuracy_elementwise,
    Loss,
    Planes,
    Ray,
    RayBundle,
    build_block,
    get_view_rays,
)


class Trainer:
    """The class who trains  the nets and displays all the important info
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialise_trainer_network(self,
        title,
        net,
        optim = torch.optim.Adam,
        looser = torch.nn.MSELoss(),

        epochs=20, 
        test_frequency=20,
        print_frequeny = 20,
        lr = 1e-5,
        batch_data = {'train':128,'test':256},
        test_train_split = 0.8
    ):
        """Classic Trainer Initialisation"""
        self.title = title
        
        self.epochs = epochs
        self.test_frequency = test_frequency
        self.print_frequeny = print_frequeny
        self.lr = lr
        self.batch_data = batch_data
        self.test_train_split = test_train_split

        self.net = net.to(self.device)
        self.optimizer = optim(self.net.parameters(), lr=self.lr)
        self.looser = looser

        torch.save(self.net.state_dict(), f'init_networks/mlp_{self.title}_initialstate.pt')
    
    def save_state(self):
        """Save State
        """
        torch.save(self.net.state_dict(), f'checkpoints/{self.title}.pt')

    def load_state(self, title, rays):
        """Load Network State
        """
        self.net.load_state_dict(torch.load(f'checkpoints/{title}.pt'))
        self.net.eval()
        self.rays = rays

    def __init__(self, 
        title=''
    ) -> None:
        """Vague Initialisation
        """
        self.title = title
        self.delta = 0.05


    def init_data_loaders(self, rays):
        """Using tensor rays, containing all rays -> initialise the trianing and testing data loaders
        """
        tt_data_ratio=self.test_train_split 
        batch_data=self.batch_data

        rays = rays[torch.randperm(rays.size()[0])] # random shuffle
                
        train_data = rays[:int(rays.size(0)*tt_data_ratio)]
        test_data = rays[int(rays.size(0)*tt_data_ratio):]

        train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                                batch_size=batch_data['train'], 
                                                shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                                batch_size=batch_data['test'], 
                                                shuffle=True)

        return train_loader, test_loader

    def train_step(self, i, x, targs, data):
        """Perform one step of training given the target dictionary and input data
        """
        t = self.net(data)
        outputs = torch.cat([t, t+self.delta], dim=1)

        n, f = targs

        loss = Loss(self.looser, outputs, n)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    
    def test_step(self, i, x, targs, data):
        """Perform one step of testing/validation
        """
        t = self.net(data)
        outputs = torch.cat([t, t+self.delta], dim=1)

        n, f = targs

        accuracy = Accuracy(outputs, n)
        psnr = PSNR(outputs, n)

        return accuracy.item(), psnr
    
    def run(self, rays, targs, get_targs_fn):
        """Run the training using rays, target dictionary and target function (which looks in dictionary to return targets given input data)
        """
        self.rays = rays

        self.init_data_loaders(rays)

        epochs = self.epochs
        test_frequency = self.test_frequency
        print_frequeny = self.print_frequeny
        # This function should do the following
        # print('Inititialise Training and Testing Environents...')
        train, test = self.init_data_loaders(rays)

    
        training = True
        accuracy_track = []
        psnr_track = []
        loss_track = []
        last_accuracy = 0.
        last_loss = None
        last_psnr = 0.
        for e in range(epochs):
            # Determine if we should be training or testing
            if ((e+1)%test_frequency)==0 or e ==2:
                training = False
            else:
                training = True

            # Execture training or testing step
            if training:
                for i,x in enumerate(train):
                    oris = x[:, 0:3].to(self.device)# split ray data into direction and origin
                    dirs = x[:, 3:6].to(self.device)
                    
                    data = RayBundle(origins=oris.float(),
                    directions=dirs.float(),
                    )
                    
                    loss = self.train_step(i,x, 
                        get_targs_fn(oris, dirs, targs),
                        data)
                    last_loss = loss
                    loss_track.append(loss)
            else:
                with torch.no_grad():
                    for i,x in enumerate(test):
                        if i == 0:
                            oris = x[:, 0:3].to(self.device)# split ray data into direction and origin
                            dirs = x[:, 3:6].to(self.device)
                            
                            data = RayBundle(origins=oris.float(),
                                                directions=dirs.float(),
                                                )

                            accuracy,psnr = self.test_step(
                                            i, 
                                            x, 
                                            get_targs_fn(oris, dirs, targs),
                                            data
                                        )
                            last_accuracy = accuracy
                            last_psnr = psnr
                            accuracy_track.append(accuracy)
                            
                            psnr_track.append(psnr)
                        else:
                            break
            
            if print_frequeny != 0 and ((e+1)%print_frequeny)==0:
                print ('Epoch [{}/{}], Accuracy: {:.4f}, PSNR: {:.4f}'.format(e+1, epochs, last_accuracy, last_psnr))

            self.trackers = {'loss':loss_track, 'accuracy':accuracy_track, 'psnr': psnr_track}

        return self.trackers
        
    def disp_scene(self, 
            title= 'Test',  
            size=(20,20,20),

            bounds = None,
            blocks_cols = None,
            display_rays = None,
            rays_cols = None,
            display_intersections = None,
            preds = None,
            alpha = .5
            ):
        """Display (Save Png) selected components of our scene
        """
        fig = plt.figure(2)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        # Cubes:
        if bounds is not None and blocks_cols is not None :
            for i, b in enumerate(bounds):
                xx = b[:, 0].tolist()
                yy = b[:, 1].tolist()
                zz = b[:, 2].tolist()
                verts = [list(zip(xx,yy,zz))]
                ax.add_collection3d(Poly3DCollection(verts, alpha=.1, color='c')) # blocks_cols[i]''))
        # Rays:
        if display_rays is not None and rays_cols is not None:
            for r, col in zip(display_rays, rays_cols):
                r[:,3:6] = r[:,3:6]/5. 
                ax.quiver(r[:,0],r[:,1],r[:,2],r[:,3],r[:,4],r[:,5],  linewidth=.5, color='g', alpha=0.5)
        # Intersection point cloud
        if display_intersections is not None:
            ax.scatter(display_intersections[:,0], display_intersections[:,1],
                        display_intersections[:,2], c='k', s=2, alpha=alpha)
        # Prediction point cloud
        if preds is not None:
            ax.scatter(preds[:,0],preds[:,1],preds[:,2], c='r', s=50, alpha=1.)

        ax.set_zlabel('z', fontsize=30, rotation = 0)
        ax.set_xlabel('x', fontsize=30, rotation = 0)
        ax.set_ylabel('y', fontsize=30, rotation = 0)
        
        ax.set_xlim3d(0, size[0])
        ax.set_ylim3d(0, size[1])
        ax.set_zlim3d(0, size[2])
        ax.view_init(elev=4., azim=34.)

        plt.show()
        # plt.savefig(f'{title}_approx_scene.png')

    def disp_approximations(self,
        title='',  
        size=(6,6,6),

        bounds = None,
        blocks_cols = None,
        display_rays = None,
        rays_cols = None,
        display_intersections = None,
        alpha = .5,
        ):
        """ Display predictions of the entire scene
        """

        oris = self.rays[:, 0:3].to(self.device) # split ray data into direction and origin
        dirs = self.rays[:, 3:6].to(self.device)
        with torch.no_grad():
            data = RayBundle(origins=oris.float(),
                        directions=dirs.float(),
                        )
            # Forward pass
            outputs = self.net(data)

            rn = oris + outputs[:,0].unsqueeze(1) * dirs
            # rf = oris + outputs[:,1].unsqueeze(1) * dirs
            preds = rn.cpu().numpy()

        # pass preds to visualisation function
        self.disp_scene(
            title, size, bounds,
            blocks_cols, display_rays, rays_cols, 
            display_intersections, preds,
            alpha = alpha
        )
    
    def save_training_results_png(self, title='Test'):
        """Display the accuracy and loss graphs
        """
        
        print(f'{title} Plotting ...')

        trackers = self.trackers
        accuracy_track, loss_track = trackers['accuracy'], trackers['loss']
        
        fig, ax = plt.subplots()
        ax.title.set_text(title+' Accuracy')
        ax.plot(accuracy_track, color=np.random.rand(3), label=f'Scene')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Avg. Pred Difference")
        ax.legend()
        plt.savefig(f'results/{title}_Error.png')

        fig, ax = plt.subplots()
        ax.title.set_text(title+ ' Loss')
        ax.plot(loss_track, color=np.random.rand(3), label=f'Scene')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Avg. Pred Difference")
        ax.legend()
        plt.savefig(f'results/{title}_Loss.png')

    def metric(self,
            targs, get_targs_fn
            ):
        """Whole scene metric result
        """
        oris = self.rays[:, 0:3].to(self.device) # split ray data into direction and origin
        dirs = self.rays[:, 3:6].to(self.device)

        with torch.no_grad():
            data = RayBundle(origins=oris.float(),
                        directions=dirs.float(),
                        )
            # Forward pass
            outputs = self.net(data)

            n, f = get_targs_fn(oris, dirs, targs)

            accuracy = Accuracy(outputs, n, f)

            rn = oris + outputs[:,0].unsqueeze(1) * dirs
            # rf = oris + outputs[:,1].unsqueeze(1) * dirs
            preds = rn.cpu().numpy()

        print(accuracy)

    def disp_heatmap(self, targs, get_targs_fn,
           ):
        """Return list containing 2D visual depth-map 
        """
        oris = self.rays[:, 0:3].to(self.device) # split ray data into direction and origin
        dirs = self.rays[:, 3:6].to(self.device)

        with torch.no_grad():
            data = RayBundle(origins=oris.float(),
                        directions=dirs.float(),
                        )
                        
            # Forward pass
            outputs = self.net(data)

            n, f = get_targs_fn(oris, dirs, targs)

            cnt_h = 0
            cnt = 0
            im0 = []
            im1 = []
            im2 = []
            gt0 = []
            gt1 = []
            gt2 = []

            for o,d, a,b  in zip(oris, dirs, outputs[:,0], n):
                if (dirs[0]- d).sum(0) == 0:
                    cnt += 1
                    if o[2] == oris[0][2]:
                        cnt_h += 1
                    im0.append(a.item()) # .item())
                    gt0.append(b.item())
                elif (dirs[601]- d).sum(0) == 0:
                    im1.append(a.item())
                    gt1.append(b.item())
                elif (dirs[1203]- d).sum(0) == 0:
                    im2.append(a.item())   
                    gt2.append(b.item())


            cnt_w = int(cnt/cnt_h)
            
            # Return three views
            img0 = [[ [im0[((cnt_w-i-1)*cnt_h) + (j)] * 100.] for j in range(cnt_h)] for i in range(cnt_w)]
            img1 = [[ [im1[((cnt_w-i-1)*cnt_h) + (j)] * 100.] for j in range(cnt_h)] for i in range(cnt_w)]
            img2 = [[ [im2[((cnt_w-i-1)*cnt_h) + (j)] * 100.] for j in range(cnt_h)] for i in range(cnt_w)]
            # Return the GT views
            gtimg0 = [[ [gt0[((cnt_w-i-1)*cnt_h) + (j)] * 100.] for j in range(cnt_h)] for i in range(cnt_w)]
            gtimg1 = [[ [gt1[((cnt_w-i-1)*cnt_h) + (j)] * 100.] for j in range(cnt_h)] for i in range(cnt_w)]
            gtimg2 = [[ [gt2[((cnt_w-i-1)*cnt_h) + (j)] * 100.] for j in range(cnt_h)] for i in range(cnt_w)]
            

            return [img0, img1, img2, gtimg0, gtimg1, gtimg2]

            

                

            
    