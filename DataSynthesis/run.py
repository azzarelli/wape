from generator import SceneGenerator
import cProfile as cp
import argparse
import torch

if __name__ == "__main__":
    with torch.no_grad():

        parser = argparse.ArgumentParser(
            prog='PROG',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument('-n', '--experiment-name', default="test",
                        help='Select a single surface to view (default: "")')

        parser.add_argument('-ev', '--eval-views',  type=float, nargs="+",
                        help='This is the list of views to omit in the training data and only be used dring evaluation.'
                        ' (i.e.)  (default: 0)')
   
        parser.add_argument('-t', '--get-runtime', action='store_true', 
                        help='Get runtime of each function/class (default: True)')
        
        parser.add_argument('-cxcy', '--camera-shape',  type=float, nargs="+",
                            help='The pixel size of the in-view images (default: (100,100))')

        parser.add_argument('-b', '--batchsize',  type=float, nargs="+",
                        help='Batchsize for processing images (default: (10240, 512))')


        parser.add_argument('-l', '--load-data', action='store_true', 
                        help='Get runtime of each function/class (default: False)')

        args = vars(parser.parse_args())

        # Set the Experiment Title
        experiment_title = args['experiment_name']
        load_flag=args['load_data']

        if args['batchsize'] is None:
            args['batchsize'] = (10240, 512)
        if args['camera_shape'] is None:
            args['camera_shape'] = (50., 50.)

        # print(args)

        if load_flag:
            pass
        else:
            # Scenerator Handler
            scenerator = SceneGenerator(name=experiment_title, camera_shape=tuple(args['camera_shape']), batch_size=tuple(args['batchsize']), eval_views=tuple(args['eval_views']))
            
            if args['get_runtime']:
                cp.run("scenerator.generate_data()")
            else:
                scenerator.generate_data()
                                    
    