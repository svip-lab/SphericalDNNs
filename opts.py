import argparse
import os
import ref

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
  def init(self):
    self.parser.add_argument('-expID', default = 'default', help = 'Experiment ID')
    self.parser.add_argument('-test', action = 'store_true', help = 'test')
    self.parser.add_argument('-DEBUG', type = int, default = 0, help = 'DEBUG level')
    self.parser.add_argument('-demo', default = '', help = 'path/to/demo/image')
    
    self.parser.add_argument('-resume', default=False, type=bool, metavar='BOOL', help='Use the checkpoint or not')
    self.parser.add_argument('-loadModel', default = None, help = 'Provide full path to a previously trained model')
    
    ## Train parameters
    self.parser.add_argument('-lr', type = float, default = 3e-1, help = 'Learning Rate')
    self.parser.add_argument('-epochs', type = int, default = 250, help = '#training epochs')
    self.parser.add_argument('-trainBatch', type = int, default = 8, help = 'Mini-batch size')
    self.parser.add_argument('--batch-size', type = int, default = 32, help = 'batch size')

    self.parser.add_argument('--clear-cache', default=False, type=bool, metavar='BOOL', help='Clear dataset cache')

    ## Visdom
    self.parser.add_argument('--plot-server', type=str, default='http://10.19.124.11', help='IP address')
    self.parser.add_argument('--exp-name', type=str, default='lstm_test', help='The env name in visdom')
    self.parser.add_argument('--plot-port', type=int, default=31830, help='Port number')
    self.parser.add_argument('--save-interval', type=int, default=20, help='Port number')
    
    
    self.parser.add_argument('--snapshot-fname-prefix', default='exps/', type=str, metavar='PATH', help='path to snapshot')
    self.parser.add_argument('--snapshot-fname-dir', default='exps/', type=str, metavar='PATH', help='path to snapshot')
    
    self.parser.add_argument('--start-epoch', type = int, default = 0, help = 'batch size')

    
    self.parser.add_argument('--sequence-len', type = int, default = 2, help = 'sequence-len')

    # Ablation Study
    self.parser.add_argument('--use-sconv', default='False', type=str, metavar='BOOL', help='Use SConv')
    self.parser.add_argument('--use-spooling', default='False', type=str, metavar='BOOL', help='Use SPooling')
    self.parser.add_argument('--use-sconvlstm', default='False', type=str, metavar='BOOL', help='Use SConvLSTM')
    self.parser.add_argument('--use-smse', default='False', type=str, metavar='BOOL', help='Use SMSE')

    # Eval save image
    self.parser.add_argument('--sal-image-fname-dir', default='exps/', type=str, metavar='PATH', help='path to sal image')
    self.parser.add_argument('--epoch-st', default=0, type=int, help='rank of distributed processes')
    self.parser.add_argument('--epoch-end', default=250, type=int, help='rank of distributed processes')

    self.parser.add_argument('--is-retrain', default='False', type=str, metavar='BOOL', help='Use SMSE')
    self.parser.add_argument('--is-train-lstm-normal', default='False', type=str, metavar='BOOL', help='Use SMSE')

    self.parser.add_argument('--sphereconv-sal', default='False', type=str, metavar='BOOL', help='Use SMSE')

    self.parser.add_argument('--input-size', default=64, type=int, help='rank of distributed processes')
    self.parser.add_argument('--is-sphereconv-convlstm', default='False', type=str, metavar='BOOL', help='Use SMSE')
    self.parser.add_argument('--is-sphereconv-convlstm-low', default='False', type=str, metavar='BOOL', help='Use SMSE')

    self.parser.add_argument('--is-standard-convlstm', default='False', type=str, metavar='BOOL', help='Use SMSE')
    self.parser.add_argument('--is-standard-sal', default='False', type=str, metavar='BOOL', help='Use SMSE')
    self.parser.add_argument('--is-standard-convlstm-low', default='False', type=str, metavar='BOOL', help='Use SMSE')

    self.parser.add_argument('--convlstm-type', default='False', type=str, metavar='BOOL', help='Use SMSE')

    self.parser.add_argument('--network-type', default='False', type=str, metavar='BOOL', help='Use SMSE')
    self.parser.add_argument('--conv-type', default='False', type=str, metavar='BOOL', help='Use SMSE')

    self.parser.add_argument('--is_vis', dest='is_vis',help='Set to True for forward network.', action='store_true',default=False)

    self.parser.add_argument('--model_type', type=str, default='spherical', help='dataset directory')
    self.parser.add_argument('--eval', dest='eval',
                  help='Set to True for forward network.', action='store_true',
                  default=False)
    self.parser.add_argument('--use_visdom', dest='use_visdom', help='Set to True for .', action='store_true',
                             default=False)
    self.parser.add_argument('--use_tensorboard', dest='use_tensorboard', help='use tensorboard', action='store_true',
                             default=False)

    self.parser.add_argument('--use_equirect_rotate', dest='use_equirect_rotate', help='use_equirect_rotate', action='store_true',
                             default=False)

    self.parser.add_argument('--use_mnist', dest='use_mnist', help='use_mnist', action='store_true', default=False)

    self.parser.add_argument('--use_grad_cam_layers', dest='use_grad_cam_layers', help='use grad cam',
                             action='store_true', default=False)
    self.parser.add_argument('--use_eval_auc', dest='use_eval_auc', help='use grad cam', action='store_true',
                             default=False)
    self.parser.add_argument('--use_grad_cam', dest='use_grad_cam', help='use grad cam', action='store_true',
                             default=False)

    self.parser.add_argument('--rot_y', default=0, type=int, help='rot_y')
    self.parser.add_argument('--rot_z', default=0, type=int, help='rot_z')

    self.parser.add_argument('--use_correct', dest='use_correct', help='use grad cam', action='store_true',
                             default=False)
    self.parser.add_argument('--use_mnist_bn', dest='use_mnist_bn', help='use grad cam', action='store_true',
                             default=False)
    self.parser.add_argument('--use_mnist_channel', dest='use_mnist_channel', help='use grad cam', action='store_true',
                             default=False)

  def parse(self):
    self.init()  
    self.opt = self.parser.parse_args()
    
    args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                if not name.startswith('_'))
    refs = dict((name, getattr(ref, name)) for name in dir(ref)
                if not name.startswith('_'))
    
    self.opt.saveDir = os.path.join(ref.expDir, self.opt.expID) # ../exp/default
    if not os.path.exists(self.opt.saveDir):
      os.makedirs(self.opt.saveDir)

    if not os.path.exists(self.opt.sal_image_fname_dir):
      os.makedirs(self.opt.sal_image_fname_dir)

    self.opt.snapshot_fname_prefix = self.opt.saveDir + '/'+ self.opt.snapshot_fname_dir +  '/' + self.opt.snapshot_fname_prefix 
    if not os.path.exists(self.opt.snapshot_fname_prefix):
      os.makedirs(self.opt.snapshot_fname_prefix)

    self.opt.snapshot_fname_log = os.path.join(ref.expDir, 'logs')

    file_name = os.path.join(self.opt.saveDir, self.opt.snapshot_fname_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> Args:\n')
      for k, v in sorted(args.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))
      opt_file.write('==> Args:\n')
      for k, v in sorted(refs.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))
 
    return self.opt
