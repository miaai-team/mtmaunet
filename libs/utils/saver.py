import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):
    def __init__(self, args, config_path=None):
        self.args = args
        self.config_path = config_path
        self.directory = os.path.join('run', args.exp.id, f'{args.dataset.cv.dir_name}',f'fold_{args.dataset.cv.fold}')
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        
        old_run_id = int(self.runs[-1].split('_')[-1]) if self.runs else 0
        if not os.path.isfile(os.path.join(self.directory, f'experiment_{old_run_id:0>4d}','best_pred.txt')):
            run_id = old_run_id
            print(f'Overwrite the previous experiment_{old_run_id:0>4d}, because it did not save the best_pred.txt\n')
        else:
            run_id = old_run_id +1
        self.id = run_id
        self.experiment_dir = os.path.join(self.directory, 'experiment_{:0>4d}'.format(run_id))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if self.config_path:
            assert os.path.isfile(self.config_path)
            if not os.path.isfile(os.path.join(self.experiment_dir, 'config.yaml')):
                shutil.copyfile(self.config_path,os.path.join(self.experiment_dir, 'config.yaml'))
        # logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        # log_file = open(logfile, 'w')
        # p = OrderedDict()
        # p['exp_id'] = self.args.exp.id
        # p['wloss'] = self.args.exp.wloss
        # p['lr'] = self.args.solver.optimizer.lr
        # p['lr_scheduler'] = self.args.lr_scheduler
        # p['loss'] = self.args.solver.loss
        # p['epoch'] = self.args.solver.epoch_max
        # p['base_size'] = self.args.solver.base_size

        # for key, val in p.items():
        #     log_file.write(key + ':' + str(val) + '\n')
        # log_file.close()