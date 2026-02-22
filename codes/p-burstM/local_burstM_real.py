class Config:
    def __init__(self):
        self.path_image_train = "../../rsp_dataset/train_val/gt/train"
        self.path_image_val = "../../rsp_dataset/train_val/gt/val"
        self.path_image_val_cropped = "../../val_48x48_down2_v2"
        self.path_image_real_train = "../../PolarBurstSR/train"
        self.path_image_real_val = "../../PolarBurstSR/val"

        self.path_model = '../../checkpoints'
        self.path_model_selected = f'{self.path_model}/pburstm_model_real.pth'
        self.path_image_original = "../../rsp_dataset/train_val/gt/val"
        self.path_image_gt = "../../PolarBurstSR/test"
        self.path_image_output = "../../PolarBurstSR/pburstm_real"
        
        self.image_type = 'polar_real'
        
        self.output_format = 'degree'

        self.is_fine_tuning = True
        self.path_base_model = '../../checkpoints/pburstm_model_syn.pth'

        self.burstm = True

        self.loss = 'l1'
        
        self.train_multithread = True
        self.ddp_package = 'nccl'
        self.port = '16010'

        self.save_model_code = True
        self.save_output_images = True
        self.val_image_idx_list = [11,74,134,194,229]
        self.polar_val_image_idx_list = [5,43,69,91,167]
        self.test_image_idx_list = [243,350,466,505,549,670,944,948,969]
        self.polar_real_val_image_idx_list = [2, 14, 19, 22, 23, 27]
        
        self.save_min_train_loss = True
        self.save_min_val_loss = True
        self.save_last_iteration = True

        self.debug = False

        self.iter_size = 100 if not self.debug else 1
        self.iter_size_val = 100 if not self.debug else 1

        self.learning_rate = 1e-4
        self.gamma = 1
        self.epochs = 5000
        self.init_epoch = -1
        self.batch_size = [12 if not self.debug else 6] #12
        self.batch_size_val = 12 if not self.debug else 6 #12
        self.burst_size = [14]
        self.burst_size_val = 14
        self.burst_size_epochs = [self.epochs/len(self.burst_size) for i in self.burst_size]
        self.img_size = 48
        self.bd_size = 24
        self.period_model_save = self.epochs
        self.period_valid = 10 if not self.debug else 1
        self.num_workers = 0
        self.device = ['cuda:0']
        self.info_bursts = {}
        self.info_bursts['max_rotation'] = 1
        self.info_bursts['max_translation'] = 24
        self.info_bursts['downsample_factor'] = 2
        self.info_bursts['burst_size'] = self.burst_size_val
        self.info_bursts['crop_size'] = self.img_size * 4 * self.info_bursts['downsample_factor']
        self.info_bursts['random_flip'] = False
        self.info_bursts_real = {}
        self.info_bursts_real['downsample_factor'] = 2
        self.info_bursts_real['burst_size'] = self.burst_size_val
        self.info_bursts_real['crop_size'] = self.img_size * 4 * self.info_bursts_real['downsample_factor']
        self.info_bursts_real['random_flip'] = False
