import glob
import random
from data_loader import *
from map_recon_nets_unet import *

mode = 'full'
seed = 5
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

with_msk_channel = False
num_labels = 5
root_dir = 'dataset'
map_list = sorted(glob.glob(os.path.join(root_dir, 'partially_observed_road_layouts', 'val', '*', '*occ_map.png')))
checkpoint_path = 'checkpoints/map_recon_checkpoint_skipconnection_trainval_full_seed_' + str(seed) + '.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define dataloaders
test_set = OccMapDataset('dataset/test_50K_top32.csv', transform=transforms.Compose([ToTensor()]))
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


G = encdec_road_layout(with_msk_channel=with_msk_channel).to(device)


if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    G.load_state_dict(state['state_dict_G'])
    print('trained model loaded...')
else:
    print('cannot load trained model...')


G.eval()  # Set model to evaluate mode

# Iterate over data.
road_placeholder = torch.LongTensor(1*64*64, 1).to(device) % num_labels
road_onehot = torch.FloatTensor(1*64*64, num_labels).to(device)

for i, temp_batch in enumerate(test_loader):
    if i % 100 == 0:
        print('example no. ', i)
    temp_map = temp_batch['map'].long().to(device)
    road_onehot.zero_()

    temp_map = road_onehot.scatter_(1, temp_map.view(-1, 1), 1).view(1, 64, 64, 5).permute(0, 3, 1, 2)
    temp_map_input = temp_map[:, 0, :, :] + 0.5 * temp_map[:, 4, :, :]
    temp_map_input = temp_map_input.unsqueeze(1)
    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        if with_msk_channel:
            pred_map = G(torch.cat([temp_map_input, temp_map[:, 4, :, :].unsqueeze(1)], dim=1), False)
        else:
            pred_map = G(temp_map_input, False)

        # torch_img_visualization(2, [pred_map.detach(), temp_map_input.detach()])

        pred_map = ((1. - pred_map) < 0.5).float()
        io.imsave(map_list[i][:-4] + '_road_pred_residual_adversarial_trainval_' + mode + '.png', pred_map.cpu().numpy().astype(np.uint16).reshape((64, 64))*65535)
