import torch, torchvision, os, collections
import torchvision.transforms as transforms
from netdissect import parallelfolder, zdataset, renormalize, segmenter
from experiment import oldalexnet, oldvgg16, oldresnet152
from torch.utils.data import Dataset, DataLoader
from experiment.ucf101.dataset.dataset import RGB_Dissect, RGB_Dataset,FLOW_Dataset
from experiment.ucf101.model.vgg16 import MyVGG16


def load_proggan(domain):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)
    from . import proggan
    weights_filename = dict(
        bedroom='proggan_bedroom-d8a89ff1.pth',
        church='proggan_churchoutdoor-7e701dd5.pth',
        conferenceroom='proggan_conferenceroom-21e85882.pth',
        diningroom='proggan_diningroom-3aa0ab80.pth',
        kitchen='proggan_kitchen-67f1e16c.pth',
        livingroom='proggan_livingroom-5ef336dd.pth',
        restaurant='proggan_restaurant-b8578299.pth',
        celebhq='proggan_celebhq-620d161c.pth')[domain]
    # Posted here.
    url = 'https://dissect.csail.mit.edu/models/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1+
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = proggan.from_state_dict(sd)
    return model

def load_ucf101_classifier(architecture):
    model_factory = dict(vgg16=oldvgg16.vgg16)[architecture]
    model = model_factory(num_classes=101)
    # model = MyVGG16("RGB")
    lr, bs, epoch = "0.001", "64", "210"
    checkpoint_path = os.path.join("/", "home", "dwijaya", "dissect",
                                   "experiment", "ucf101", "checkpoint")
    checkpoint_path = os.path.join(checkpoint_path, "best", '_lr=%s_bs=%s_EPOCH=%s.pth' % (lr, bs, epoch))
    # checkpoint_path = os.path.join(os.getcwd(), "..", "checkpoint", "best",'_lr=%s_bs=%s_EPOCH=%s.pth' % (lr, bs, epoch))
    state = torch.load(checkpoint_path, map_location='cuda:1')
    renamed_keys = [a.replace("vgg16.", "") for a in state.keys()]
    state = collections.OrderedDict(zip(renamed_keys, state.values()))

    model.load_state_dict(state_dict=state)

    model.features = torch.nn.Sequential(collections.OrderedDict(zip([
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2',
        'pool1',
        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3',
        'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3',
        'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3',
        'pool5'],
        model.features)))

    model.classifier = torch.nn.Sequential(collections.OrderedDict(zip([
        'fc6', 'relu6',
        'drop6',
        'fc7', 'relu7',
        'drop7',
        'fc8a'],
        model.classifier)))

    model.eval()
    return model

def load_classifier(architecture):
    model_factory = dict(
            alexnet=oldalexnet.AlexNet,
            vgg16=oldvgg16.vgg16,
            resnet152=oldresnet152.OldResNet152)[architecture]
    weights_filename = dict(
            alexnet='alexnet_places365-92864cf6.pth',
            vgg16='vgg16_places365-0bafbc55.pth',
            resnet152='resnet152_places365-f928166e5c.pth')[architecture]
    model = model_factory(num_classes=365)
    baseurl = 'https://dissect.csail.mit.edu/models/'
    url = baseurl + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model.load_state_dict(sd)
    model.eval()
    return model

def load_ucf101_dataset(crop_size=None,in_dataloader=False, is_train=False, is_all_frames=False):
    DATASET_DIR = os.path.abspath(os.path.join("/", "mnt", "sdb", "danielw"))
    RGB_DIR = os.path.join(DATASET_DIR, "jpegs_256")

    if(is_all_frames):
        _Dataset = RGB_Dataset
    else:
        _Dataset = RGB_Dissect

    test_transform = transforms.Compose([
        transforms.CenterCrop((crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # renormalize.NORMALIZER['imagenet']
    ])

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    if(is_train):
        dataset = _Dataset(data_root=RGB_DIR, is_train=is_train, transform=train_transform)
    else:
        # dataset = RGB_Dissect(data_root=RGB_DIR, is_train=False,is_all_frames=is_all_frames, transform=test_transform)
        dataset = _Dataset(data_root=RGB_DIR, is_train=is_train, transform=test_transform, test_frame_size=10)

    if(in_dataloader):
        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=6)
        return dataloader

    return dataset


def load_dataset(domain, split=None, full=False, crop_size=None, download=True):
    if domain in ['places', 'imagenet']:
        if split is None:
            split = 'val'
        #Create the directory.
        dirname = 'datasets/%s/%s' % (domain, split)
        if download and not os.path.exists(dirname) and domain == 'places':
            os.makedirs('datasets', exist_ok=True)
            torchvision.datasets.utils.download_and_extract_archive(
                'https://dissect.csail.mit.edu/datasets/' +
                'places_%s.zip' % split,
                'datasets',
                md5=dict(val='593bbc21590cf7c396faac2e600cd30c',
                         train='d1db6ad3fc1d69b94da325ac08886a01')[split])
        places_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(crop_size or 224),
            torchvision.transforms.ToTensor(),
            renormalize.NORMALIZER['imagenet']])

        return parallelfolder.ParallelImageFolders([dirname],
                classification=True,
                shuffle=True,
                transform=places_transform)
    else:
        #This is what I added, here is also on UCF101 but for network dissection experiment.
        DATASET_DIR = os.path.abspath(os.path.join("/", "mnt", "sdb", "danielw"))
        RGB_DIR = os.path.join(DATASET_DIR, "jpegs_256")

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            # torchvision.transforms.CenterCrop(crop_size or 224),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            renormalize.NORMALIZER['imagenet']
        ])

        dataset = RGB_Dissect(data_root=RGB_DIR, is_train=True, transform=train_transform)
        dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=6)

        return dataset


def load_segmenter(segmenter_name='netpqc'):
    '''Loads the segementer.'''
    all_parts = ('p' in segmenter_name)
    quad_seg = ('q' in segmenter_name)
    textures = ('x' in segmenter_name)
    colors = ('c' in segmenter_name)

    segmodels = []
    segmodels.append(segmenter.UnifiedParsingSegmenter(segsizes=[256],
            all_parts=all_parts,
            segdiv=('quad' if quad_seg else None)))
    if textures:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'texture')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="texture", segarch=("resnet18dilated", "ppm_deepsup")))
    if colors:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'color')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="color", segarch=("resnet18dilated", "ppm_deepsup")))
    if len(segmodels) == 1:
        segmodel = segmodels[0]
    else:
        segmodel = segmenter.MergedSegmenter(segmodels)
    seglabels = [l for l, c in segmodel.get_label_and_category_names()[0]]
    segcatlabels = segmodel.get_label_and_category_names()[0]
    return segmodel, seglabels, segcatlabels

# if __name__ == '__main__':
#     main()

