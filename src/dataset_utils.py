from PIL import Image
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import random
from utils import *
import itertools
from tqdm import tqdm
from datetime import datetime
def crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.ANTIALIAS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.ANTIALIAS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.ANTIALIAS)
    return source_resized

def original_crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.ANTIALIAS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.ANTIALIAS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.ANTIALIAS)
    return source_resized


def original_combine_and_mask(img_new, mask, img_black):
    """
    Combine img_new, mask, and image_black based on the mask

    img_new: new (unmasked image)
    mask: binary mask of bird image
    img_black: already-masked bird image (bird only)
    """
    # Warp new img to match black img
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized)

    # Mask new img
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

    # Combine
    img_combined_np = np.asarray(img_black) + img_masked_np
    img_combined = Image.fromarray(img_combined_np)

    return img_combined
def combine_and_mask(background, foreground_mask, masked_foreground, noise_level=0,
                     noise_type="gaussian", masked_constant_background_value_for_foreground_image=0):
    """
    background: img
    foreground_mask: float np
    masked_foreground: img
    """
    assert noise_type in ["gaussian", "background"]
    # Warp place img to match bird img
    masked_foreground_np = np.asarray(masked_foreground)

    background_resized = crop_and_resize(background, masked_foreground)
    background_resized_np = np.asarray(background_resized)

    # Mask new img
    background_mask_np = np.around(background_resized_np * (1 - foreground_mask)).astype(bool)
    foreground_mask_np = ~background_mask_np
    # add noise
    if noise_level > 0:
        foreground_mask = foreground_mask[:, :, 0]
        pos_mask = np.random.rand(foreground_mask.shape[0], foreground_mask.shape[1]) < noise_level

        if noise_type == "background":
            pos_mask = np.repeat(pos_mask[..., np.newaxis], 3, axis=2)
            foreground_foreground_pos_mask = (~pos_mask) & foreground_mask_np
            foreground_background_pos_mask = pos_mask & foreground_mask_np
            masked_foreground_np = foreground_foreground_pos_mask * masked_foreground + foreground_background_pos_mask * background_resized_np
        elif noise_type == "gaussian":
            pos_mask = pos_mask * foreground_mask
            pos_mask = np.repeat(pos_mask[..., np.newaxis], 3, axis=2)
            random_color_mask = np.around(np.random.rand(*masked_foreground_np.shape) * 255).astype(np.uint8)
            masked_foreground_np = np.where(pos_mask, random_color_mask, masked_foreground_np)
    # Combine
    img_combined_np = masked_foreground_np + background_mask_np * background_resized_np
    img_combined = Image.fromarray(img_combined_np)
    masked_foreground_np = np.where(masked_foreground_np > 0, masked_foreground_np,
                                    masked_constant_background_value_for_foreground_image)
    img_masked_foreground = Image.fromarray(masked_foreground_np)
    return img_combined, img_masked_foreground


WATER_BIRDS_LIST = {
    'Albatross',  # Seabirds
    'Auklet',
    'Cormorant',
    'Frigatebird',
    'Fulmar',
    'Gull',
    'Jaeger',
    'Kittiwake',
    'Pelican',
    'Puffin',
    'Tern',
    'Gadwall',  # Waterfowl
    'Grebe',
    'Mallard',
    'Merganser',
    'Guillemot',
    'Pacific_Loon'
}


def in_water_birds(s):
    if any([water_bird.lower() in s for water_bird in WATER_BIRDS_LIST]):
        return 1
    else:
        return 0


def construct_waterbirds_dataset(places_dir, cub_dir, output_dir,
                                 target_places,
                                 confounder_strength,
                                 val_frac, noise_type, noise_level,
                                 background_color_for_birds_images):
    os.makedirs(places_dir, exist_ok=True)
    os.makedirs(cub_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    assert noise_level >= 0 and noise_level < 1
    assert background_color_for_birds_images >= 0 and background_color_for_birds_images < 256
    assert confounder_strength >= 0 and confounder_strength <= 1
    assert val_frac >= 0 < 1

    # read cub csv
    cub_images_path = os.path.join(cub_dir, 'images.txt')

    df = pd.read_csv(
        cub_images_path,
        sep=" ",
        header=None,
        names=['img_id', 'img_filename'],
        index_col='img_id')

    # extract species name from file path recorded in the csv file of cub
    df["species"] = df["img_filename"].str.split('/', expand=True)[0]
    df["species"] = df["species"].str.split('.', expand=True)[1].str.lower()

    # create y column if y=0 -> landbird y=1 waterbirds
    df["y"] = df["species"].apply(in_water_birds)

    # read cub train test split metadata and create the split column
    # split=0->train ,1 -> val, 2->test
    train_test_df = pd.read_csv(
        os.path.join(cub_dir, 'train_test_split.txt'),
        sep=" ",
        header=None,
        names=['img_id', 'split'],
        index_col='img_id')

    df = df.join(train_test_df, on='img_id')
    test_ids = df.loc[df['split'] == 0].index
    train_ids = np.array(df.loc[df['split'] == 1].index)
    val_ids = np.random.choice(
        train_ids,
        size=int(np.round(val_frac * len(train_ids))),
        replace=False)

    df.loc[train_ids, 'split'] = 0
    df.loc[val_ids, 'split'] = 1
    df.loc[test_ids, 'split'] = 2

    # ---------note here we implicitly set place: 0->land 1->water----------
    df["place"] = df["y"].copy()

    # The dict is like {dataset_type:(dataset type corresponding split label, confounder strength)}
    datatype_id_confounder_strength_dict = {"train": (0, confounder_strength),
                                            "val": (1, 0.5),
                                            "test": (2, 0.5)}

    # modify training data to be spurious based on confounder strength
    # And set val and test dataset to be balanced
    for data_type, params in datatype_id_confounder_strength_dict.items():
        train_0_ids = np.array(df.loc[(df['split'] == params[0]) & (df["y"] == 0)].index)
        train_1_ids = np.array(df.loc[(df['split'] == params[0]) & (df["y"] == 1)].index)

        ids_0_to_1 = np.random.choice(train_0_ids, size=int(np.round((1 - params[1]) * len(train_0_ids))))
        ids_1_to_0 = np.random.choice(train_1_ids, size=int(np.round((1 - params[1]) * len(train_1_ids))))

        df.loc[ids_0_to_1, 'place'] = 1
        df.loc[ids_1_to_0, 'place'] = 0

        # record keeping
    with open(os.path.join(output_dir, "dataset group distribution.txt"), "a") as file:
        for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
            split_df = df.loc[df['split'] == split, :]
            file.writelines([f"{split_label}:",
                             f"waterbirds are {np.mean(split_df['y']):.3f} of the examples",
                             f"y = 0, c = 0: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}",
                             f"y = 0, c = 1: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}",
                             f"y = 1, c = 0: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}",
                             f"y = 1, c = 1: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}"
                             ])

    # read place df
    place_ids_df = pd.read_csv(
        os.path.join(places_dir, 'categories_places365.txt'),
        sep=" ",
        header=None,
        names=['place_name', 'place_id'],
        index_col='place_id')

    target_place_ids = []
    for idx, places in enumerate(target_places):
        place_filenames = []

        # check whether the input target places can be found in the csv
        # Add the corresponding idx of the place to the target_place_ids list
        for target_place in places:
            target_place_full = f"/{target_place[0]}/{target_place}"
            assert place_ids_df["place_name"].str.contains(target_place_full).sum() == 1
            target_place_ids.append(place_ids_df.index[place_ids_df['place_name'] == target_place_full][0])
            # add all file names under the corresponding target_places dir to list
            place_filenames += [
                f'/{target_place[0]}/{target_place}/{filename}' for filename in os.listdir(
                    os.path.join(places_dir, target_place[0], target_place))
                if filename.endswith('.jpg')]

        random.shuffle(place_filenames)

        # map each setted place id to a dinstinct sample of the place image
        # place idx is corresponding to the idx of target_places so 0 is land 1 is water
        assert len(place_filenames) >= np.sum(df['place'] == idx), \
            f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(df['place'] == idx)})"
        df.loc[df['place'] == idx, 'place_filename'] = place_filenames[:int(np.sum(df['place'] == idx))]
    # ------saving dataset---------------------------------
    # save metadata csv
    df.to_csv(os.path.join(output_dir, 'metadata.csv'))

    for i in tqdm.tqdm(df.index):
        # Load bird image and segmentation
        img_path = os.path.join(cub_dir, 'images', df.loc[i, 'img_filename'])
        seg_path = os.path.join(cub_dir, 'segmentations', df.loc[i, 'img_filename'].replace('.jpg', '.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255
        place_path = os.path.join(places_dir, df.loc[i, 'place_filename'][1:])
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img, masked_foreground_img = combine_and_mask(place, seg_np, img_black, noise_level=noise_level,
                                                               noise_type=noise_type,
                                                               masked_constant_background_value_for_foreground_image=background_color_for_birds_images)
        # masked_foreground_img.show()
        # combined_img.show()
        # bird_img = combine_and_mask(Image.fromarray(np.ones_like(place) * 150), seg_np, img_black)

        seg_np *= 0.
        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        place_img, _ = combine_and_mask(place, seg_np, img_black)

        combined_path = os.path.join(output_dir, "combined", df.loc[i, 'img_filename'])
        masked_foreground_path = os.path.join(output_dir, "birds", df.loc[i, 'img_filename'])
        place_path = os.path.join(output_dir, "places", df.loc[i, 'img_filename'])

        os.makedirs('/'.join(combined_path.split('/')[:-1]), exist_ok=True)
        os.makedirs('/'.join(masked_foreground_path.split('/')[:-1]), exist_ok=True)
        os.makedirs('/'.join(place_path.split('/')[:-1]), exist_ok=True)
        combined_img.save(combined_path)
        masked_foreground_img.save(masked_foreground_path)
        place_img.save(place_path)


def get_transform_cub(target_resolution, train, augment_data, flatten=False):
    # The reason for using 256.0 / 224.0 as the scale factor is to ensure that the resized image has a slightly
    # larger dimension than the target resolution before cropping. By multiplying the target resolution with this scale factor, the code ensures that the resized image is slightly larger in size. This allows for more flexibility during the cropping step, ensuring that the center crop can capture sufficient content from the resized image.
    # The specific value of 256.0 / 224.0 is used as a scaling factor to achieve this slight increase in size.
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if flatten:
        transform = transforms.Compose([
            transform,
            torch.flatten
        ])
    return transform


def log_data(logger, train_data, test_data, val_data=None, get_yp_func=None):
    logger.write(f'Training Data (total {len(train_data)})\n')
    # group_id = y_id * n_places + place_id
    # y_id = group_id // n_places
    # place_id = group_id % n_places
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')
    if val_data is not None:
        logger.write(f'Validation Data (total {len(val_data)})\n')
        for group_idx in range(val_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')


def add_noise_to_foreground(image, foreground_seg, percent):
    image_np = np.asarray(image)
    foreground_seg = foreground_seg[:, :, 0]
    pos_mask = np.random.rand(image_np.shape[0], image_np.shape[1]) < percent

    pos_mask = pos_mask * foreground_seg
    pos_mask = np.repeat(pos_mask[..., np.newaxis], 3, axis=2)
    random_color_mask = np.around(np.random.rand(*image_np.shape) * 255).astype(np.uint8)
    noisy_image = np.where(pos_mask, random_color_mask, image_np)
    return Image.fromarray(noisy_image)



def construct_spurious_boolean_dataset_from_sample_space(x_core,y_core, x_spurious_list, y_spurious_list,
                                       confounder_strength_list, sample_num, train_per=0.7, 
                                       test_per=0.3, randomize=True, F_2=True):
    ###if not F_2 change it to F_2###
    assert sample_num <= len(x_core)
    if not F_2:
        y_core[y_core==-1] = 0
        for y in y_spurious_list:
            y[y==-1] = 0
    if randomize:

        core_randomized_index = sample_index(x_core, sample_num,replacement=False)
        
        x_core, y_core = x_core[core_randomized_index], y_core[core_randomized_index]
        for x,y in zip(x_spurious_list, y_spurious_list):

            spurious_randomized_index = sample_index(x, sample_num, replacement=True)
            x, y = x[spurious_randomized_index], y[spurious_randomized_index]
    
    
    train_n = round(sample_num * train_per)
    test_n = sample_num - train_n
    x_core_train, y_core_train = x_core[:train_n], y_core[:train_n]
    x_core_test, y_core_test = x_core[train_n:], y_core[train_n:]

    x_spurious_train_list, x_spurious_test_list, y_spurious_train_list, y_spurious_test_list = [],[],[],[]
    for x,y in zip(x_spurious_list, y_spurious_list):
        x_spurious_train_list.append(x[:train_n])
        y_spurious_train_list.append(y[:train_n])
        x_spurious_test_list.append(x[train_n:]) 
        y_spurious_test_list.append(y[train_n:])
    
    x_core_train, y_core_train, x_spurious_train, y_spurious_train = mix_spurious_correlated_arrays(x_core_train, y_core_train, x_spurious_train_list,
                                                                                                                   y_spurious_train_list, confounder_strength=confounder_strength_list)
    x_core_test, y_core_test, x_spurious_test, y_spurious_test = mix_spurious_correlated_arrays(x_core_test, y_core_test, x_spurious_test_list, y_spurious_test_list
                                                                                                                ,confounder_strength=0.5)
    
    if not F_2:
        y_core_train[y_core_train==-1] = 0
        y_core_test[y_core_test==-1] = 0
        for y1, y2 in zip(y_spurious_train_list, y_spurious_test_list):
            y[y==-1] = 0
    return (x_core_train, y_core_train, x_spurious_train, y_spurious_train), (x_core_test, y_core_test, x_spurious_test, y_spurious_test)


def construct_spurious_boolean_dataset(core_feature_len, spurious_feature_len,
                                       core_feature_model, spurious_feature_model,
                                       sample_num, confounder_strength, train_per=0.7, test_per=0.3):
    assert sample_num < 2 ** min(core_feature_len, spurious_feature_len)
    core_sample_space, y_sample_space = get_sample_space(core_feature_model, core_feature_len)
    #change to False
    sampled_index = sample_index(core_sample_space, sample_num, replacement=True)
    
    x_core = core_sample_space[sampled_index]
    y_core = y_sample_space[sampled_index]
    
    train_n = round(sample_num * train_per)
    test_n = sample_num - train_n
    x_core_train, y_core_train = x_core[:train_n], y_core[:train_n]
    x_core_test, y_core_test = x_core[train_n:], y_core[train_n:]

    spurious_sample_space, spurious_sample_space_y = get_sample_space(spurious_feature_model,
                                                                      spurious_feature_len)
    spurious_sample_space_y = spurious_sample_space_y
    #change to False
    sampled_index = sample_index(spurious_sample_space, sample_num, replacement=True)
    x_spurious, y_spurious = spurious_sample_space[sampled_index], spurious_sample_space_y[sampled_index]
    x_spurious_train, y_spurious_train = x_spurious[:train_n], y_spurious[:train_n]
    x_spurious_test, y_spurious_test = x_spurious[train_n:], y_spurious[train_n:]
    
    x_core_train, y_core_train, x_spurious_train, y_spurious_train = mix_spurious_correlated_arrays(x_core_train, y_core_train, x_spurious_train,
                                                                                                                   y_spurious_train, confounder_strength=confounder_strength)
    x_core_test, y_core_test, x_spurious_test, y_spurious_test = mix_spurious_correlated_arrays(x_core_test, y_core_test, x_spurious_test, y_spurious_test
                                                                                                                ,confounder_strength=0.5)
    
    return (x_core_train, y_core_train, x_spurious_train, y_spurious_train), (x_core_test, y_core_test, x_spurious_test, y_spurious_test)
    
    

def mix_spurious_correlated_arrays(x_core, y_core, x_spurious_list, y_spurious_list, confounder_strength, unique_spurious=False):
    #the ordering of x_core, y_core is not changed so the result always has the same size as the given core arrays
    if type(confounder_strength) == list:
        assert len(confounder_strength) == len(x_spurious_list)
        confounder_strength_list = confounder_strength
    else:
        confounder_strength_list = [confounder_strength for _ in range(len(x_spurious_list))]
    y_0_size = (y_core == 0).sum().item()
    y_1_size = (y_core == 1).sum().item()

    res_x_spurious_list = []
    res_y_spurious_list = []
    for x_spurious, y_spurious, confounder_strength in zip(x_spurious_list, y_spurious_list, confounder_strength_list):
        group_0_1_size = round(y_0_size * (1 - confounder_strength))
        group_0_0_size = y_0_size - group_0_1_size
        group_1_0_size = round(y_1_size * (1 - confounder_strength))
        group_1_1_size = y_1_size - group_1_0_size

        
        y_spurious_new = y_core.clone()
        group_0_1_index = sample(torch.nonzero(y_core == 0), group_0_1_size, replacement=False)
        group_1_0_index = sample(torch.nonzero(y_core == 1), group_1_0_size, replacement=False)
        y_spurious_new[group_0_1_index] = 1
        y_spurious_new[group_1_0_index] = 0
        spurious_0_sample_needed_size = group_0_0_size + group_1_0_size
        spurious_1_sample_needed_size = group_1_1_size + group_0_1_size
        
        spurious_0 = x_spurious[y_spurious == 0]
        spurious_1 = x_spurious[y_spurious == 1]
        #print(spurious_0.shape, spurious_0_sample_needed_size)
        spurious_0_selected = sample(spurious_0, spurious_0_sample_needed_size, replacement=not unique_spurious)
        #print(spurious_1.shape, spurious_1_sample_needed_size)
        spurious_1_selected = sample(spurious_1, spurious_1_sample_needed_size, replacement=not unique_spurious)

        # create empty tensor for spurious features
        if len(x_spurious.shape) == 1:
            x_spurious = torch.zeros((x_core.shape[0])).type(torch.LongTensor)
        elif len(x_spurious.shape) == 2:
            x_spurious = torch.zeros((x_core.shape[0], x_spurious.shape[1])).type(torch.LongTensor)
        x_spurious[y_spurious_new == 0] = spurious_0_selected
        x_spurious[y_spurious_new == 1] = spurious_1_selected

        res_x_spurious_list.append(x_spurious)
        res_y_spurious_list.append(y_spurious_new)
    if len(x_spurious.shape) == 1:
        res_x_spurious = torch.concat(res_x_spurious_list)
    elif len(x_spurious.shape) == 2:
        res_x_spurious = torch.concat(res_x_spurious_list, dim=1)
    return x_core, y_core, res_x_spurious, res_y_spurious_list

class GrayscaleToRGB(object):
    def __call__(self, img):
        # Convert the single-channel (grayscale) image to a three-channel image
        img = img.convert('RGB')
        return img

class BinaryDatasetWrapper(Dataset):
    def __init__(self, dataset, class_labels=[]):
        super().__init__()
        assert len(class_labels) == 2
        self.dataset = dataset
        self.selected_index = []
        self.original_class_2_label = dict(zip(class_labels, [0,1]))
        for i, (x,y) in enumerate(dataset):
            if y in class_labels:
                self.selected_index.append(i)
    
    def __len__(self):
        return len(self.selected_index)
    
    def __getitem__(self, index):
        element = self.dataset[self.selected_index[index]]
        return element[0], torch.tensor([self.original_class_2_label[element[1]]])

def original_construct_waterbirds_dataset(output_dir, cub_dir, places_dir, confounder_strength, val_frac, dataset_name):

    ################ Paths and other configs - Set these #################################
    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds

    ######################################################################################

    images_path = os.path.join(cub_dir, 'images.txt')

    df = pd.read_csv(
        images_path,
        sep=" ",
        header=None,
        names=['img_id', 'img_filename'],
        index_col='img_id')

    ### Set up labels of waterbirds vs. landbirds
    # We consider water birds = seabirds and waterfowl.
    species = np.unique([img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']])
    water_birds_list = [
        'Albatross', # Seabirds
        'Auklet',
        'Cormorant',
        'Frigatebird',
        'Fulmar',
        'Gull',
        'Jaeger',
        'Kittiwake',
        'Pelican',
        'Puffin',
        'Tern',
        'Gadwall', # Waterfowl
        'Grebe',
        'Mallard',
        'Merganser',
        'Guillemot',
        'Pacific_Loon'
    ]

    water_birds = {}
    for species_name in species:
        water_birds[species_name] = 0
        for water_bird in water_birds_list:
            if water_bird.lower() in species_name:
                water_birds[species_name] = 1
    species_list = [img_filename.split('/')[0].split('.')[1].lower() for img_filename in df['img_filename']]
    df['y'] = [water_birds[species] for species in species_list]

    ### Assign train/tesst/valid splits
    # In the original CUB dataset split, split = 0 is test and split = 1 is train
    # We want to change it to
    # split = 0 is train,
    # split = 1 is val,
    # split = 2 is test

    train_test_df =  pd.read_csv(
        os.path.join(cub_dir, 'train_test_split.txt'),
        sep=" ",
        header=None,
        names=['img_id', 'split'],
        index_col='img_id')

    df = df.join(train_test_df, on='img_id')
    test_ids = df.loc[df['split'] == 0].index
    train_ids = np.array(df.loc[df['split'] == 1].index)
    val_ids = np.random.choice(
        train_ids,
        size=int(np.round(val_frac * len(train_ids))),
        replace=False)

    df.loc[train_ids, 'split'] = 0
    df.loc[val_ids, 'split'] = 1
    df.loc[test_ids, 'split'] = 2

    ### Assign confounders (place categories)

    # Confounders are set up as the following:
    # Y = 0, C = 0: confounder_strength
    # Y = 0, C = 1: 1 - confounder_strength
    # Y = 1, C = 0: 1 - confounder_strength
    # Y = 1, C = 1: confounder_strength

    df['place'] = 0
    train_ids = np.array(df.loc[df['split'] == 0].index)
    val_ids = np.array(df.loc[df['split'] == 1].index)
    test_ids = np.array(df.loc[df['split'] == 2].index)
    for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):
        for y in (0, 1):
            if split_idx == 0: # train
                if y == 0:
                    pos_fraction = 1 - confounder_strength
                else:
                    pos_fraction = confounder_strength
            else:
                pos_fraction = 0.5
            subset_df = df.loc[ids, :]
            y_ids = np.array((subset_df.loc[subset_df['y'] == y]).index)
            pos_place_ids = np.random.choice(
                y_ids,
                size=int(np.round(pos_fraction * len(y_ids))),
                replace=False)
            df.loc[pos_place_ids, 'place'] = 1

    for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
        print(f"{split_label}:")
        split_df = df.loc[df['split'] == split, :]
        print(f"waterbirds are {np.mean(split_df['y']):.3f} of the examples")
        print(f"y = 0, c = 0: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}")
        print(f"y = 0, c = 1: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}")
        print(f"y = 1, c = 0: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}")
        print(f"y = 1, c = 1: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}")

    ### Assign places to train, val, and test set
    place_ids_df = pd.read_csv(
        os.path.join(places_dir, 'categories_places365.txt'),
        sep=" ",
        header=None,
        names=['place_name', 'place_id'],
        index_col='place_id')

    target_place_ids = []

    for idx, target_places in enumerate(target_places):
        place_filenames = []

        for target_place in target_places:
            target_place_full = f'/{target_place[0]}/{target_place}'
            assert (np.sum(place_ids_df['place_name'] == target_place_full) == 1)
            target_place_ids.append(place_ids_df.index[place_ids_df['place_name'] == target_place_full][0])
            print(f'train category {idx} {target_place_full} has id {target_place_ids[idx]}')

            # Read place filenames associated with target_place
            place_filenames += [
                f'/{target_place[0]}/{target_place}/{filename}' for filename in os.listdir(
                    os.path.join(places_dir, target_place[0], target_place))
                if filename.endswith('.jpg')]

        random.shuffle(place_filenames)

        # Assign each filename to an image
        indices = (df.loc[:, 'place'] == idx)
        assert len(place_filenames) >= np.sum(indices),\
            f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(df.loc[:, 'place'] == idx)})"
        df.loc[indices, 'place_filename'] = place_filenames[:np.sum(indices)]

    ### Write dataset to disk
    output_subfolder = os.path.join(output_dir, dataset_name)
    os.makedirs(output_subfolder, exist_ok=True)

    df.to_csv(os.path.join(output_subfolder, 'metadata.csv'))

    for i in tqdm(df.index):
        # Load bird image and segmentation
        img_path = os.path.join(cub_dir, 'images', df.loc[i, 'img_filename'])
        seg_path = os.path.join(cub_dir, 'segmentations', df.loc[i, 'img_filename'].replace('.jpg','.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        # Skip front /
        place_path = os.path.join(places_dir,  df.loc[i, 'place_filename'][1:])
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = original_combine_and_mask(place, seg_np, img_black)

        output_path = os.path.join(output_subfolder, df.loc[i, 'img_filename'])
        os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)

        combined_img.save(output_path)

    # Getting the current date and time
    current_datetime = datetime.now()
    complete_sig = f"completed at {datetime.now()}"
    with open(os.path.join(output_subfolder, "completed"), "w") as file:
        file.write(complete_sig)
    
        
def check_dataset_completed(root):
    return os.path.exists(os.path.join(root, "completed"))
    