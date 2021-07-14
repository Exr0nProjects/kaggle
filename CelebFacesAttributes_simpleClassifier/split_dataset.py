import csv
import shutil
from tqdm import tqdm
from os import mkdir
from operator import itemgetter

DATASET_DIR = 'data'
SPLITS = ( 'train', 'dev', 'test' )

try:
    for split in SPLITS:
        mkdir(f"{DATASET_DIR}/{split}")

    with open(f"{DATASET_DIR}/list_eval_partition.csv") as rf:
        for line in csv.DictReader(rf):
            file, split = itemgetter('image_id', 'partition')(line)
            split = SPLITS[int(split)]
            shutil.move(f"{DATASET_DIR}/img_align_celeba/img_align_celeba/{file}", f"{DATASET_DIR}/{split}/{file}")
except FileExistsError:
    print("splits directories already exist")

with open(f"{DATASET_DIR}/list_attr_celeba.csv", 'r') as rf, open(f"{DATASET_DIR}/list_eval_partition.csv", 'r') as splitf:
    keys = "image_id,5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young".split(',')
    files = [open(f'{DATASET_DIR}/{split}/attr.csv', 'w+') for split in SPLITS]
    writers = [csv.DictWriter(f, fieldnames=keys) for f in files]
    for w in writers:
        w.writeheader()

    for line, split in tqdm(zip(csv.DictReader(rf), csv.DictReader(splitf))):
        file, split = itemgetter('image_id', 'partition')(split)
        split = int(split)
        writers[split].writerow(line)

    for file in files:
        file.close()

