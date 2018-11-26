import image_generator as ig
import utils
import cv2
import os
from data_model import ResultRow
import pandas as pd


def generate_dataset(logo_path, background_dir, output_dir):
    bgs = utils.load_filepaths_in_directory(background_dir)
    logo = utils.load_image(image_path=logo_path, mode=cv2.IMREAD_UNCHANGED)

    rows = []

    label = os.path.basename(logo_path).split('.')[0]

    for ind, background in enumerate(bgs):
        back = utils.load_image(background, cv2.IMREAD_UNCHANGED)
        output_path = output_dir + "/" + label + '_' + str(ind) + ".png"
        params = ig.transform_image(logo, back)
        image = params[0]
        utils.save_image(image, output_path)
        row = ResultRow(output_path, params[1], params[2], params[3], params[4], label)
        rows.append(row)

    save_data(rows, os.path.join(output_dir, label + '_data.csv'))


def save_data(list_of_objects, output_path):
    df = pd.DataFrame.from_records([row.to_dict() for row in list_of_objects])
    df.to_csv(output_path, index=False)
