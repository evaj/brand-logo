import dataset_generator

LOGO_DIR = 'C:/Users/centu/Pictures/loga/biedronka_full.png'
BACKGROUND_DIR = 'C:/Users/centu/Pictures/train2014'
OUTPUT_DIR = 'C:/Users/centu/Pictures/transformed_bf'

dataset_generator.generate_dataset(LOGO_DIR, BACKGROUND_DIR, OUTPUT_DIR)