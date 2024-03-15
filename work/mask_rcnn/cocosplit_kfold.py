""" Based on https://github.com/akarazniewicz/cocosplit#readme.
    Modified to split annotations for k-fold cross validation.
"""

import json
import argparse
import funcy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def save_coco(file_path, info, licenses, images, annotations, categories):
    """ Saves COCO formatted annotiations in a JSON file at the file_path specified.
    """
    with open(file_path, 'w+', encoding='UTF-8') as coco:
        json.dump({ 
            'info': info,
            'licenses': licenses,
            'images': images, 
            'annotations': annotations,
            'categories': categories
        }, coco, indent=2, sort_keys=True)
        

def filter_annotations(annotations, images):
    """Filters annotations from master annotations file to only include annotations for the
    images specified.

    Args:
        annotations (list): A list of dicts representing COCO annotations.
        images (list): A list of dicts representing images.

    Returns:
        list: a list of dicts representing COCO annotations.
    """
    
    # get ID's for each image in images arg
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    
    # filter images
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, manual_images):
    """ Filters images to those that have manually calculated area. We can only use those with manual
        area results because the Mask-RCNN results must be compared to the manual results.
    
    Args:
        images (list): A list of dicts representing images.
        manual_images (list): A list of dicts representing images with manual calculations.

    Returns:
        list: a list of dicts representing COCO annotations.
    """
    
    images_with_manual_area = []
    images_without_manual_area = []
    
    for image in images:
        filename = image['file_name'].split('/')[-1].split('.')[0]
        
        if filename in manual_images:
            images_with_manual_area.append(image)
        else:
            images_without_manual_area.append(image)

    print('Filtered out', len(images_without_manual_area), 'images missing manual area calculations.')
    return images_with_manual_area


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('output_folder', type=str, help='Where to store split annotations')
parser.add_argument('-k', dest='k', type=int, required=True)
parser.add_argument('-p', dest='print_indices', action='store_true', help='Print indices for k-fold splits.')
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

args = parser.parse_args()

def main(args):
    """ Splits annotations for k-fold cross-validation.
    
    Because some images do not have associated manual measurements calculated by researcher, we remove
    these annotations before creating the splits.
    """
    
    # read excel file with manual area values
    df = pd.read_excel('/home/jovyan/work/data/manual_area.xlsx')
    
    # get filenames of images with manual area values without extension
    df['picture_split'] = df['picture'].str.split('.' , expand = True)[0]
    manual_images = df['picture_split'].values
    
    # open annotations file
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        
        coco = json.load(annotations)
        info = coco.get('info', {})
        licenses = coco.get('licenses', [])
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)
        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        
        
        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)
        
        # filter out images without manual results
        images = filter_images(images, manual_images)
        print('Final dataset size:', len(images), 'images')
    
        # convert to numpy array
        numpy_images = np.array(images)
        
        # get indicies for images in k-fold splits
        kf = KFold(n_splits=args.k, shuffle=True, random_state=7)
        split_indices = kf.split(images)
        
        
        for index, value in enumerate(split_indices):
            
            train_indices = value[0]
            test_indices = value[1]
            
            train_images = numpy_images[train_indices].tolist()
            test_images = numpy_images[test_indices].tolist()
            
            print("\nFold", index) 
            print("Training set size:", len(train_images), "\nTest set size:", len(test_images))
            
            if args.print_indices:
                print("Train indicies:", train_indices)
                print("Test indicies:", test_indices)
            
            train_annotations = filter_annotations(annotations, train_images)
            test_annotations = filter_annotations(annotations, test_images)
            
            save_coco(
                args.output_folder + '/train_' + str(index) + '.json',
                info,
                licenses,
                train_images,
                train_annotations,
                categories
            )
            
            save_coco(
                args.output_folder + '/test_' + str(index) + '.json',
                info,
                licenses,
                test_images,
                test_annotations,
                categories
            )
            
#             print("Saved {} entries in {} and {} in {}".format(
#                 len(train_annotations), args.train, len(test_annotations), args.test)
#             )      


if __name__ == "__main__":
    main(args)