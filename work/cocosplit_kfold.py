
import json
import argparse
import funcy
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 
            'info': info,
            'licenses': licenses,
            'images': images, 
            'annotations': annotations,
            'categories': categories
        }, coco, indent=2, sort_keys=True)
        

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('output_folder', type=str, help='Where to store split annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('-k', dest='k', type=int, required=True)
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')


args = parser.parse_args()

def main(args):

    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco.get('info', {})
        licenses = coco.get('licenses', [])
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        numpy_images = np.array(images)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)
            
        kf = KFold(n_splits=args.k, shuffle=True, random_state=7)
        split_indices = kf.split(images)
        print(split_indices)
        
        for index, value in enumerate(split_indices):
            train_indices = value[0]
            test_indices = value[1]
            
            train_images = numpy_images[train_indices].tolist()
            test_images = numpy_images[test_indices].tolist()
             
            # print("\n\nTRAIN:", train_images, "\nTEST:", test_images)
            
            anns_train = filter_annotations(annotations, train_images)
            anns_test = filter_annotations(annotations, test_images)
            
            print(isinstance(train_images, np.ndarray))
            print(isinstance(anns_test, np.ndarray))
            
            
            # print(test_images)
            
            save_coco(
                args.output_folder + '/train_' + str(index) + '.json',
                info,
                licenses,
                train_images,
                anns_train,
                categories
            )
            
            save_coco(
                args.output_folder + '/test_' + str(index) + '.json',
                info,
                licenses,
                test_images,
                anns_test,
                categories
            )
            
#             print("Saved {} entries in {} and {} in {}".format(
#                 len(anns_train), args.train, len(anns_test), args.test)
#             )      


if __name__ == "__main__":
    main(args)