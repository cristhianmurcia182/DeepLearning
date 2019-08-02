import os
import cv2
import pickle
import helper_functions
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--option", required=True,
                help="This parameter can be either i or f. "
                     "Type i if you want to recognize faces in a single image, type f if you want "
                     "to recognize faces inside a folder containing several images.")

ap.add_argument("-p", "--path", required=True,
                help="Path pointing to the image(s) to be recognized. If you typed i as option"
                     " it is expected to be a path pointing to a single file, if you typed f "
                     "it is expected to be a path pointing to a folder."
                )
args = vars(ap.parse_args())

data = pickle.loads(open(os.path.sep.join(["model", "embedding.pickle"]), "rb").read())

#image_path = "test/1.jpg"
# load the input image and convert it from BGR to RGB

if args["option"] == "i":
    image_path = args["path"]
    try:
        print(f"Recognizing faces in {image_path}")
        image = cv2.imread(image_path)
        image = helper_functions.resize_image(image, 1000)
        image = helper_functions.recognize_faces_in_image(image)
        helper_functions.show_image(image)
    except:
        print(f"Could not open input image {image_path}")

elif args["option"] == "f":
    image_paths = os.listdir(args["path"])
    for image_path in  image_paths:
        try:
            image_path = os.path.sep.join([args["path"], image_path])
            image = cv2.imread(image_path)
            image = helper_functions.resize_image(image, 1000)
            print(f"Recognizing faces in {image_path}")
            image = helper_functions.recognize_faces_in_image(image)
            helper_functions.show_image(image)
        except:
            print(f"Could nodasat open input image {image_path}")
            print(f"Could nodasat open input image {image_path}")

else:
    print("Unknown argument it shoul be either i or f, type -h to get help.")


