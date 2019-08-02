import helper_functions
import os
import cv2
import face_recognition
import pickle

# path pointing to the project directory
base_dir = os.path.dirname(os.path.abspath("__file__"))
output_embeddings = os.path.sep.join([base_dir, "model"])

if not os.path.exists(output_embeddings):
    os.makedirs(output_embeddings)

# Load Images
metadata = helper_functions.load_metadata('images')

imagePaths = [m.image_path() for m in metadata]

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    try :
        image = cv2.imread(imagePath)
        image = helper_functions.resize_image(image, 600)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
                                                model="hog")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)
    except:
        continue

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"embeddings": knownEncodings, "names": knownNames}
path = os.path.sep.join(["model", "embedding.pickle"])
f = open(path, "wb")
f.write(pickle.dumps(data))
f.close()