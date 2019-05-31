import bz2
import os
from urllib.request import urlopen
import numpy as np
import cv2
from align import align_image
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import face_recognition

def resize_image(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized




def download_landmarks(dst_file):
    """
     Downloads landmarks to execute face alignment
    """
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()

    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)


def download_landmarks_to_disk(dst_dir):

    dst_file = os.path.join(dst_dir, 'landmarks.dat')
    if not os.path.exists(dst_file):
        os.makedirs(dst_dir)
        download_landmarks(dst_file)


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return metadata


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]


def generate_embeddings(metadata, pretrained_network, load_from_disk=False, embedings_output_path="" ):
    """
    Inputs the metadata through the network to generate a set of encodings/embeddings
    """

    embedded = np.zeros((metadata.shape[0], 128))
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if embedings_output_path == "":
        if not os.path.exists(os.path.sep.join([base_dir, "models"])):
            os.mkdir(os.path.sep.join([base_dir, "models"]))
        embeddings_disk = os.path.sep.join([base_dir, "models", "embeddings.pickle"])

    else:
        embeddings_disk = embedings_output_path

    if load_from_disk:
        pickleFile = open(embeddings_disk, 'rb')
        embedded = pickle.load(pickleFile)
        correct_indexes = pickle.load(pickleFile)
        return embedded, correct_indexes

    else:
        error_indexes = []
        for i, m in enumerate(metadata):

            try:
                img = load_image(m.image_path())
                # Align face to boost the model performance
                img = align_image(img)
                # scale RGB values to interval [0,1]
                img = (img / 255.).astype(np.float32)
                # obtain embedding vector for image
                embedded[i] = pretrained_network.predict(np.expand_dims(img, axis=0))[0]
                print(f"Generating embedding for image {m.image_path()}")
            except TypeError:
                print(f"Failed to generate embedding for image {m.image_path()}")
                error_indexes.append(i)

        correct_indexes = [i for i in range(len(metadata)) if i not in error_indexes]
        embedded = embedded[correct_indexes]
        pickleFile = open(embeddings_disk, 'wb')
        pickle.dump(embedded, pickleFile)
        pickle.dump(correct_indexes, pickleFile)
        pickleFile.close()

        return embedded,correct_indexes

def generate_embedding_from_image(image, pretrained_network):

    embedding = None
    try:
        image = align_image(image)
        # scale RGB values to interval [0,1]
        img = (image / 255.).astype(np.float32)
        # obtain embedding vector for image
        embedding = pretrained_network.predict(np.expand_dims(img, axis=0))[0]
        print(f"Generating embedding for image")
    except:
        print("Could not generate embedding")
    return embedding

def train_models(embeddings, metadata, models_output_path="" ):
    """
    Takes as input a set of encodings and metadata containing their labels to train a KNN and SVM model
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if models_output_path == "":
        if not os.path.exists(os.path.sep.join([base_dir, "models"])):
            os.mkdir(os.path.sep.join([base_dir, "models"]))
        models_disk = os.path.sep.join([base_dir, "models", "models.pickle"])

    else:
        models_disk = models_output_path

    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    train_idx, test_idx = train_test_split(np.arange(metadata.shape[0]), test_size = 0.3, random_state = 42)
    # 50 train examples of 10 identities (5 examples each)
    X_train = embeddings[train_idx]
    # 50 test examples of 10 identities (5 examples each)
    X_test = embeddings[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()

    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))

    print(f'KNN accuracy = {acc_knn * 100}%, SVM accuracy = {acc_svc * 100}%')

    pickleFile = open(models_disk, 'wb')
    pickle.dump(knn, pickleFile)
    pickle.dump(svc, pickleFile)
    pickleFile.close()

    return knn, svc




def recognize_faces_in_frame(image):


    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    boxes = face_recognition.face_locations(rgb,
                                            model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []
    probs = []
    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(os.path.sep.join(["model", "recognizer.pickle"]), "rb").read())
    le = pickle.loads(open(os.path.sep.join(["model", "le.pickle"]), "rb").read())


    for index, face in enumerate(boxes):

        vec = face_recognition.face_encodings(rgb, [boxes[index]])
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j] if proba > 0.6 else "unknown"
        names.append(name)
        probs.append(proba)

    detections = zip(boxes, names, probs)
    return detections



def show_image(image):
    # show the output image
    cv2.imshow("Press any key to close this image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_faces_in_image(image):


    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    boxes = face_recognition.face_locations(rgb,
                                            model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []
    probs = []
    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(os.path.sep.join(["model", "recognizer.pickle"]), "rb").read())
    le = pickle.loads(open(os.path.sep.join(["model", "le.pickle"]), "rb").read())


    for index, face in enumerate(boxes):

        vec = face_recognition.face_encodings(rgb, [boxes[index]])
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j] if proba > 0.6 else "unknown"
        names.append(name)
        probs.append(proba)


    # loop over the recognized faces
    for ((top, right, bottom, left), name, prob) in zip(boxes, names, probs):
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        label = f"{name} prob : {round(prob, 3)*100}%"
        cv2.putText(image, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    return image