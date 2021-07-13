import os

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
from model import create_model, TripletLossLayer
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import confusion_matrix
from align import AlignDlib
from train import train_model

from data import triplet_generator
from sklearn.metrics import f1_score, accuracy_score
from utils import load_metadata, load_image, download_landmarks

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from time import time
import warnings


dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)

nn4_small2_train = create_model()
nn4_small2_train.load_weights('weights/nn4.small2.v1.h5')



metadata = load_metadata('images')

#Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

#Load an image
jc_orig = load_image(metadata[0].image_path())

#Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

#Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

#Show original image
plt.subplot(131)
plt.imshow(jc_orig)

#Show original image with bounding box
plt.subplot(132)
plt.imshow(jc_orig)
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

#Show aligned image
plt.subplot(133)
plt.imshow(jc_aligned)
plt.show()


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


embedded = np.zeros((metadata.shape[0], 128))
for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    #scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    #img = (img * 255).round().astype(np.float32)
    #obtain embedding vector for image
    embedded[i] = nn4_small2_train.predict(np.expand_dims(img, axis=0))[0]


#Verify
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));

#Pair two images
show_pair(0 , 1)
show_pair(0, 11)
plt.show()

distances = [] #squared L2 distance between pairs
identical = [] #1 if same identity, 0 otherwise

num = len(metadata)

for i in range(num - 1):
    for j in range(1, num):
        distances.append(distance(embedded[i], embedded[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)

#Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
#Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

#Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score')
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
plt.xlabel('Distance threshold')
plt.legend()

dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.hist(dist_pos)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (pos. pairs)')
plt.legend()

plt.subplot(122)
plt.hist(dist_neg)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (neg. pairs)')
plt.legend()


dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.hist(dist_pos)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (pos. pairs)')
plt.legend()

plt.subplot(122)
plt.hist(dist_neg)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (neg. pairs)')
plt.legend()
plt.show()



targets = np.array([m.name for m in metadata])

encoder = LabelEncoder()
encoder.fit(targets)
#Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

#205 train examples of 41 identities (5 examples each)
X_train = embedded[train_idx]
#205 test examples of 41 identities (5 examples each)
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
svc = LinearSVC()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))




print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')

#Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

example_idx = 30  #Check why get Brad Pitt ever

#example_image = load_image('images/bellucci.jpg')
example_image = load_image(metadata[test_idx][example_idx].image_path())
bb = alignment.getLargestFaceBoundingBox(example_image)
example_prediction = svc.predict([embedded[test_idx][example_idx]])
example_identity = encoder.inverse_transform(example_prediction)[0]
print(example_identity)

plt.imshow(example_image)
plt.title(f'Recognized as {example_identity} using SVM')
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
plt.show()



t0 = time()
y_pred = svc.predict(X_test)
print("done in %0.3fs" % (time() - t0))

n_classes = metadata.shape[0]

tg = [
      "Alessandro_Gassmann",
      "Ariel_Sharon",
      "Arnold_Schwarzenegger",
      "Ben_Affleck",
      "Brad_Pitt",
      "Bradley_Cooper",
      "Chris_Hemsworth",
      "Christian_DeSica",
      "Colin_Powell",
      "Daniel_Radcliffe",
      "Donald_Rumsfeld",
      "George_Clooney",
      "George_W_Bush",
      "Gerhard_Schroeder",
      "Halle_Berry",
      "Harrison_Ford",
      "Hugo_Chavez",
      "Jacques_Chirac",
      "Jennifer_Aniston",
      "Johnny_Depp",
      "Julia_Roberts",
      "KimRossi_Stuart",
      "Leonardo_Dicaprio",
      "Luca_Argentero",
      "Massimo_Boldi",
      "Matt_Damon",
      "Matthew_Lewis",
      "Michelle_Pfeiffer",
      "Monica_Bellucci",
      "Nicole_Kidman",
      "Orlando_Bloom",
      "Patrick_Dempsey",
      "Raoul_Bova",
      "Richard_Gere",
      "Robert_DownayJr",
      "Sandra_Bullock",
      "Tom_Cruise",
      "Tony_Blair",
      "Vladimir_Putin",
      "Will_Smith",
      "TEST"
      ]

print(classification_report(y_test, y_pred, target_names=tg))
#print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))