from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

'''
    model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble
    distance_metric (string): cosine, euclidean, euclidean_l2
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

    PARAMETERS
        verify(img1_path, img2_path='', model_name='VGG-Face', distance_metric='cosine', model=None, enforce_detection=True, detector_backend='opencv', align=True, prog_bar=True, normalization='base')
'''

BACKENDS = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
MODELS = ['VGG-Face', 'Facenet', 'OpenFace',
          'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'Ensemble']
DISTANCE_METRICES = ['cosine', 'euclidean', 'euclidean_l2']
THRESHOLD = 0.4

imgs = [
    's/b.jpg',
    's/c.jpg',
    's/d.jpg',
    's/a.jpg',
    's/f.jpg',
    's/g.jpg',
    's/h.jpg',
    's/i.jpg',
    's/j.jpg',
    's/1.jpg',
    's/2.jpg',
    's/3.jpg',
    's/4.jpg',
    'a.jpg',
    'b.jpg',
    'c.jpg',
    'd.jpg',
    'e.jpg',
]

# img1 = cv2.imread('D:\WS-Santosh\SampleImages/s/h.jpg')
# # plt.imshow(img1[:,:,::-1])
# # plt.show()
# img2 = cv2.imread('D:\WS-Santosh\SampleImages/s/a.jpg')
# # plt.imshow(img2[:,:,::-1])
# # plt.show()


def faceMatch(src: str, dest: str):
    destFile = 'D:\WS-Santosh\SampleImages/' + dest
    result = DeepFace.verify(src, destFile,  model_name='Facenet',
                             distance_metric=DISTANCE_METRICES[0], detector_backend='mtcnn')
    # result = DeepFace.verify(img1, img2, model_name=MODELS[1], detector_backend='mtcnn', distance_metric='euclidean_l2')
    # print(result)
    dist = result['distance']
    score = (1-dist) * 100
    matched = True if THRESHOLD > dist else False
    print(f'{dest}: Score: {score:.2f} %, Distance: {dist:2f}, Verified: {result["verified"]}, Matched: {matched}')


def sentiments():
    obj = DeepFace.analyze(img_path="D:\WS-Santosh\SampleImages/c.jpg",
                           actions=['age', 'gender', 'race', 'emotion'])
    print(obj)


for img in imgs:
    faceMatch(src="D:\WS-Santosh\SampleImages/1.jpg", dest=img)
