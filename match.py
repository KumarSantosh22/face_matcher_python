from deepface import DeepFace

'''
This file has been using in api.py
'''

THRESHOLD = 0.4


def faceMatch(srcFile: str, destFile: str):
    result = DeepFace.verify(srcFile, destFile,  model_name='Facenet',
                             distance_metric='cosine', detector_backend='mtcnn')
    dist = result['distance']
    score = (1-dist) * 100
    matched = True if THRESHOLD > dist else False
    obj = {
        "score": f"{score:.2f} %",
        "distance": f"{dist:2f}",
        "matched": f"{matched}",
        "verified": f"{result['verified']}"
    }
    return obj


if __name__ == '__main__':
    print(faceMatch("D:\WS-Santosh\SampleImages/1.jpg",
        "D:\WS-Santosh\SampleImages/1.jpg"))
