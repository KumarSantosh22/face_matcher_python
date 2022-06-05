def verify(img1_path, img2_path='', model_name='VGG-Face', distance_metric='cosine', model=None, enforce_detection=True, detector_backend='opencv', align=True, prog_bar=True, normalization='base')
This function verifies an image pair is same person or different persons.

Parameters:
        img1_path, img2_path: exact image path, numpy array or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.

        e.g. img1_path = [
                ['img1.jpg', 'img2.jpg'],
                ['img2.jpg', 'img3.jpg']
        ]

        model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble

        distance_metric (string): cosine, euclidean, euclidean_l2

        model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.

                model = DeepFace.build_model('VGG-Face')

        enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

        prog_bar (boolean): enable/disable a progress bar

Returns:
        Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.

        {
                "verified": True
                , "distance": 0.2563
                , "max_threshold_to_verify": 0.40
                , "model": "VGG-Face"
                , "similarity_metric": "cosine"
        }
Full name: deepface.DeepFace.verify