VERSION_STR = 'v1.0.0'

import cv2
import base64
import requests
import numpy as np
from error import *
from flask import Blueprint, request, jsonify
from PIL import Image
import uuid
import shutil
import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
import time
import os
import re
import resnet
from image_processing import image_preprocessing
import argparse
from sklearn.externals import joblib

import argparse
from LSTM import LSTM

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                               'Must divide evenly into the dataset sizes.')

INDIVIDUAL_MODEL_PATH = "models/individual_5netsRRDE_SVR.pkl"
GROUP_MODEL_PATH = "models/weighted_meaning_encoding_group_5netsRRDE_SVR.pkl"
RRDE_MODEL_PATH = "models/5nets_model.npy"
LSTM_MODEL_PATH = "models/5nets_lstm_model.npy"

NUMBER_OF_NETS = 5

# load models and resources
INDIVIDUAL_SVR = joblib.load(INDIVIDUAL_MODEL_PATH)
GROUP_SVR = joblib.load(GROUP_MODEL_PATH)
FACE_CASCADE = cv2.CascadeClassifier("resources/cascades/haarcascades/haarcascade_frontalface_alt.xml")

blueprint = Blueprint(VERSION_STR, __name__)

# execute mongoDB in terminal
# db_cilent = MongoClient()
# db = db_cilent['mememoji']
# collection = db['photoinfo']

def base64_encode_image(image_rgb):
    """Encode image in base64 format
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ret, image_buf = cv2.imencode('.jpg', image_bgr, (cv2.IMWRITE_JPEG_QUALITY, 40))
    image_str = base64.b64encode(image_buf)
    return 'data:image/jpeg;base64,' + image_str


def normalME_group_estimate(FaceInfo):
    """Normal mean-encoding based group estimation
    """

    features = np.asarray([FaceInfo[i]['LSTM_aggregateed_feature'][0] for i in range(0, len(FaceInfo))])
    mean_encoding = np.sum(features, 0) / len(features)

    return (GROUP_SVR.predict(mean_encoding))[0]


def individual_estimate(FaceInfo):
    for i in range(0, len(FaceInfo)):
        individual_estimation = INDIVIDUAL_SVR.predict(FaceInfo[i]['LSTM_aggregateed_feature'][0])
        FaceInfo[i]['happiness_intensity'] = individual_estimation[0]


def LSTM_aggregate(FaceInfo):
    """Aggregate features using pre-trained LSTM
    """

    # load LSTM model
    tf.reset_default_graph() # clear
    sess = tf.Session()
    LSTM_MODEL = LSTM(NUMBER_OF_NETS, sess)
    LSTM_MODEL.inference([]) # ???
    LSTM_MODEL.assign_weights(LSTM_MODEL_PATH)

    features = np.asarray([FaceInfo[i]['feature'] for i in range(0, len(FaceInfo))])
    LSTM_aggregateed_features = LSTM_MODEL.extract(features, [], [])
    for i in range(0, len(LSTM_aggregateed_features)):
        print(str(i) + " face LSTM feature extracted" )
        FaceInfo[i]['LSTM_aggregateed_feature'] = LSTM_aggregateed_features[i]

    return LSTM_aggregateed_features


def parse_args():
    """Utility function for extract_features
    """
    parser = argparse.ArgumentParser(description='group emotion analysis')
    parser.add_argument('--net_number', dest='number_of_nets', default=1, type=int)
    parser.add_argument('--test_set', dest='test_set', type=str)
    parser.add_argument('--model_path', dest='model_path', type=str)
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--loss', dest='loss_function', type=str)
    return parser.parse_args()


def extract_features(FaceInfo, SESSION_ID):
    """Extract face feature representation using pre-trained ResNet-20

    Args:
    FaceInfo: an array of face information
    SESSION_ID: id for the session
    """
    tf.reset_default_graph()  # clear
    sess = tf.Session()
    args = parse_args()
    args.number_of_nets = 5
    args.data_path = "temp/" + str(SESSION_ID) + "/"  # path to images
    args.loss = 'softmax'
    args.model_path = RRDE_MODEL_PATH
    args.file_names = [str(x) + ".jpg" for x in range(0, len(FaceInfo))]

    images = tf.placeholder(tf.float32, shape=[1,FLAGS.input_size,FLAGS.input_size,3])

    print("get net starts")
    import get_net
    is_training = tf.placeholder(tf.bool)
    logits = get_net.preprocessing(args, images, 'feature', is_training)
    print("get net finished")

    # load RRDE model
    print("load RRDE model"),
    init = tf.initialize_all_variables()
    sess.run(init)
    data=np.load(args.model_path).item()
    op=[]
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        op.append(v.assign(data[v.name]))
        print("."),
    sess.run(op)
    print("\nRRDE model loaded")

    features=[]
    file_names = args.file_names
    import data_input
    ims=data_input.get_filequeue([args.data_path + f for f in file_names]) # get all images for processing

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    # ii=0
    # while ii<len(FaceInfo):
    #     im=sess.run(ims)
    #     print(ii)
    #     feature=[]
    #     for logit in logits:   # for logit in logits:
    #         out=sess.run(logit,{is_training:False,images:im})
    #         feature.append(out)
    #     FaceInfo[ii]['feature'] = feature
    #     ii+=1
    # np.save('features/temp.npy', features)

    print("extract features")
    for logit, i in zip(logits, range(1, args.number_of_nets + 1)):
        ii = 0
        print("i: " + str(i))
        while ii < len(FaceInfo):
            im = sess.run(ims)
            print ii
            out = sess.run(logit, {is_training: False, images: im})
            if i == 1:
                FaceInfo[ii]['feature'] = [out]
                features.append({'feature':[out]})
            else:
                FaceInfo[ii]['feature'].append(out) # stack feature vector together e.g. if there are 5 nets in RRDE, the dimession of feature is 64*5
                features[ii]['feature'].append(out)
            ii += 1

        np.save('features/' + '%dnets_feature.npy' % i, features)

    # sess.close()
    print("extraction done")


def detect_face(image_gray, image_rgb, annotated_rgb, crop_faces, SESSION_ID):
    """Detect fases in the given images

    Args:
    image_gray: gray image in cv2 img format
    image_rgb: rgb images in cv2 img format
    annotated_rgb: annoated image after detection
    crop_faces: flag indicating whether to have faces cropped
    SESSION_ID: id for the session
    """

    os.mkdir("temp/" + str(SESSION_ID))
    faces = FACE_CASCADE.detectMultiScale(image_gray,
										  scaleFactor = 1.3,
										  minNeighbors=3,
                                          minSize=(45, 45),
                                          flags = cv2.CASCADE_SCALE_IMAGE)

    face_color = [0, 0, 255] #blue
    thickness = 4
    FaceInfo = []
    index = 0

    for x_face, y_face, w_face, h_face in faces:
        faceinfo = {'index': index}
        faceinfo['location_xy'] = (int(x_face), int(y_face))
        faceinfo['width'] = int(w_face)
        faceinfo['height'] = int(h_face)

        # face_image_gray = image_gray[y_face : y_face + h_face,
        #                              x_face : x_face + w_face]

        face_image_rgb = image_rgb[y_face: y_face + h_face,
                         x_face: x_face + w_face]

        if crop_faces:
            # crop_image = cv2.resize(face_image_rgb, (40, 50))
            crop_image = face_image_rgb
            faceinfo['thumbnail'] = base64_encode_image(crop_image)

        if annotated_rgb != None: # opencv drawing the box
            cv2.rectangle(annotated_rgb, (x_face, y_face),
                (x_face + w_face, y_face + h_face),
                face_color, thickness)

        # save detected faces
        pil_im = Image.fromarray(face_image_rgb)
        pil_im.save("temp/" + str(SESSION_ID) + "/" + str(index) + ".jpg")

        FaceInfo.append(faceinfo)
        index += 1

    return FaceInfo


def run_RRDE(image_gray, image_rgb, annotated_rgb, crop_faces):
    '''
    Conduct group-level happiness intensity estimation based on RRDE framework

    Args:
    image_gray: gray image in cv2 img format
    image_rgb: rgb images in cv2 img format
    annotated_rgb: annoated image after detection
    crop_faces: flag indicating whether to have faces cropped
    '''

    print('run_RRDE activated')
    photoinfo = {}
    SESSION_ID = uuid.uuid1()

    # detect faces
    FaceInfo = detect_face(image_gray, image_rgb, annotated_rgb, crop_faces, SESSION_ID)
    if annotated_rgb != None:
        photoinfo['annotated_image'] = base64_encode_image(annotated_rgb)

    # extract face features
    extract_features(FaceInfo, SESSION_ID)
    LSTM_aggregate(FaceInfo)

    # estimate happiness intensity
    individual_estimate(FaceInfo)
    normalME_group_estimation = normalME_group_estimate(FaceInfo)

    # convert array to list
    for i in range(0, len(FaceInfo)):
        features = FaceInfo[i]['feature']
        features_list = []
        for feature in features:
            features_list.append(feature[0].tolist())

        FaceInfo[i]['feature'] = features_list

        aggregated_feature = FaceInfo[i]['LSTM_aggregateed_feature']
        FaceInfo[i]['LSTM_aggregateed_feature'] = aggregated_feature[0].tolist()

    photoinfo['faces'] = FaceInfo
    photoinfo['id'] = SESSION_ID
    photoinfo['normalME_group_estimation'] = normalME_group_estimation

    # remove temp files
    shutil.rmtree("temp/" + str(SESSION_ID))

    return photoinfo


def obtain_images(request):
    '''Obtain the image from http request. It throws an error if the image cannot be obtained.
    '''
    print("obtain_images activated")
    if 'image_url' in request.args:
        print('this is image_url')
        image_url = request.args['image_url']
        try:
            response = requests.get(image_url)
            encoded_image_str = response.content
        except:
            raise Error(2873, 'Invalid `image_url` parameter')

    elif 'image_buf' in request.files:
        image_buf = request.files['image_buf']
        encoded_image_str = image_buf.read()

    elif 'image_base64' in request.args:
        print('this is image_base64')
        image_base64 = request.args['image_base64']

        ext, image_str = image_base64.split(';base64,')
        try:
            encoded_image_str = base64.b64decode(image_str)
        except:
            raise Error(2873, 'Invalid `image_base64` parameter')

    else:
        raise Error(35842, 'You must supply either `image_url` or `image_buf`')

    if encoded_image_str == '':
        raise Error(5724, 'You must supply a non-empty input image')

    encoded_image_buf = np.fromstring(encoded_image_str, dtype=np.uint8)
    decoded_image_bgr = cv2.imdecode(encoded_image_buf, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(decoded_image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    annotate_image = (request.args.get('annotate_image', 'false').lower() == 'true')
    if annotate_image:
        annotated_rgb = np.copy(image_rgb)
    else:
        annotated_rgb = None
    crop_image = (request.args.get('crop_image', 'false').lower() == 'true')
    if crop_image:
        crop_faces = True
    else:
        crop_faces = False
    return image_rgb, image_gray, annotated_rgb, crop_faces


def obtain_feedback(request):
    '''Obtain the feedback from http request.
    '''
    feedback = {}
    if 'image_id' in request.args:
        feedback['id'] = request.args['image_id']
    else:
        raise Error(2873, 'No `image_id` provided')

    if 'face_index' in request.args:
        feedback['face_index'] = request.args['face_index']

    if 'feedback' in request.args:
        if request.args['feedback'] in emotions:
            feedback['feedback'] = request.args['feedback']
        else:
            raise Error(2873, 'Invalid `feedback` parameter')

    # insert = collection.update({"pic_id": feedback['id'], "faces.index": int(feedback['face_index'])},
    #                            {"$push": {"face.index.$.feedback": feedback['feedback']}})
    # print "INSERT STATUS: ", insert
    return feedback


@blueprint.route('/predict', methods=['POST'])
def predict():
    '''
    Detect faces in the image and predict both individual-level and group-level happiness intensity
    Detect faces, extract highly efficient features, predict happiness intensity and provide an annotated image and thumbnails of predicted faces.
    ---
    tags:
      - v1.0.0

    responses:
      200:
        description: An image info object
        schema:
          $ref: '#/definitions/PhotoInfo'
      default:
        description: Unexpected error
        schema:
          $ref: '#/definitions/Error'

    parameters:
      - name: image_base64
        in: query
        description: A base64 string of an image taken via webcam or photo uploading. This field must be specified, the user must pass an image via the `image_base64` form parameter.
        required: false
        type: string
      - name: image_url
        in: query
        description: The URL of an image that should be processed. If this field is not specified, the user must pass an image via the `image_url` form parameter.
        required: false
        type: string
      - name: image_buf
        in: formData
        description: An image that should be processed. This is used when the user uploads a local image for processing rather than specifying the URL of an existing image. If this field is not specified, the user must pass an image URL via the `image_buf` parameter
        required: false
        type: file
      - name: annotate_image
        in: query
        description: A boolean input flag (default=false) indicating whether or not to build and return annotated images within the `annotated_image` field of each response object
        required: false
        type: boolean
      - name: crop_image
        in: query
        description: A boolean input flag (default=false) indicating whether or not to crop and return faces within the `thumbnails` field of each response object
        required: false
        type: boolean

    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
          id: PhotoInfo
          type: object
          required:
            - id
            - faces
            - normalME_group_estimation
          properties:
            id:
                type: string
                format: byte
                description: an identification number for received image
            faces:
                schema:
                    type: object
                    description: an array of face information
                    properties:
                        index:
                            type: int
                            format: byte
                            description: index for each face image
                        location_xy:
                            type: array
                            description: an array of the coordinate of the left-up corner of the bounding box
                        width:
                            type: int
                            description: the width of the detected face
                        height:
                            type: int
                            description: the height of the detected face
                        thumbnail:
                            type: string
                            description: base64 encoded thumbnail image of the detected face
                        feature:
                            type: array
                            description: an array of feature representations extracted from 5 various CNNs
                        LSTM_aggregateed_feature:
                            type: array
                            description: an array of LSTM aggregated feature
                        happiness_intensity:
                            type: fload
                            description: happiness intensity estimation of the face
            annotated_image:
                type: string
                format: byte
                description: a base64 encoded annotated image
            normalME_group_estimation:
                type: float
                description: group-level happiness intensity estimation using normal mean-encoding group emotion modeling  
    '''
    print('predict activated')

    image_rgb, image_gray, annotated_rgb, crop_faces = obtain_images(request)
    photoinfo = run_RRDE(image_gray, image_rgb,annotated_rgb, crop_faces)

    response = jsonify(photoinfo)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@blueprint.route('/feedback', methods=['POST'])
def feedback():
    '''
    Record user feedback
    Return the face and true label when user clicks on the corresponding emoji. The users are given the option to teach the model by clicking on the true emoji icon should the model makes a wrong predictions on a face. Faces with no feedback will default to None and assumes the model made a correct prediction.
    ---
    tags:
      - v1.0.0

    responses:
      200:
        description: A user feedback channel
        schema:
          $ref: '#/definitions/Feedback'
      default:
        description: Unexpected error
        schema:
          $ref: '#/definitions/Error'

    parameters:
      - name: image_id
        in: query
        description: The id of the image processed. This field must be specified in order to insert the feedback to the correct image documentation.
        required: false
        type: string
      - name: face_index
        in: query
        description: The index of the face in question. This field must be specified in order to insert the feedback to the correct image documentation.
        required: false
        type: string
      - name: feedback
        in: query
        description: User feedback of the true emotion if the model predicted less than accurate.
        required: false
        type: string

    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded

    definitions:
      - schema:
          id: Feedback
          type: object
          required:
            - image_id
          properties:
            response:
                type: string
                format: byte
                description: status of received feedback
    '''
    feedback = obtain_feedback(request)
    emojicon = {'angry': 'ANGRY', 'fear': 'FEAR', 'happy': 'HAPPY', 'sad': 'SAD','surprise': 'SURPRISE','neutral': 'NEUTRAL'}
    # print (emojicon[feedback['feedback']]+'-->')*8
    # print feedback
    response = jsonify(feedback)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# query: db.photoinfo.find({"_id": ObjectId("578fd839beba87784205b73b")},{})

from app import app
app.register_blueprint(blueprint, url_prefix='/'+VERSION_STR)
