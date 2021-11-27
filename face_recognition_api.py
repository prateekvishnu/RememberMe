#https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
import face_recognition
import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify,flash
from flask import Markup
# Initialize Flask App
app = Flask(__name__)
app.config["DEBUG"] = True


KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # 'hog' or 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
    

# print('Loading known faces...')
known_faces = []
known_names = []
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            # Append encodings and name
            known_faces.append(encoding[0].tolist())
            known_names.append(name)
# print('Done.')

# print('Print training algorithm...')

# training the encodings with C-Support Vector Classification.
# Input: face_encodings from face_recognition and names. Output: Classifier.

from sklearn.svm import SVC
# class_weight='balanced' ensures the classifier doesn't overfit to a class and avoid class imbalance issues.
clf = SVC(class_weight='balanced', probability=True)
clf.fit(known_faces, known_names)
# print('Done.')
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

#function for output
def face_predicitons(image):
    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)
    
    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)
    
    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
#     print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = clf.predict([face_encoding])[0]
        results2= clf.predict_proba([face_encoding])[0]
        
        #Setting thershold for predicition accuracy to 30%
        threshold = 0.4
        
        if(np.max(results2)>threshold):
            return results, face_location
       
    return None, None
            
#function for putting name
def bbox(image, results, face_location):
    # Each location contains positions in order: top, right, bottom, left
    top_left = (face_location[3], face_location[0])
    bottom_right = (face_location[1], face_location[2])

    # Get color by name using our fancy function
    color = name_to_color(results[0])

    # Paint frame
    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

    # Now we need smaller, filled grame below for a name
    # This time we use bottom in both corners - to start from bottom and move 50 pixels down
    top_left = (face_location[3], face_location[2])
    bottom_right = (face_location[1], face_location[2] + 22)

    # Paint frame
    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

    # Wite a name
    cv2.putText(image, results, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    
    return image

@app.route('/')
def home():
    return render_template('home.html' )
        

@app.route('/reloadDatabase')
def reload():
    
    # print('Loading known faces...')
    known_faces = []
    known_names = []
    for name in os.listdir(KNOWN_FACES_DIR):
    
        # Next we load every file of faces of known person
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
    
            # Load an image
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
    
            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                # Append encodings and name
                known_faces.append(encoding[0].tolist())
                known_names.append(name)
    # print('Done.')
    
    # print('Print training algorithm...')
    
    # training the encodings with C-Support Vector Classification.
    # Input: face_encodings from face_recognition and names. Output: Classifier.

    from sklearn.svm import SVC
    # class_weight='balanced' ensures the classifier doesn't overfit to a class and avoid class imbalance issues.
    clf = SVC(class_weight='balanced', probability=True)
    clf.fit(known_faces, known_names)
    message = "Successfully reloaded."
    
    return render_template('home.html' , messages= message)
    
    
    
@app.route('/uploadAndRun')
def uploadAndRun():    
    file = request.args['videoPath']
    # calling functions
    frame_array =[]
    # cap = cv2.VideoCapture('test_videos/WristCam_Vishnu.mp4')
    # cap = cv2.VideoCapture(r'test_videos/BenAffleck.mp4')
    cap = cv2.VideoCapture(file)
    
    # For streams:
    #   cap = cv2.VideoCapture('rtsp://url.to.stream/media.amqp')
    # Or e.g. most common ID for webcams:
    #   cap = cv2.VideoCapture(0)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
    
        if ret:
    #         print(face_predicitons(frame))
    
            # call function for recognition
            results, face_location = face_predicitons(frame)
            
            if count == 0:
                shape = (frame.shape[1],frame.shape[0])
            if face_location:
                frame_array.append(bbox(frame, results, face_location))
            else:
                frame_array.append(frame)
            count += 1 # i.e. predictions every 10 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    
        else:
            cap.release()
            break
    #Setting output Video Path
    pathOut = 'output/'+file
    fps = 25.0
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, shape)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    message = "Successfully recognized. Output is stored at: "+ str(pathOut)
    # flash(message)
    return render_template('home.html' ,messages=message)
 
    
 
app.run()



