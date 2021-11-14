**Remember Friends and Family project**  
This project utilizes the face_recognition library to help people with Dementia remember family and friends. This project is an application that uses facial recognition to identify people and match them with a picture of friends or relatives that are uploaded to the application. 


**Installation steps:**

1. Anaconda Installation. https://docs.anaconda.com/anaconda/install/index.html 
2. conda env create -f environment.yml
3. Run the ipython notebook using the cse598_gpu environment.
4. Folder structure 

	.    
	├── ...   
	├── known_faces         		# root folder of known face with images  
	│   ├── person1         		# images of person1  
	│   ├── person2         		# images of person2  
	│   └── ...  
	├── unknown_faces			# folder with images the model is tested on.  
	├── face_recognition.ipynb  
	└── environment.yml  

**Resources:**
1. Project Library: https://pypi.org/project/face-recognition/
2. Running on CUDA library: https://sparkle-mdm.medium.com/python-real-time-facial-recognition-identification-with-cuda-enabled-4819844ffc80
3. Run dlib using CUDA: GPU dlib and face_recognition: https://gist.github.com/MikeTrizna/4964278bb6378de72ba4b195553a3954  
	git clone https://github.com/davisking/dlib.git  
	cd dlib  
	mkdir build  
	cd build  
	cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1  
	cmake --build .  
	cd ..  
	python setup.py install --set DLIB_USE_CUDA=1  
4. Resolving gcc error: - Install alternative gcc compiler https://github.com/ethereum-mining/ethminer/issues/731


