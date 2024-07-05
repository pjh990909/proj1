import urllib.request #.request 실행할때 붙여줌

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg','am.png']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

#   https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg
#   https://storage.googleapis.com/mediapipe-tasks/image_classifier/cat.jpg

# import cv2
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0)

# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)






# STEP 1: Import the necessary modules. 패키지를 가져오는거임
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론기 객체 만들기(추론기 마다 옵션이 다름)
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite0.tflite')#쓸려고 하는 모델 경로를 넣어줌
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1) #갯수 설정
classifier = vision.ImageClassifier.create_from_options(options) 


# STEP 3: Load the input image. 추론 데이터 가져오기
image = mp.Image.create_from_file(IMAGE_FILENAMES[2]) # 이미지 경로 바뀌줌

# STEP 4: Classify the input image. 추론하기
classification_result = classifier.classify(image)
# print(classification_result)

# STEP 5: Process the classification result. In this case, visualize it. 어떻게 보여줄거냐
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")