from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. 추론기 만들기 위한 패키지 가져오기
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object. 추론 만들기
base_options = python.BaseOptions(model_asset_path='models\det\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5) #숫자 이상 찾아라
detector = vision.ObjectDetector.create_from_options(options)

app = FastAPI()

from PIL import Image
import numpy as np
import io

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    content = await file.read()

    # STEP 3: Load the input image.추론 데이터 가져오기
    binary = io.BytesIO(content)

    pil_img = Image.open(binary)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect objects in the input image. 추론
    detection_result = detector.detect(image)
    # DetectionResult(
    #     detections=[
    #         Detection(
    #             bounding_box=BoundingBox(origin_x=72, origin_y=162, width=252, height=191), 
    #             categories=[Category(index=None, score=0.7803766131401062, display_name=None, category_name='cat')], 
    #             keypoints=[]), 
    #         Detection(
    #             bounding_box=BoundingBox(origin_x=303, origin_y=27, width=249, height=345), 
    #             categories=[Category(index=None, score=0.7627291083335876, display_name=None, category_name='dog')], 
    #             keypoints=[])
    #     ]
    # )

    count = len(detection_result.detections)
    object_list = []
    

    for detection in detection_result.detections:
        object_category = detection.categories[0].category_name
        object_list.append(object_category)
    #     if object_category in "person":
    #         person = True
    #     else:
    #         person = False
    human__count = 0
    for detection_human in object_list:
        if "person" in detection_human:
            human__count += 1

    # # STEP 5: Process the detection result. In this case, visualize it.
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    # # cv2_imshow(rgb_annotated_image)
    # cv2.imshow("test",rgb_annotated_image)
    # cv2.waitKey(0)
    return {"count": count,
            "object_list" : object_list,
            "human__count" : human__count}