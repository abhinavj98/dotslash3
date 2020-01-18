from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import cv2
import numpy as np

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)
detector.reset_class(["person"], reuse_weights=['person'])

im_fname = "vid_1_s_1.jpg"
#print(im_fname)

x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
#print("needed : ",type(img))
class_IDs, scores, bounding_boxs = detector(x)

#img = cv2.imread("vid_1_s_1.jpg")
#print("provided : ",type(img))

pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

predicted_heatmap = pose_net(pose_input)
pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                            class_IDs, bounding_boxs, scores,
                            box_thresh=0.5, keypoint_thresh=0.2)

#print('Shape of pre-processed image:', x.shape)
#print(type(x),type(img))
plt.show()
'''cap = cv2.VideoCapture('vid_1.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")



while(cap.isOpened()):

    ret, img = cap.read()
    if ret == True:
        

        pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                                    class_IDs, bounding_boxs, scores,
                                    box_thresh=0.5, keypoint_thresh=0.2)
        plt.show()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()'''
plt.close()
'''cv2.imshow("he",ax)
cv2.waitKey()
cv2.destroyAllWindows()'''



 
# Closes all the frames
#cv2.destroyAllWindows()