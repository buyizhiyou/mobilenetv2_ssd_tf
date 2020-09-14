import numpy as np
import tensorflow as tf
import cv2


def get_boxes(output, width, height):
    bboxes = []
    conf_thresh = 0.001
    box = output[0][0]
    scores = output[2][0]

    for i in range(10):
        if scores[i] >= conf_thresh:
            bboxes.append([int(box[i][1] * width), int(box[i][0] * height), int(box[i][3] * width),
                           int(box[i][2] * height), float("%.2f" % scores[i])])

    return bboxes


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

im = cv2.imread("test.png")
im = cv2.resize(im, (300, 300))
inp = im.reshape(1, 300, 300, 3)
inp = inp.astype(np.float32)
norm = (inp/128)-1
index = input_details[0]['index']
interpreter.set_tensor(index, norm)
interpreter.invoke()
output_data0 = interpreter.get_tensor(output_details[0]['index'])
output_data1 = interpreter.get_tensor(output_details[1]['index'])
output_data2 = interpreter.get_tensor(output_details[2]['index'])
output_data3 = interpreter.get_tensor(output_details[3]['index'])
print(output_data0.shape, output_data1.shape,
      output_data2.shape, output_data3.shape)
output = [output_data0, output_data1, output_data2, output_data3]
bboxes = get_boxes(output, 300, 300)
for bbox in bboxes:
    left_top = (bbox[0], bbox[1])
    right_bottom = (bbox[2], bbox[3])
    confidence = bbox[4]
    cv2.rectangle(im, left_top, right_bottom, (255, 0, 0), thickness=2)
    label_text = 'oneheart-{}'.format(confidence)
    cv2.putText(im, label_text, (int(bbox[0]), int(bbox[1]) - 2),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

cv2.imshow("im", im)
cv2.waitKey()
