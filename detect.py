from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from yolov3.util import letterbox_image, load_weights
import config as cfg
from yolov3.model import Yolov3


def detect(image_path, class_names, model_path, yolo_weights=None):
    image = Image.open(image_path)
    resize_image = letterbox_image(image, cfg.INPUT_SHAPE)
    image_data = np.array(resize_image, dtype=np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis=0)
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,))
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)
    model = Yolov3(cfg.BATCH_NORM_DECAY, cfg.BATCH_NORM_EPSILON, cfg.LEAKY_RELU, cfg.ANCHORS, len(class_names))
    yolo_outputs = model.yolo_inference(input_image, is_training=False)
    boxes, scores, classes = model.yolo_eval(yolo_outputs, input_image_shape)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = model.yolo_eval(yolo_outputs, input_image_shape)
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                input_image: image_data,
                input_image_shape: [image.size[1], image.size[0]]
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = None
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(66, 134, 244))
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=(66, 134, 244))
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        image.show()
        image.save('./result1.jpg')


if __name__ == '__main__':
    with open('./model-data/coco_classes.txt', 'r') as f:
        lines = f.readlines()
        class_names = list(x.strip() for x in lines)
    # print(len(class_names))
    detect('./testimage.jpg', class_names, model_path=None, yolo_weights='./model-data/yolov3.weights')
