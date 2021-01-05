from flask_restful import Resource, reqparse, Api
import flask
import werkzeug
import tensorflow.compat.v1 as tf
import cv2 as cv
import utils

tf.disable_v2_behavior()
app = flask.Flask(__name__)
api = Api(app)


class ProcessImageEndpoint(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
        self.req_parser = parser

    def get(self):
        return {"hello": "world"}
    
    def post(self):
        image_file = self.req_parser.parse_args(strict=True).get("image", None)
        if image_file:
            _image = image_file
            _image.save('my_test.jpg')
            image = cv.imread('my_test.jpg')
            # image_file = open('my_test.jpg', 'wb')
            # image_file.write(_image.read())
            # image_file.close()
            sess = tf.Session()
            model = tf.saved_model.loader.load(sess, tags=['serve'], export_dir='model_pb')

            resized_image = tf.image.resize_image_with_pad(image, 64, 1024).eval(session=sess)
            img_gray = cv.cvtColor(resized_image, cv.COLOR_RGB2GRAY).reshape(64, 1024, 1)

            output = sess.run('Dense-Decoded/SparseToDense:0',
                              feed_dict={
                                  'Deep-CNN/Placeholder:0': img_gray
                              })
            output_text = utils.dense_to_text(output[0])

            print(output_text)
            return {"text": output_text}
        else:
            return {"message": "Image not sent"}


api.add_resource(ProcessImageEndpoint, "/upload")
api.add_resource(ProcessImageEndpoint, "/")

if __name__ == '__main__':
    app.run(debug=True)
