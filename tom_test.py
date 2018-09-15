import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

import time

# Yitao-TLS-Begin
import tensorflow as tf
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End

cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

iteration_list = [15, 1, 10]
for iteration in iteration_list:
	start = time.time()

	for i in range(iteration):
		# Read image from file
		file_name = "demo/image.png"
		image = imread(file_name, mode='RGB')

		image_batch = data_to_input(image)

		# Compute prediction with the CNN
		outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})

		# print(outputs_np.keys())
		# print(outputs["locref"])
		# print(outputs["part_prob"])
		# print(outputs_np["locref"].shape)
		# print(outputs_np["locref"][0, 63, 35, :])

		scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

		# Extract maximum scoring location from the heatmap, assume 1 person
		pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

	end = time.time()
	print("It takes %s sec to run %d images for pose-tensorflow" % (str(end - start), iteration))

# print(pose)
# # print(cfg)
# print(scmap.size)

# # Visualise
# visualize.show_heatmaps(cfg, image, scmap, pose)
# visualize.waitforbuttonpress()


# export_flag = True
# if export_flag:
#     # Yitao-TLS-Begin
#     export_path_base = "pose_tensorflow"
#     export_path = os.path.join(
#         compat.as_bytes(export_path_base),
#         compat.as_bytes(str(FLAGS.model_version)))
#     print('Exporting trained model to %s' % str(export_path))
#     builder = saved_model_builder.SavedModelBuilder(export_path)

#     tensor_info_x = tf.saved_model.utils.build_tensor_info(inputs)
#     tensor_info_y1 = tf.saved_model.utils.build_tensor_info(outputs["locref"])
#     tensor_info_y2 = tf.saved_model.utils.build_tensor_info(outputs["part_prob"])

#     prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
#         inputs={'tensor_inputs': tensor_info_x},
#         outputs={'tensor_locref': tensor_info_y1,
#         			'tensor_part_prob': tensor_info_y2},
#         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

#     legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
#     builder.add_meta_graph_and_variables(
#         sess, [tf.saved_model.tag_constants.SERVING],
#         signature_def_map={
#           'predict_images':
#             prediction_signature,
#         },
#         legacy_init_op=legacy_init_op)

#     builder.save()

#     print('Done exporting!')
#     # Yitao-TLS-End