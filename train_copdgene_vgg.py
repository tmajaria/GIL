####### MAIN ######
import argparse
import tensorflow as tf
import numpy as np

from models.vgg16 import VGG_16
from utils.copdgene_data_generator import *

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--insp_exp', help='Specify INSP(iration) or EXP(iration). Default is both', default='')
	parser.add_argument('--std_sharp', help='Specify STD or SHARP images. Default is both', default = '')
	parser.add_argument('--num_files', help='Number of files to include in training. Default is 100', type=int, default=100)
	parser.add_argument('--epochs', help='Number of epochs. Default is 15', type=int, default=15)
	parser.add_argument('--batch_size', help='Training batch size. Default is 8', type=int, default=8)
	parser.add_argument('--output', help="Specify file name for output. Default is 'model'", default='model')
	args = parser.parse_args()

	insp_exp = args.insp_exp
	std_sharp = args.std_sharp
	num_files = args.num_files
	epochs = args.epochs
	batch_size = args.batch_size
	output = args.output

	# Point to project files folder
	parent_dir = '/sbgenomics/project-files/copdgene-nrrd/'

	# Pull the list of files
	train_images, train_labels  = pullRandomNrrds(parent_dir, num_files=1000)

	# Split test set
	test_images = []
	test_labels = []
	test_count = int(0.3 * len(train_images)) # Using 30% of subjects for test set

	while len(test_images) < test_count:
	    random_index = random.randrange(len(train_images))
	    test_images.append(train_images.pop(random_index))
	    test_labels.append(train_labels.pop(random_index))

    # Get total number of images in each set
    test_image_sizes, test_image_count = getImageSetSize(test_images)
	train_image_sizes, train_image_count = getImageSetSize(train_images)

	# Create a mirrored strategy
	strategy = tf.distribute.MirroredStrategy()
	print(f'Number of devices: {strategy.num_replicas_in_sync}')

	# Initialize settings for training
	batch_size = 8
	epochs =  15
	train_steps = train_image_count // batch_size
	val_steps = test_image_count // batch_size

	# Create the data generators
	trainGen = batchGenerator(train_images, train_labels, batch_size)
	testGen = batchGenerator(test_images, test_labels, batch_size)

	# Build the model
	with strategy.scope():
		model = VGG_16(input_shape=(512,512,1), classes=2)
		opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
		model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	# Train the model
	model_checkpoint = ModelCheckpoint(output+'.h5', monitor='accuracy', verbose=1, save_best_only=True)
	H = model.fit(
		x=trainGen,
		steps_per_epoch=train_steps,
		validation_data=testGen,
		validation_steps=val_steps,
		epochs=epochs)

	# Save loss history
    loss_history = np.array(history.history['loss'])
    np.savetxt(output+'_loss.csv', loss_history, delimiter=",")
