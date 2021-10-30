import time
import os
import logging
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.utils import download, viz
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxboard import SummaryWriter
from gluoncv.utils import LRScheduler, LRSequential




def main():

	## try to use GPU for training
	# try:
	# 	ctx = [mx.gpu(1)]
	# except:
	# 	ctx = [mx.cpu()]

	ctx = [mx.cpu(0)]

	# prepare model
	# model_name = "ssd_512_resnet50_v1_coco"
	model_name = "ssd_512_mobilenet1.0_coco"
	## this will be used to automatically determine input and output file names
	project_name = "box_score"
	# other_info = ""
	classes = ["box_score"]
	# classes = ['blue_disc_spikes',
	# 		   'blue_snake',
	# 		   'child_seat',
	# 		   'curtain_bells',
	# 		   'double_drum',
	# 		   'football',
	# 		   'glowing_wire',
	# 		   'gold_drum',
	# 		   'guitar',
	# 		   'hammock',
	# 		   'red_noisy_snake',
	# 		   'sea_drum',
	# 		   'singing_dog',
	# 		   'sound_cuboid',
	# 		   'tommel',
	# 		   'tambourine']

	batch_size = 2
	# pre-trained model, reset network to predict new class
	net = gcv.model_zoo.get_model(model_name, pretrained=True)
	# net = gcv.model_zoo.get_model(model_name, classes=classes, pretrained=False, transfer='coco')
	net.reset_class(classes)
	# folder where trained model will be saved
	# saved_weights_path = f"saved_weights/{project_name}_{model_name}_{other_info}/"
	saved_weights_path = f"saved_weights/{project_name}_{model_name}_run_00/"
	if not os.path.exists(saved_weights_path):
		os.makedirs(saved_weights_path)

	# prepare data
	data_shape = 512
	# train_dataset = gcv.data.RecordFileDetection(f'custom_dataset/{project_name}_{other_info}_train.rec', coord_normalized=True)
	train_dataset = gcv.data.RecordFileDetection(f'custom_dataset/{project_name}_train.rec', coord_normalized=True)
	val_dataset  = gcv.data.RecordFileDetection(f'custom_dataset/{project_name}_test.rec', coord_normalized=True)

	## show one image
	# image, label = val_dataset[35]
	# print('label:', label)
	# # display image and label
	# ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)
	# plt.show()



	# output log file
	log_file = open(f'{saved_weights_path}{project_name}_{model_name}_log_file.txt', 'w')
	log_file.write("Epoch".rjust(8))
	for class_name in classes:
		log_file.write(f"{class_name:>15}")
	log_file.write("Total".rjust(15))
	log_file.write("\n")
	# summary file for tensorboard
	sw = SummaryWriter(logdir=saved_weights_path+'logs/', flush_secs=30)


	eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
	# COCO metrics seem to work only on COCO dataset, while custom dataset is a RecordFileDetection file!
	# eval_metric = COCODetectionMetric(val_dataset, '_eval', data_shape=(data_shape, data_shape))

	# create data batches from dataset (net, train_dataset, data_shape, batch_size, num_workers):
	train_batches_list, val_batches_list = get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers=0)
	print(f"Number of train batches -> {len(train_batches_list)}")
	print(f"Number of test batches -> {len(val_batches_list)}")


	# ---------------------
	#   Training SSD
	# ---------------------
	# configuration
	net.collect_params().reset_ctx(ctx)

	## freezing non trainable parameters of the model
	# non_predictor_params = net.collect_params('^((?!predictor).)*$')
	# non_predictor_params.setattr('grad_req', 'null')

	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})

	mbox_loss = gcv.loss.SSDMultiBoxLoss()
	ce_metric = mx.metric.Loss('CrossEntropy')
	smoothl1_metric = mx.metric.Loss('SmoothL1')

	for epoch in range(0, 100):
		ce_metric.reset()
		smoothl1_metric.reset()
		tic = time.time()
		btic = time.time()
		net.hybridize(static_alloc=True, static_shape=True)
		loss1 = 0
		loss2 = 0
		name1 = ''
		name2 = ''
		for batch_counter, train_batch in enumerate(train_batches_list):
			batch_size = train_batch[0].shape[0]
			data = gluon.utils.split_and_load(train_batch[0], ctx_list=ctx, batch_axis=0)
			cls_targets = gluon.utils.split_and_load(train_batch[1], ctx_list=ctx, batch_axis=0)
			box_targets = gluon.utils.split_and_load(train_batch[2], ctx_list=ctx, batch_axis=0)
			with autograd.record():
				cls_preds = []
				box_preds = []
				for x in data:
					cls_pred, box_pred, _ = net(x)
					cls_preds.append(cls_pred)
					box_preds.append(box_pred)
				sum_loss, cls_loss, box_loss = mbox_loss(cls_preds, box_preds, cls_targets, box_targets)
				autograd.backward(sum_loss)
			# since we have already normalized the loss, we don't want to normalize
			# by batch-size anymore
			trainer.step(1)
			ce_metric.update(0, [l * batch_size for l in cls_loss])
			smoothl1_metric.update(0, [l * batch_size for l in box_loss])
			name1, loss1 = ce_metric.get()
			name2, loss2 = smoothl1_metric.get()


			if batch_counter % 20 == 0:
				print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
					epoch, batch_counter, batch_size / (time.time() - btic), name1, loss1, name2, loss2))
			btic = time.time()



		# at end of every epoch, add training info to summary, run evaluation, write in log_file, and save finetuned weights to disk
		sw.add_scalar(tag=name1, value=loss1, global_step=epoch)
		sw.add_scalar(tag=name2, value=loss2, global_step=epoch)
		map_name, mean_ap = validate(net, val_batches_list, ctx, eval_metric)

		print(f"\nValidation accuracies for epoch: {epoch}")
		log_file.write(f"{epoch:>8.0f}")
		for k,v in zip(map_name, mean_ap):
			sw.add_scalar(tag=f'{k}_val_acc', value=v, global_step=epoch)
			print(f"\t{k}={v}")
			log_file.write(f"{v:15.5}")
			log_file.write("\n")
		print("\n")
		# save model parameters for inference every 5 epochs
		if epoch%5==0:
			# net.export(f"{saved_weights_path}", epoch=epoch)
			net.save_parameters(f"{saved_weights_path}ep_{epoch:03d}.params")

	sw.close()






def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
	"""Get dataloader."""
	width, height = data_shape, data_shape
	# width, height = 512, 512
	# use fake data to generate fixed anchors for target generation
	with autograd.train_mode():
		_, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
	batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
	train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
	val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
	val_loader = gluon.data.DataLoader(
		val_dataset.transform(SSDDefaultValTransform(width, height)),
		batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
	return train_loader, val_loader

	### this code was here, and it was unreachable...hope I didn't accidentally delete anything
	# val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
	# val_loader = gluon.data.DataLoader(
	# 	val_dataset.transform(YOLO3DefaultValTransform(width, height)),
	# 	batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
	# return train_loader, val_loader







def validate(net, val_data, ctx, eval_metric):
	"""Test on validation dataset."""
	eval_metric.reset()
	# set nms threshold and topk constraint
	net.set_nms(nms_thresh=0.45, nms_topk=400)
	net.hybridize(static_alloc=True, static_shape=True)
	for batch in val_data:
		data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
		label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
		det_bboxes = []
		det_ids = []
		det_scores = []
		gt_bboxes = []
		gt_ids = []
		gt_difficults = []
		for x, y in zip(data, label):
			# get prediction results
			ids, scores, bboxes = net(x)
			det_ids.append(ids)
			det_scores.append(scores)
			# clip to image size
			det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
			# split ground truths
			gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
			gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
			gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

		# update metric
		eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
	return eval_metric.get()








if __name__ == "__main__":
	main()
