---  
layout: post  
title: caffe python  
---  
  
1. caffe python source code  
caffe是 **[N C H W]**  
2. net = caffe.Net(prototxt, caffemodel, caffe.TEST)  
	dir(net)  
	['__class__',  
	 '__delattr__',  
	 '__dict__',  
	 '__doc__',  
	 '__format__',  
	 '__getattribute__',  
	 '__hash__',  
	 '__init__',  
	 '__module__',  
	 '__new__',  
	 '__reduce__',  
	 '__reduce_ex__',  
	 '__repr__',  
	 '__setattr__',  
	 '__sizeof__',  
	 '__str__',  
	 '__subclasshook__',  
	 '__weakref__',  
	 '_backward',  
	 '_batch',  
	 '_blob_loss_weights',  
	 '_blob_loss_weights_dict',  
	 '_blob_names',  
	 '_blobs',  
	 '_blobs_dict',  
	 '_bottom_ids',  
	 '_forward',  
	 '_input_list',  
	 '_inputs',  
	 '_layer_names',  
	 '_output_list',  
	 '_outputs',  
	 '_params_dict',  
	 '_set_input_arrays',  
	 '_top_ids',  
	 'backward',  
	 'blob_loss_weights',  
	 'blobs',  
	 'bottom_names',  
	 'copy_from',  
	 'forward',  
	 'forward_all',  
	 'forward_backward_all',  
	 'inputs',  
	 'layers',  
	 'outputs',  
	 'params',  
	 'reshape',  
	 'save',  
	 'set_input_arrays',  
	 'share_with',  
	 'top_names']  
  
	重要的是 layers 和 blobs  
	一个网络（prototxt）有多少个layer {},这里的layers就是多长的list，每个layer是一个layer，包含blob  
	blob就是每个数据块，类似tensorflow的tensor  
	net._blobs是所有blob对象列表  
	net._blobs_dict就是blobs的字典  
	net._blob_loss_weights就是每个blob的权重列表  
	net._blob_loss_weights_dict就是_blob_loss_weights的字典  
	net._blob_names是所有blob名字对象列表  
	  
	net.blobs是所有blob的字典  
  
	input_blob = net.blobs['input']  
	conv1 = net.blobs['net1_conv1']  
	dir(conv1)  
	['__class__',  
	 '__delattr__',  
	 '__dict__',  
	 '__doc__',  
	 '__format__',  
	 '__getattribute__',  
	 '__hash__',  
	 '__init__',  
	 '__module__',  
	 '__new__',  
	 '__reduce__',  
	 '__reduce_ex__',  
	 '__repr__',  
	 '__setattr__',  
	 '__sizeof__',  
	 '__str__',  
	 '__subclasshook__',  
	 '__weakref__',  
	 'channels',  
	 'count',  
	 'data',  
	 'diff',  
	 'height',  
	 'num',  
	 'reshape',  
	 'shape',  
	 'width']  
	  
	shape是caffe的对象 blob的shape是 [num, channels, height, width]  count=num*channels*height*width  
	data是blob的数据，是shape的形状  
	diff是梯度，与data的shape一样  
	blob.data[0]表示的是 一个样本数据 [channels, height, width]  
  
3. 重要的一点是，layer中正向传播bottom --> top，如果有多个bottom(也是就blob)生成一个top，这多个bottom之间是**有顺序的**  
