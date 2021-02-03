import torch
import torch.nn as nn
from .backbones.CSPDarknet53 import _BuildCSPDarknet53
# from loss_2 import Loss_recon as Loss
# from loss_2 import FocalLoss as Loss
from layers_yolov4 import GetPBB
import config.yolov4_config as cfg

config = {}
config["anchors"] = cfg.MODEL["ANCHORS3D"]
config["stride"] = cfg.MODEL["STRIDES"]

class Conv_block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
		super(Conv_block, self).__init__()

		self.conv_bn_acti = nn.Sequential(
								nn.Conv3d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias = False),
								nn.BatchNorm3d(out_channels),
								nn.LeakyReLU()
							)

	def forward(self, x):
		return self.conv_bn_acti(x)

class SpatialPyramidPooling(nn.Module):
	def __init__(self, feature_channels, pool_sizes = [3, 5, 9]):
		super(SpatialPyramidPooling, self).__init__()

		self.head_conv = nn.Sequential(
							Conv_block(feature_channels[-1], feature_channels[-1] // 2, 1),
							Conv_block(feature_channels[-1] // 2, feature_channels[-1], 3),
							Conv_block(feature_channels[-1], feature_channels[-1] // 2, 1),
						 )
		self.maxpools = nn.ModuleList([nn.MaxPool3d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
		self.__initialize_weights()

	def forward(self, x):
		x = self.head_conv(x)
		features = [maxpool(x) for maxpool in self.maxpools]
		features = torch.cat([x] + features, dim = 1)

		return features

	def __initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class Upsample(nn.Module):
	def __init__(self, in_channels, out_channels, scale = 2):
		super(Upsample, self).__init__()

		self.upsample = nn.Sequential(
							Conv_block(in_channels, out_channels, 1),
							nn.Upsample(scale_factor = scale)
						)

	def forward(self, x):
		return self.upsample(x)

class Downsample(nn.Module):
	def __init__(self, in_channels, out_channels, scale = 2):
		super(Downsample, self).__init__()

		self.downsample = Conv_block(in_channels, out_channels, 3, 2)

	def forward(self, x):
		return self.downsample(x)

class PANet(nn.Module):
	def __init__(self, feature_channels):
		super(PANet, self).__init__()

		self.feature_transform3 = Conv_block(feature_channels[0], feature_channels[0] // 2, 1)
		self.feature_transform4 = Conv_block(feature_channels[1], feature_channels[1] // 2, 1)

		self.resample5_4 = Upsample(feature_channels[2] // 2, feature_channels[1] // 2)
		self.resample4_3 = Upsample(feature_channels[1] // 2, feature_channels[0] // 2)
		self.resample3_4 = Downsample(feature_channels[0] // 2, feature_channels[1] // 2)
		self.resample4_5 = Downsample(feature_channels[1] // 2, feature_channels[2] // 2)

		self.downstream_conv5 = nn.Sequential(
			Conv_block(feature_channels[2] * 2, feature_channels[2] // 2, 1), 
			Conv_block(feature_channels[2] // 2, feature_channels[2], 3), 
			Conv_block(feature_channels[2], feature_channels[2] // 2, 1), 
		)
		self.downstream_conv4 = nn.Sequential(
			Conv_block(feature_channels[1], feature_channels[1] // 2, 1),
			Conv_block(feature_channels[1] // 2, feature_channels[1], 3),
			Conv_block(feature_channels[1], feature_channels[1] // 2, 1),
			Conv_block(feature_channels[1] // 2, feature_channels[1], 3),
			Conv_block(feature_channels[1], feature_channels[1] // 2, 1),
		)
		self.downstream_conv3 = nn.Sequential(
			Conv_block(feature_channels[0], feature_channels[0] // 2, 1),
			Conv_block(feature_channels[0] // 2, feature_channels[0], 3),
			Conv_block(feature_channels[0], feature_channels[0] // 2, 1),
			Conv_block(feature_channels[0] // 2, feature_channels[0], 3),
			Conv_block(feature_channels[0], feature_channels[0] // 2, 1),
		)

		self.upstream_conv4 = nn.Sequential(
			Conv_block(feature_channels[1], feature_channels[1] // 2, 1),
			Conv_block(feature_channels[1] // 2, feature_channels[1], 3),
			Conv_block(feature_channels[1], feature_channels[1] // 2, 1),
			Conv_block(feature_channels[1] // 2, feature_channels[1], 3),
			Conv_block(feature_channels[1], feature_channels[1] // 2, 1),
		)
		self.upstream_conv5 = nn.Sequential(
			Conv_block(feature_channels[2], feature_channels[2] // 2, 1),
			Conv_block(feature_channels[2] // 2, feature_channels[2], 3),
			Conv_block(feature_channels[2], feature_channels[2] // 2, 1),
			Conv_block(feature_channels[2] // 2, feature_channels[2], 3),
			Conv_block(feature_channels[2], feature_channels[2] // 2, 1)
		)
		self.__initialize_weights()

	def forward(self, features):
		features = [self.feature_transform3(features[0]), self.feature_transform4(features[1]), features[2]]

		downstream_feature5 = self.downstream_conv5(features[2])
		downstream_feature4 = self.downstream_conv4(torch.cat([features[1], self.resample5_4(downstream_feature5)], dim=1))
		downstream_feature3 = self.downstream_conv3(torch.cat([features[0], self.resample4_3(downstream_feature4)], dim=1))

		upstream_feature4 = self.upstream_conv4(torch.cat([self.resample3_4(downstream_feature3), downstream_feature4], dim=1))
		upstream_feature5 = self.upstream_conv5(torch.cat([self.resample4_5(upstream_feature4), downstream_feature5], dim=1))

		return [downstream_feature3, upstream_feature4, upstream_feature5]

	def __initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class PredictNet(nn.Module):
	def __init__(self, feature_channels, target_channels):
		super(PredictNet, self).__init__()

		self.predict_conv = nn.ModuleList([
			nn.Sequential(
				# nn.Dropout3d(p = 0.3),
				Conv_block(feature_channels[i] // 2, feature_channels[i], 3),
				nn.Conv3d(feature_channels[i], target_channels, 1)
			) for i in range(len(feature_channels))
		])
		self.__initialize_weights()

	def forward(self, features):
		# [b, h, w, d, anchor, 5]
		# 5: c, x, y, z, d
		predicts = [predict_conv(feature).permute(0, 2, 3, 4, 1).contiguous().view((feature.size(0), ) + feature.size()[2:] + (len(config['anchors']), 5)) for predict_conv, feature in zip(self.predict_conv, features)]
		return predicts

	def __initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class YOLOv4(nn.Module):
	def __init__(self, out_channels = 5 * len(config['anchors'])):
		super(YOLOv4, self).__init__()

		self.backbone, feature_channels = _BuildCSPDarknet53()
		self.spp = SpatialPyramidPooling(feature_channels)
		self.panet = PANet(feature_channels)
		self.predict_net = PredictNet(feature_channels, out_channels)

		#ccy
		self.getpbb = GetPBB(config)

	def forward(self, x):
		features = self.backbone(x)
		print("After backbone:", features[0].shape)
		features[-1] = self.spp(features[-1])
		print("After spp:", features[0].shape)
		features = self.panet(features)
		print("After panet:", features[0].shape)
		predicts = self.predict_net(features)
		# yushen's code need sigmoid
		for p in predicts:
			print("predicts shape:", p.shape)
			p[..., 0] = torch.sigmoid(p[..., 0])
        
		#ccy
		for idx, p in enumerate(predicts):
			p = self.getpbb(p.detach().cpu(), idx, 0.5)
			print("getpbb:", p.shape)
		raise EOFError
		return predicts
