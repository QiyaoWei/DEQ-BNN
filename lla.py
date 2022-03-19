import time
from collections import OrderedDict
import os
import re
import math
import argparse
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from modules.model import ResDEQ
# import matplotlib.pyplot as plt

from backpack import backpack, extend
from backpack.extensions import BatchGrad
# pip install git+http://github.com/f-dangel/backpack.git@development

parser = argparse.ArgumentParser(description='LLA for DEQ')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--not-use-gpu', action='store_true', default=False)

parser.add_argument('-b', '--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--test-batch-size', default=128, type=int)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data-root', type=str, default='./cifar')
parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
					help='path to pretrained MAP checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='./logs/', type=str)
parser.add_argument('--job-id', default='default', type=str)

parser.add_argument('--nystrom-samples', default=100, type=int,
					metavar='N', help='subsample size for nystrom')
parser.add_argument('--sigma2', default=0.1, type=float)
parser.add_argument('--num-samples-eval', default=256, type=int,
					metavar='N', help='subsample size for nystrom')
parser.add_argument('--ntk-std-scale', default=1, type=float)
parser.add_argument('--use-normalizer', action='store_true', default=False)

def main():
	args = parser.parse_args()
	args.save_dir = os.path.join(args.save_dir, args.job_id)
	args.num_classes = 10 if args.dataset == 'cifar10' else 100

	if os.path.isdir('/data/LargeData/cifar/'):
		args.data_root = '/data/LargeData/cifar/'

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	device = torch.device('cuda') if not args.not_use_gpu and torch.cuda.is_available() else torch.device('cpu')

	train_loader, val_loader = cifar_loaders(args)
	train_loader_noaug, _ = cifar_loaders(args, batch_size=args.nystrom_samples, noaug=True)

	model = ResDEQ(3, 48, 64, args.num_classes).float().to(device)
	print(model)
	print("Number of parameters", count_parameters(model))
	if args.pretrained is not None:
		print("Load MAP model from", args.pretrained)
		model.load_state_dict(torch.load(args.pretrained))

	print("---------MAP model ---------")
	test(val_loader, model, device, args)

	###### lla ######
	bp_model = extend(copy.deepcopy(model))
	bp_model.eval()

	## calculate eigenfunctions by the nystrom method
	x_nystrom, _ = next(iter(train_loader_noaug))
	x_nystrom = x_nystrom.to(device, non_blocking=True)
	J_nystrom = jacob(bp_model, x_nystrom, args).permute(1, 0, 2)
	K = torch.einsum('onp,omp->onm', J_nystrom, J_nystrom)
	if args.use_normalizer:
		normalizer = K.diagonal(dim1=1, dim2=2).mean(-1)
		K = K / normalizer.view(-1, 1, 1)

	p, q = torch.linalg.eigh(K)
	eigenvalues = p / args.nystrom_samples
	#
	if args.use_normalizer:
		Psi = lambda x: torch.einsum('bop,onp,onm->bom', jacob(bp_model, x, args), J_nystrom, q) / p.add(1e-8).sqrt() / normalizer.view(-1, 1)
		eigenfuncs = lambda x: torch.einsum('bop,onp,onm->bom', jacob(bp_model, x, args), J_nystrom, q) / p.add(1e-8) / normalizer.view(-1, 1) * math.sqrt(args.nystrom_samples)
	else:
		Psi = lambda x: torch.einsum('bop,onp,onm->bom', jacob(bp_model, x, args), J_nystrom, q) / p.add(1e-8).sqrt()
		eigenfuncs = lambda x: torch.einsum('bop,onp,onm->bom', jacob(bp_model, x, args), J_nystrom, q) / p.add(1e-8) * math.sqrt(args.nystrom_samples)

	## check if the nystrom method is correct
	print(K[0])
	print((Psi(x_nystrom).permute(1, 0, 2) @ Psi(x_nystrom).permute(1, 2, 0))[0])

	## pass the training set
	model.eval()
	with torch.no_grad():
		cov = torch.zeros(args.nystrom_samples, args.nystrom_samples).cuda(non_blocking=True)
		for i, (x, _) in enumerate(train_loader_noaug):
			x = x.to(device, non_blocking=True)
			output = model(x)
			prob = output.softmax(-1)
			Delta_x = prob.diag_embed() - prob[:, :, None] * prob[:, None, :] # please check this
			Psi_x = Psi(x).to(device, non_blocking=True)
			cov += torch.einsum('bok,boj,bjl->kl', Psi_x, Delta_x, Psi_x)

		cov.diagonal().add_(1/args.sigma2)
		cov_inv = cov.inverse()

	## test on validation data
	print("---------LLA model ---------")
	lla_test(val_loader, model, device, args, Psi, cov_inv)


def jacob(bp_model, x, args):
	with torch.enable_grad():
		J = []
		for k in range(args.num_classes):
			output = bp_model(x)
			bp_model.zero_grad()
			with backpack(BatchGrad()):
				output[:,k].sum().backward()
			J.append(torch.cat([p.grad_batch.flatten(1) for p in bp_model.parameters()], -1).cpu())
	return torch.stack(J, 1)

def lla_test(test_loader, model, device, args, Psi, cov_inv):
	model.eval()

	probs = []
	targets = []
	with torch.no_grad():
		for x, y in test_loader:
			x = x.to(device)
			y = y.to(device)
			y_pred = model(x)

			Psi_x = Psi(x).to(device, non_blocking=True)
			F_var = Psi_x @ cov_inv.unsqueeze(0) @ Psi_x.permute(0, 2, 1)
			F_samples = (psd_safe_cholesky(F_var) @ torch.randn(F_var.shape[0], F_var.shape[1], args.num_samples_eval, device=F_var.device)).permute(2, 0, 1) * args.ntk_std_scale + y_pred
			prob = F_samples.softmax(-1).mean(0)

			probs.append(prob)
			targets.append(y)

		probs, targets = torch.cat(probs), torch.cat(targets)
		confidences, predictions = torch.max(probs, 1)

		acc = (predictions == targets).float().mean().item()
		ece = _ECELoss()(confidences, predictions, targets).item()
		loss = F.cross_entropy(probs.log(), targets).item()

	print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(loss, acc, ece))


def test(test_loader, model, device, args):
	model.eval()

	preds = []
	targets = []
	with torch.no_grad():
		for x_batch, y_batch in test_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)
			y_pred = model(x_batch)

			preds.append(y_pred)
			targets.append(y_batch)

		preds, targets = torch.cat(preds), torch.cat(targets)
		probs = preds.softmax(-1)
		confidences, predictions = torch.max(probs, 1)

		acc = (predictions == targets).float().mean().item()
		ece = _ECELoss()(confidences, predictions, targets).item()
		loss = F.cross_entropy(preds, targets).item()

	print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(loss, acc, ece))

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

class _ECELoss(torch.nn.Module):
	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(_ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

		bin_boundaries_plot = torch.linspace(0, 1, 11)
		self.bin_lowers_plot = bin_boundaries_plot[:-1]
		self.bin_uppers_plot = bin_boundaries_plot[1:]

	def forward(self, confidences, predictions, labels, title=None):
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=confidences.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

		accuracy_in_bin_list = []
		for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			accuracy_in_bin = 0
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean().item()
			accuracy_in_bin_list.append(accuracy_in_bin)

		if title:
			fig = plt.figure(figsize=(8,6))
			p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align = 'edge', edgecolor ='black')
			p2 = plt.plot([0,1], [0,1], '--', color='gray')

			plt.ylabel('Accuracy', fontsize=18)
			plt.xlabel('Confidence', fontsize=18)
			#plt.title(title)
			plt.xticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.yticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.xlim(left=0,right=1)
			plt.ylim(bottom=0,top=1)
			plt.grid(True)
			#plt.legend((p1[0], p2[0]), ('Men', 'Women'))
			plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=18)
			fig.tight_layout()
			plt.savefig(title, format='pdf', dpi=600, bbox_inches='tight')

		return ece

def cifar_loaders(args, batch_size=None, noaug=None):
	if args.dataset == 'cifar10':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		data_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda()
		data_std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda()

		if noaug:
			T = transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])
		else:
			T = transform=transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32, 4),
				transforms.ToTensor(),
				normalize,
			])
		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(root=args.data_root, train=True, transform=T, download=True),
			batch_size=batch_size or args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True)

		val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(root=args.data_root, train=False,
			transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])),
			batch_size=args.test_batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True)
	elif args.dataset == 'cifar100':
		normalize = transforms.Normalize(mean=[x / 255 for x in [129.3, 124.1, 112.4]],
										 std=[x / 255 for x in [68.2, 65.4, 70.4]])
		data_mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]]).view(1,-1,1,1).cuda()
		data_std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]]).view(1,-1,1,1).cuda()

		if noaug:
			T = transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])
		else:
			T = transform=transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32, 4),
				transforms.ToTensor(),
				normalize,
			])
		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(root=args.data_root, train=True, transform=T, download=True),
			batch_size=batch_size or args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True)

		val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(root=args.data_root, train=False,
			transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])),
			batch_size=args.test_batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True)
	else:
		raise NotImplementedError

	return train_loader, val_loader

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
	"""Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
	Args:
		:attr:`A` (Tensor):
			The tensor to compute the Cholesky decomposition of
		:attr:`upper` (bool, optional):
			See torch.cholesky
		:attr:`out` (Tensor, optional):
			See torch.cholesky
		:attr:`jitter` (float, optional):
			The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
			as 1e-6 (float) or 1e-8 (double)
	"""
	try:
		L = torch.cholesky(A, upper=upper, out=out)
		return L
	except RuntimeError as e:
		isnan = torch.isnan(A)
		if isnan.any():
			raise NanError(
				f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
			)

		if jitter is None:
			jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
		Aprime = A.clone()
		jitter_prev = 0
		for i in range(5):
			jitter_new = jitter * (10 ** i)
			Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
			jitter_prev = jitter_new
			try:
				L = torch.cholesky(Aprime, upper=upper, out=out)
				warnings.warn(
					f"A not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
				return L
			except RuntimeError:
				continue
		raise e

if __name__ == '__main__':
	main()
