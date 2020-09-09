def augment_image(batch_img, K = 2):
	'''Function augment_image:
		Input: PIL Image/Torch Tensor
		Output: K number of augmented images'''
	
	batch_augment_images = []
	for i in range(batch_img.shape[0]):
		img = batch_img[i]
		img = TF.to_pil_image(img.cpu())
		img_1 = TF.to_tensor(TF.adjust_brightness(img, np.random.uniform(0.5, 1.5)))
		img_2 = TF.to_tensor(TF.adjust_contrast(img, np.random.uniform(0.5, 1.5)))
		img_3 = TF.to_tensor(TF.adjust_saturation(img, np.random.uniform(0.5, 1.5)))
		
		img_4 = TF.to_tensor(TF.hflip(img))
		img_5 = TF.to_tensor(TF.rotate(img, angle=np.random.uniform(-10,10)))

		img_6 = TF.to_tensor(TF.to_grayscale(img, num_output_channels=3))
		img_7 = TF.to_tensor(TF.adjust_gamma(img, np.random.uniform(0.5, 1.5)))

		random_numbers = random.sample(range(1, 8), K)
		img_dict = {'1': img_1, '2': img_2, '3': img_3, '4': img_4, '5': img_5, '6': img_6, '7': img_7}

		augment_images = []
		for i in random_numbers:
			augment_images.append(img_dict[str(i)])
		#augment_images = torch.FloatTensor(augment_images)
		batch_augment_images.append(augment_images)
	#batch_augment_images = torch.tensor(batch_augment_images)
	return batch_augment_images

def label_guessing(model, augment_images, device):
	''' Function label_guessing
		Input: Classifier model, K augmentations of the unlabelled data
		Ouput: Predictions for the K augmentations and averages them to get the guessed
				labels for unlabelled data.
		'''
	predictions = []

	for i in range(0,len(augment_images)):
		img = torch.stack(augment_images[i], dim=0)
		img = img.to(device)
		logits = model(img)
		probas = nn.functional.softmax(logits, dim=1)
		predictions.append(probas)
	predictions = torch.stack(predictions,dim=0)
	q_hat = torch.mean(predictions, dim=1)

	return q_hat

def sharpen(p, T=0.5):
	'''Sharpening function as described in the paper.
	   Increases confidence of the model in its predictions.
	   Entropy minimization is implicitly achieved through this function.'''
	p_sharp = torch.pow(p, 1/T)/(torch.sum(torch.pow(p, 1/T), dim=0))
	return p_sharp

def mixup(x1,y1,x2,y2,alpha=0.75):
	'''Mixup function as described in the paper. Instead of passing single images and their corresponding labels
	   a linear combination of two images and their respective labels is passed to the model.'''
	l = np.random.beta(alpha,alpha)
	l = max(l,1-l)
	x = l * x1 + (1-l) * x2
	y = l* y1 + (1-l) * y2
	return x,y

class MixMatch_Dataset(Dataset):
	'''Supply a batch of labelled and unlabelled data, X and U.'''
	
	def __init__(self, Labelled_data, Unlabelled_data):
		self.Labelled_data = Labelled_data
		self.Unlabelled_data = Unlabelled_data
  
	def __getitem__(self, index):
		
		size_labelled = len(self.Labelled_data)
		size_unlabelled = len(self.Unlabelled_data)
		
		if(index < size_labelled):
			l_index = index
			
		else:
			l_index = int(index*len(self.Labelled_data)/len(self.Unlabelled_data))

		if(index < size_unlabelled):
			u_index = index
		else:
			u_index = index - size_unlabelled

		x = self.Labelled_data[l_index][0]
		y = self.Labelled_data[l_index][1]
		u = self.Unlabelled_data[u_index][0]

		return x, y, u
	
	def __len__(self):
		return max(len(self.Labelled_data), len(self.Unlabelled_data))

MixMatch_dataset = MixMatch_Dataset(Labelled_data=AAF_train_labelled, Unlabelled_data=AAF_train_unlabelled)
MixMatch_loader = DataLoader(MixMatch_dataset, batch_size=BATCH_SIZE, shuffle=True)