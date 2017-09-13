# 2 conv layers - same padding, 2 max pool layers, 1 FC layer
# import libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

# load data
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# setup batch size, iterations and epochs
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

# make dataset iterable
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# **** Output Formula for Conv= [ (W-K+2P)/S ]+1, Padding Formula = (K-1)/2, Pooling formula = W/K
# Model Class
class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__()
		# Conv 1
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
		self.relu1 = nn.ReLU()
		# Max Pool 1
		self.maxpool1 = nn.MaxPool2d(kernel_size=2)
		# Conv 2
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
		self.relu2 = nn.ReLU()
		# Max Pool 2
		self.maxpool2 = nn.MaxPool2d(kernel_size=2)
		# Fully Connected
		self.fc1 = nn.Linear(32*4*4, 10)

	def forward(self, x):
		# Conv 1
		out = self.conv1(x)
		out = self.relu1(out)
		# Max Pool 1
		out = self.maxpool1(out)
		# Conv2
		out = self.conv2(out)
		out = self.relu2(out)
		# Max Pool 2
		out = self.maxpool2(out)
		# resize
		out = out.view(out.size(0), -1)
		# Fully connected
		out = self.fc1(out)
		return out

# Instantiate Model
model = CNNModel()

# Setup Loss and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
iter = 0
for epoch in range(num_epochs):
	for i, (images,labels) in enumerate(train_loader):
		# Load Images as variables
		images = Variable(images)
		labels = Variable(labels)

		# clear gradient
		optimizer.zero_grad()

		# forward pass to get output/logits
		outputs = model(images)

		# calculate loss
		loss = criterion(outputs, labels)

		# get gradients wrt parameters
		loss.backward()

		# update parameters
		optimizer.step()

		iter += 1

		if iter % 500 == 0:
			correct = 0
			total = 0
			# iterate through test data
			for images, labels in test_loader:
				# convert images to variables
				images = Variable(images)
				# forward pass for output on test data
				outputs = model(images)
				# get prediction from maximum value
				_, predicted = torch.max(outputs.data, 1)
				# total number of labels
				total += labels.size(0)
				# total correct predictions
				correct += (predicted == labels).sum()
			# accuracy
			accuracy = 100 * correct/total
			# Print loss iterations and accuracy
			print("Iter: {}. Loss: {}. Accuracy: {}".format(iter, loss.data[0], accuracy))
