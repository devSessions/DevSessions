import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class FeedForwardNeuralNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(FeedForwardNeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu1(out)
		out = self.fc2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		return out

input_size = 28*28
hidden_size = 100
num_classes = 10

model = FeedForwardNeuralNetwork(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = Variable(images.view(-1, 28*28))
		labels = Variable(labels)

		optimizer.zero_grad()

		outputs = model(images)

		loss = criterion(outputs, labels)

		loss.backward()

		optimizer.step()

		iter += 1
		if iter % 500 == 0:
			correct = 0
			total = 0

			for images, labels in test_loader:
				images = Variable(images.view(-1, 28*28))
				outputs = model(images)

				_, predicted = torch.max(outputs.data, 1)

				total += labels.size(0)
				correct += (predicted == labels).sum()
			accuracy = 100* correct/total
			print("Iteration {}. Loss {}. Accuracy {}".format(iter, loss.data[0], accuracy))


