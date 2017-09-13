import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# y = 2x+1, alpha = 2, beta = 1

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
print(x_train.shape)

x_train = x_train.reshape(-1,1)
print(x_train.shape)

y_values = [2*i+1 for i in x_values]

y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(y_train.shape)


class LinearRegressionModel(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LinearRegressionModel, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		out = self.linear(x)
		return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

if torch.cuda.is_available():
	model.cuda()



criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100

for epoch in range(epochs):
	epoch +=1

	if torch.cuda.is_available():
		inputs = Variable(torch.from_numpy(x_train).cuda())
		labels = Variable(torch.from_numpy(y_train).cuda())
	else:
		inputs = Variable(torch.from_numpy(x_train))
		labels = Variable(torch.from_numpy(y_train))

	optimizer.zero_grad()

	outputs = model(inputs)

	loss = criterion(outputs, labels)

	loss.backward()

	optimizer.step()

	print('epoch {}, loss {}'.format(epoch, loss.data[0]))

save_model = True
if save_model is True:
	torch.save(model.state_dict(), 'linReg.pk1')

load_model = False
if load_model is True:
	model.load_state_dict(torch.load('linReg.pk1'))

print("Predicted Values ******")
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
print(predicted)

print("Actual Values ******")
print(y_train)

print("***** plotting *******")

plt.clf()
plt.plot(x_train, y_train, 'go', label="true data", alpha=0.5)
plt.plot(x_train, predicted, '--', label="Predicted data", alpha=0.5)

plt.legend(loc='best')
plt.show()




