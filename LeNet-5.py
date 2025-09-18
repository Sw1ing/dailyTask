# Download data from open datasets
# from download import download
#
# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
#       "notebook/datasets/MNIST_Data.zip"
# path = download(url, "./", kind="zip", replace=True)
import mindspore
from mindspore import nn, LossMonitor, Model, save_checkpoint
from mindspore.dataset import MnistDataset, vision, transforms
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Momentum, Adam

train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')

images, labels = next(iter(train_dataset))

#数据预处理和批函数
def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)

# shape 64 1 28 28 Float32
# label 64 Int32
#batch_size = 64

# #构建 LeNet-5网络
class LeNet5(nn.Cell):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, pad_mode='valid')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode='valid')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Dense(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Dense(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Dense(84, num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

# 定义网络、损失函数和优化器
net = LeNet5(num_classes=10)
loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
# 定义模型
model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
# 训练网络
print("Starting Training...")
model.train(epoch=5, train_dataset=train_dataset, dataset_sink_mode=False,callbacks=[LossMonitor()])
# 测试
print("Starting Testing...")
acc = model.eval(test_dataset, dataset_sink_mode=False)
print(f"Test Accuracy: {acc['accuracy']}")

#保存模型参数
save_checkpoint(net, "lenet5.ckpt")
print("Model parameters saved as lenet5.ckpt")

model = LeNet5(num_classes=10)
# #加载模型，装载参数,用于测试
param_dict = mindspore.load_checkpoint("lenet5.ckpt")
model.set_train(False)
for data, label in test_dataset:
    pred = model(data)
    predicted = pred.argmax(1)
    print(f'Predicted: "{predicted[:10]}", Actual: "{label[:10]}"')
    break
