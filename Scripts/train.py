from dataset import dataset_model
import matplotlib.pyplot as plt
from params import get_params
from pix2pix import pix2pix

params = get_params()
dataset = dataset_model(params)
model = pix2pix(params)
model.restore()
data = list(dataset.train.take(1).as_numpy_iterator())
left, right, target = data[0]
model.fit(dataset.train,
          500000)
predict = model.generator([left,
                           right])
predict = predict.numpy()[0]
target = (target+1)/2
predict = (predict+1)/2
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(12, 6))
ax1.imshow(target[0],
           cmap="gray")
ax1.axis("off")
ax2.imshow(predict,
           cmap="gray_r")
ax2.axis("off")
plt.tight_layout()
plt.savefig("test_train.png")
data = list(dataset.test.take(1).as_numpy_iterator())
left, right, target = data[0]
predict = model.generator([left,
                           right])
predict = predict.numpy()[0]
# print(min(predict),max(predict),predict.shape)
target = (target+1)/2
predict = (predict+1)/2
fig, (ax1, ax2) = plt.subplots(1, 2,
                               figsize=(12, 6))
ax1.imshow(target[0],
           cmap="gray")
ax1.axis("off")
ax2.imshow(predict,
           cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.savefig("test.png")
# print(min(predict),max(predict),predict.shape)
