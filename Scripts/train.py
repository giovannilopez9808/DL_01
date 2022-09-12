from dataset import dataset_model
from params import get_params
from pix2pix import pix2pix
from PIL import Image
import matplotlib.image
from numpy import min,max

params = get_params()
dataset = dataset_model(params)
model = pix2pix()
data=list( dataset.test.take(1).as_numpy_iterator())
left,right,target=data[0]
model.fit(dataset.train,
          20000)
predict = model.generator([left,
                           right])
predict=predict.numpy()[0]
print(min(predict),max(predict),predict.shape)
predict=(predict+1)/2
print(min(predict),max(predict),predict.shape)
matplotlib.image.imsave('test.png', predict)

