
# coding: utf-8




from keras.utils import np_utils
import numpy as np


# 下載Mnist資料


from keras.datasets import mnist
(x_Train,y_Train),(x_Test,y_Test) = mnist.load_data()


# 將資料轉為60000*28*28*1 四維陣列


x_Train4D = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')


# 將資料標準化


x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255


# 一位有效編碼


y_Train4D_OneHot = np_utils.to_categorical(y_Train)
y_Test4D_OneHot = np_utils.to_categorical(y_Test)





from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


# 加入線性模型


model = Sequential()


""" 加入卷積層 
filters=產生的濾鏡數 
kernel_size=濾鏡的尺寸 
padding='same' 不改變圖像大小
input_shape=(28,28,1) 第1,2維代表輸入影像形狀28*28大小 第3維陰影像是單色灰階所以是1
activation=激活函數
"""

model.add(Conv2D(filters=16,
                kernel_size=(5,5),
                padding='same',
                input_shape=(28,28,1),
                activation='relu'))


# 加入池化層


model.add(MaxPooling2D(pool_size=(2,2)))


# 加入卷積層


model.add(Conv2D(filters=36,
                kernel_size=(5,5),
                padding='same',
                activation='relu'))


# 加入池化層


model.add(MaxPooling2D(pool_size=(2,2)))


# 加入Dropout層


model.add(Dropout(0.25))


# 加入平坦層


model.add(Flatten())


# 加入隱藏層 units=神經元數量 activation=激活函數


model.add(Dense(units=128,
                activation='relu'))


# 加入Dropout層


model.add(Dropout(0.5))


# 加入輸出層


model.add(Dense(units=10,
                activation='softmax'))


# 顯示模型概要


print(model.summary())


# 訓練初始化


model.compile(loss='categorical_crossentropy',
              optimizer = 'adam' , metrics=['accuracy'])


# 模型訓練


train_history=model.fit(x=x_Train4D_normalize,
                        y=y_Train4D_OneHot,validation_split=0.2,
                        epochs=10, batch_size=300,verbose=2)


# 畫圖函數


import matplotlib.pyplot as plt
def show_train_history(train_history,train,vlidation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[vlidation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train' , 'vlidation'], loc='upper left')
    plt.show()


# 畫出訓練準確度及驗證準確度的曲線


show_train_history(train_history,'acc','val_acc')


# 畫出訓練損失函數及驗證損失函數的曲線


show_train_history(train_history,'loss','val_loss')


# 評估準確率


scores = model.evaluate(x_Test4D_normalize, y_Test4D_OneHot)
print()
print('accuracy=',scores[1])


# 預測


prediction = model.predict_classes(x_Test4D)


# 畫圖函數

def plot_images_labels_prediction(images,labels,
                                 prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25 : num=25
    for i in range(num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx], cmap='binary')
        title='label='+str(labels[idx])
        if len(prediction) > 0:
            title+=',predict='+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()


# 畫出預測結果


plot_images_labels_prediction(x_Test,y_Test,
                               prediction,idx=0,num=25)


# 混淆陣列


import pandas as pd
pd.crosstab(y_Test,prediction,
            rownames=['label'],colnames=['predict'])

