# -*- coding: utf-8 -*-
# ---
# @File: model.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/6/5
# Describe: SigNet模型，包含了参数较多的标准版本和参数较少的thin版本
# ---

from keras.layers import  Conv2D,BatchNormalization,Dense,Input,MaxPool2D,Activation,Flatten
from keras.models import Sequential,Model
from keras.optimizer_v2.gradient_descent import SGD
import os
import keras.utils.np_utils

class signet():
    def __init__(self,num_class,mod='thin'):
        self.rows=150
        self.cols=220
        self.channles=1
        self.imgshape = (self.rows, self.cols, self.channles)
        self.user_dim=num_class

        self.batchsize=32
        self.epochs=6
        self.optimizer=SGD(learning_rate=1e-3,momentum=0.9,nesterov=True,decay=5e-4)

        assert mod=='thin' or 'std',"model has only two variant: thin and std"
        if mod=='thin':
            self.backbone=self.backbone_thin()
        else:
            self.backbone=self.backbone_std()
        input=Input(shape=self.imgshape)
        x=self.backbone(input)
        output=Dense(self.user_dim,activation='softmax')(x)
        self.signet=Model(input,output)
        self.signet.compile(optimizer=self.optimizer,loss='categorical_crossentropy',metrics='accuracy')
        self.signet.summary()

    def backbone_thin(self):
        model=Sequential()

        model.add(Conv2D(64,kernel_size=5,strides=5,input_shape=self.imgshape,use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Conv2D(96,kernel_size=5,strides=2,padding='same',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Conv2D(128,kernel_size=3,strides=1,padding='same',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(128,kernel_size=3,strides=1,padding='same',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Flatten())
        model.add(Dense(256,use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.summary()
        input=Input(shape=self.imgshape)
        output=model(input)

        return Model(input,output)

    def backbone_std(self):
        model=Sequential()

        model.add(Conv2D(96,kernel_size=11,strides=4,input_shape=self.imgshape,use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Conv2D(256,kernel_size=5,strides=2,padding='same',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Conv2D(384,kernel_size=3,strides=1,padding='same',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(384,kernel_size=3,strides=1,padding='same',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(256,kernel_size=3,strides=1,padding='same',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPool2D(pool_size=3,strides=2))

        model.add(Flatten())
        model.add(Dense(2048,use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(2048,use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.summary()
        input=Input(shape=self.imgshape)
        output=model(input)

        return Model(input,output)

    def train(self,data,weights='',save=False):
        save_dir = './NetWeights/Signet_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.signet.load_weights(filepath)
            doc=None
        else:
            filepath = os.path.join(save_dir, 'signet.h5')
            train_img=data.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            time=0
            doc=[]
            for i in range(1,self.epochs):
                for batch in train_img:
                    train_label=batch[1]
                    loss=self.signet.train_on_batch(x=batch[0],y=train_label)
                    doc.append(loss)
                    print("round %d=> loss:%f, acc:%f%% " % (time,loss[0],loss[1]*100))
                    time+=1
                # 总共进行三次学习率下降，每次下降10%
                if i%(self.epochs//3)==0:
                    self.optimizer.lr-=0.1*self.optimizer.lr
            if save:
                self.signet.save_weights(filepath)
        return doc
