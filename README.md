# 基于TensorFlow实现的闲聊机器人

GitHub上实际上有些实现，不过最出名的那个是torch实现的，DeepQA这个项目到是实现的不错，不过是针对英文的。

这个是用TensorFlow实现的sequence to sequence生成模型，代码参考的TensorFlow官方的

https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/translate

这个项目，还有就是DeepQA

https://github.com/Conchylicultor/DeepQA

语料文件是db/dgk_shooter_min.conv，来自于 https://github.com/rustch3n/dgk_lost_conv

参考论文：

Sequence to Sequence Learning with Neural Networks

A Neural Conversational Model

# 依赖

python3 是的这份代码应该不兼容python2吧

numpy 科学运算

sklearn 科学运算

tqdm 进度条

tensorflow 深度学习

大概也就依赖这些，如果只是测试，装一个cpu版本的TensorFlow就行了，也很快。

如果要训练还是要用CUDA，否则肯定超级慢超级慢～～

# 本包的使用说明

本包大体上是用上面提到的官方的translate的demo改的，官方那个是英文到法文的翻译模型

下面的步骤看似很复杂……其实很简单

## 第一步

输入：首先从[这里](https://github.com/rustch3n/dgk_lost_conv)下载一份dgk_shooter_min.conv.zip

输出：然后解压出来dgk_shooter_min.conv文件

## 第二步

在***项目录下***执行decode_conv.py脚本

输入：python3 decode_conv.py

输出：会生成一个sqlite3格式的数据库文件在db/conversation.db

## 第三步

在***项目录下***执行data_utils.py脚本

输入：python3 data_utils.py

输出：会生成一个bucket_dbs目录，里面包含了多个sqlite3格式的数据库，这是将数据按照大小分到不同的buckets里面

例如问题ask的长度小于等于5，并且，输出答案answer长度小于15，就会被放到bucket_5_15_db里面

## 第四步 训练

下面的参数仅仅为了测试，训练次数不多，不会训练出一个好的模型

size: 每层LSTM神经元数量

num_layers: 层数

num_epoch: 训练多少轮（回合）

num_per_epoch: 每轮（回合）训练多少样本

具体参数含义可以参考train.py

输入：python3 train.py --size 128 --num_layers 2 --num_epoch 5 --num_per_epoch 10000

输出：默认在model目录会输出模型文件，上面的参数大概会生成230MB的模型

## 第五步 测试

下面的测试参数应该和上面的训练参数一样，只是最后加了--test true 进入测试模式

输入：python3 train.py --size 128 --num_layers 2 --num_epoch 5 --num_per_epoch 10000 --test true

输出：在命令行输入问题，机器人就会回答哦！但是上面这个模型会回答的很糟糕哦！

如果 --size 512 --num_layers 2 --num_epoch 5 --num_per_epoch 1000000 基本上就可以训练出会说点人话的机器人了

# 项目文件

db/chinese.txt 小学生必须掌握的2500个汉字

db/gb2312_level1.txt GB2312编码内的一级字库

db/gb2312_level2.txt GB2312编码内的二级字库

*上面几个汉字文件主要是生成字典用的，我知道一般的办法可能是跑一遍数据库，然后生成词频（字频）之类的，然后自动生成一个词典，不过我就是不想那么做……总觉得那么做感觉不纯洁～～*

db/dictionary.json 字典

# 测试结果

**不同的参数和数据集，结果都可能变化很大，仅供参考**

说：你好

你好，我是乔治

---

说：你好呀

你好

---

说：你是谁

我是谁

---

说：你从哪里来？

我从来没有告诉过你

---

说：你到哪里去？

我要去

---

说：你想咋？

我想我们应该去看看

---

说：你想去哪？

我想去

---

说：你吃了吗？

你吃了我的蛋糕

---

说：你想吃啥？

我想吃点东西

---

说：你喝酒吗？

我不喝酒

---

说：你喜欢我吗？

我喜欢你

---

说：你恨我吗？

没错

---

说：你要疯啊？

我不知道

---

说：你知道啥？

我知道，我知道

---

说：你想不想跳伞？

不想跳，我们不会跳的

---

说：你喜欢大闸蟹吗？

我不知道

---

说：你知道什么？

我知道

---

说：陪我聊天吧

我们有个约会

---

说：千山万水总是情

我们还是不知道

---

说：一枝红杏出墙来

邱掌柜

---

说：玩去

你们两个

# 后记

调通这个竟然花了我将近3个星期的课余时间，本来一开始一个多星期想用Keras的，结果不给力啊老师。

期间到是用Keras调了一个Pairs相关的模型……（文献： Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks）

然后开始用TensorFlow，花了一个多星期，最终还是完成了hohoho……（其实我觉得我早就成功了，只是一开始输出的代码写错了，结果耽误了很久，我都以为模型出错了）
