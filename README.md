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

需要安装TensorFlow

其他的一些我没有用pip导出，至少会依赖：

numpy 科学运算

sklearn 科学运算

tqdm 进度条


# 项目文件

db/ 数据文件夹

dgk_shooter_min.conv 数据，来源 https://github.com/rustch3n/dgk_lost_conv 电影对话 ***太大了传不到GitHub，在gitignore***

dgk_shooter_min.conv.7z 上个文件的压缩版

chinese.txt 小学生必须掌握的2500个汉字

gb2312_level1.txt GB2312编码内的一级字库

gb2312_level2.txt GB2312编码内的二级字库

decode_conv.ipynb jupyter notebook文件，用来把dgk_shooter_min.conv文件中的对话转换到conversation.db的数据库

conversation.db 上个程序生成的数据库 ***太大了传不到GitHub，在gitignore***

generate_dict.ipynb jupyter notebook文件，用来生成词典，例如2500个小学生必备汉字+英文+标点，这样的词典，我没有用数据库按字频生成词典

dictionary.json 上个程序生成的词典

model/ 模型文件夹，保存生成好的模型

model.ckpt 这个文件和下面文件都是TensorFlow的模型文件，或者说session文件？ ***太大了传不到GitHub，在gitignore***

model.ckpt.meta ***太大了传不到GitHub，在gitignore***

./ 根目录

config.json 配置文件，配置模型超参

data_util.py 一个辅助库，里面的函数是：用来处理数据啦，产生模型啦，各种我懒得写注释的小破函数啦

train.py 训练程序，需要用到上面两个文件，还有字典文件和数据库文件

test.py 测试文件，运行之后会提示你输入

# 测试结果

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
