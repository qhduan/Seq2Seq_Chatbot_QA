# 放在前头的“废话”

这个repo诞生比较早，那个时候tensorflow还没到1.0版本，
所以这个模型当时用的tf.contrib.seq2seq库，现在已经是tf.contrib.legacy_seq2seq了，
我想大家明白**legacy**的意思。

这个repo的本身目的是学习与实现seq2seq的相关内容，
并不是一个完整的software，所以它除了学习和别人参考来说，就有各种各样的问题。

有一些同学问我的一些问题，反而是一些常见的python编码问题，
还有一些是有些童鞋想在windows上调试整个模型，
这个我真的没用windows弄过，
鉴于这个repo的性质并不是一个完整软件，我也没有想过要做的太完美，很抱歉。

在我看来，bot，特指会与人产生互动的bot，是人机协作（互动）的一部分。
而关于自然语言的bot，又是它的一个子集。而chatbot本身，又是这种bot的一种子集。

以大家熟知的个人助理为例（siri，cortana，echo），
它包含了如dialogue system，各种qa system，当然也包含了chatbot。
而单看chatbot的实现，截止到今天，主要可实际落地的还是retrieval based的chatbot。

不过seq2seq模型，作为一个关于deeplearning的前沿，依然不断地奋斗在NLG，QA，chatbot等领域。

关于seq2seq的实现有很多，我后来觉得这个repo实际上有点对不起大家的stars，
但是这个repo本身因为那个legacy_seq2seq的问题，确实也没什么好“更新”的了，因为更新就是重写。

***一部分是为了自己学习吧，我就重写了另一个repo***（我是不是很表脸）

https://github.com/qhduan/just_another_seq2seq

这个repo主要是：

- 增加了使用上的各种测试例子（翻译，NER，chatbot-adversial）
- 各种中文注释与README（如果注释行数也算钱的话，这个里面注释可能比代码值钱）
- 各种简单的测试用例，代码发布经过pylint检查

简单的来说如果有同学只是想抄个大作业的话，这个repo更有效……

我认为现在研究关于语言交互的bot有三个主要方面，
一方面人在主攻对话系统，我觉得微软的paper比较多，例如这篇 Xiujun Li, End-to-End Task-Completion Neural Dialogue Systems, 2017

另一拨人主要在做QA，或者QA相关的集成，这部分亚马逊相关的比较多，alexa prize相关的很多文章
都有这样的感觉，
例如这篇 Huiting Liu, RubyStar: A Non-Task-Oriented Mixture Model Dialog System, 2017

还有一拨人主要在做chatbot之类的，例如本repo和上面我提到我的repo相关的，例如这篇 Jiwei Li, Adversarial Learning for Neural Dialogue Generation, 2017

（诶？怎么感觉都是华裔？）

上面那三篇都是17年下半年的，现在才18年2月，其实每个方向现在都很前沿。

因为，本身一个QA的子课题，就已经是比较前沿的了，
把所有这些整合大系统的技术，应该主要握在大佬手里，这里重点指国外大佬手里，
国内我看不出哪个大佬有，大概都比较内敛，当然也是因为这方面研究***并不是很有用***。

（单说能看到的国内的，很可能有错：图灵是一个数据不少的retrieval模型；
一个ai（世纪佳缘的）还有阿里的ruyi是一个简化后的dialogue system模型，
他们是“中国版”的api.ai）


如果要对bot即相关技术有粗略了解的，推荐看看斯坦福正在写的这本
Speech and Language Processing 3rd
地址[这里](http://web.stanford.edu/~jurafsky/slp3/)
看第28~30章


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

输入：

```
./train_model.sh
```

上面这个脚本内容相当于运行：

```
python3 s2s.py \
--size 1024 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 64 \
--num_per_epoch 500000 \
--model_dir ./model/model1
```

输出：在 model/model1 目录会输出模型文件，上面的参数大概会生成700MB的模型

如果是GPU训练，尤其用的是<=4GB显存的显卡，很可能OOM(Out Of Memory)，
这个时候就只能调小size，num_layers和batch_size

## 第五步 测试

下面的测试参数应该和上面的训练参数一样，只是最后加了--test true 进入测试模式

输入：

```
./train_model.sh test
```

上面这个脚本命令相当于运行：

```
python3 s2s.py \
--size 1024 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 64 \
--num_per_epoch 500000 \
--model_dir ./model/model1 \
--test true
```

输出：在命令行输入问题，机器人就会回答哦！但是上面这个模型会回答的不是很好……当然可能怎么训练都不是很好，不要太期待～～

# 项目文件

db/chinese.txt 小学生必须掌握的2500个汉字

db/gb2312_level1.txt GB2312编码内的一级字库

db/gb2312_level2.txt GB2312编码内的二级字库

*上面几个汉字文件主要是生成字典用的，我知道一般的办法可能是跑一遍数据库，然后生成词频（字频）之类的，然后自动生成一个词典，不过我就是不想那么做……总觉得那么做感觉不纯洁～～*

db/dictionary.json 字典

# 测试结果

**不同的参数和数据集，结果都可能变化很大，仅供参考**

**下面训练结果是用train_model.sh的参数训练的**

> 你好
你好

> 你好呀
你好

> 你是谁
我是说，我们都是朋友

> 你从哪里来
我不知道

> 你到哪里去
你不是说你不是我的

> 你喜欢我吗？
我喜欢你

> 你吃了吗？
我还没吃饭呢

> 你喜欢喝酒吗？
我不知道

> 你讨厌我吗？
我不想让你失去我的家人

> 你喜欢电影吗？
我喜欢

> 陪我聊天吧
好啊

> 千山万水总是情
你不是说你不是我的错

> 你说话没有逻辑啊
没有

> 一枝红杏出墙来
你知道的

# 其他

很多论文进行 bleu 测试，这个本来是测试翻译模型的，其实对于对话没什么太大意义

不过如果想要，可以加 bleu 参数进行测试，例如

```
./train_model.sh bleu 1000
```

具体可以参考 s2s.py 里面的 test_bleu 函数

最后，这个跟现在的机器人平台，和他们所用的技术其实没啥关系，
如果对于机器人(平台)感兴趣，可以看看[这里](https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md)

更多问题欢迎与我交流
