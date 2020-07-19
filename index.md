# 一、闲话

其实，这个生成对抗网络（GAN）的系列项目是那个《训练数据太少？过拟合？一文带你领略“数据增长魔法”》系列的衍生剧。本以为完成最后一个“GAN进行CV数据增广”的项目，就把数据增广的tricks集齐了。没想到，GAN做数据增强的水这么深，坑那么容易进。想着兴趣相同的小伙伴们可能在前进的道路上跟我一样遇上这些坑儿，所以，把出坑姿势在这给大家在这里晒晒，防止趴坑儿浪费时间。首先，第一个出坑心得就是，GAN不像其他CV的数据增强技术（几何变换、图像融合等）那样，敲下代码就能给模型涨点。要想用GAN做好数据增强，得充分的理解GAN的原理，知道GAN在数据集上能做什么和怎么做的，知道如何正确的使用它进行数据增强。

# 二、GAN介绍

GAN全称是 Generative Adversarial Network，即生成对抗网络。2014年我们的好伙伴Goodfellow大神提出了GAN，一经推出便引爆全场，此后各种花式变体DCGAN、WGAN、CGAN、CYCLEGAN、STARGAN、LSGAN等层出不穷，在“换脸”、“换衣”、“换天地”等应用场景下生成的图像、视频以假乱真，好不热闹。深度学习三巨神之一的LeCun也对GAN大加赞赏，称“adversarial training is the coolest thing since sliced bread”。

生成对抗网络一般由一个生成器（生成网络），和一个判别器（判别网络）组成。生成器的作用是，通过学习训练集数据的特征，在判别器的指导下，将随机噪声分布尽量拟合为训练数据的真实分布，从而生成具有训练集特征的相似数据。而判别器则负责区分输入的数据是真实的还是生成器生成的假数据，并反馈给生成器。两个网络交替训练，能力同步提高，直到生成网络生成的数据能够以假乱真，并与与判别网络的能力达到一定均衡。

本着开心玩耍的宗旨，正儿八经介绍GAN流程不如给大伙讲一个“终成一代大师”的励志故事。

故事里的方学芹同学就是GAN网络里的生成器，而文谈同学就是判别器。故事的发展过程就是GAN网络的训练过程。

```
方学芹同学和文谈同学从小就是一对热爱文学的诤友。小方爱讲，小文爱听后发表意见。
（GAN网络由两个网络组成，一个是生成器，一个是判别器。）

上小学时，小方给小文推荐了《孟母三迁》、《司马光砸缸》和自己照着前两篇写的《司马光砸锅》。
（将真数据和生成器生成的假数据一起送给判别器判别真假。）
小文看后说：“《司马光砸锅》是你编的吧，故事讲的不够流畅。”说完，小文赶紧拿小本记下鉴别心得。
（判别器通过鉴别真假数据的训练，提高判别能力。)
小方红着脸，去练习如何流畅叙事了。
（生成器通过学习判别器的判别结果，提高生成假数据的逼真程度，以获得骗过判别器的能力。）

中学时代，文笔已褪去青涩的方同学推荐了《庆余年》、《海棠依旧》和自己写的《海棠朵朵》给文同学。
（将真数据和生成器生成的假数据一起送给判别器判别真假。）
文同学也已刷剧无数不可与小学时同日而语，看后评价：“这个《海棠朵朵》不如前两篇写得引人入胜，又是出自你手吧。”鉴定完毕，文同学的信心又增加了不少。
（判别器通过鉴别真假数据的训练，提高判别能力。)
方同学坦然一笑，继续去练习叙事结构与情节渲染。
（生成器通过学习判别器的判别结果，提高生成假数据的逼真程度，以获得骗过判别器的能力。）

方同学和文同学就这样“在文学的蒙蔽与反蒙蔽斗争”中度过了他们的中学时代、大学时代、找工作时代和找工作时代，一路共同进步，来到了属于他们的大师时代。
（判别器与生成器按前面的套路交替训练，逐步分别提高各自的判别能力和生成以假乱真的数据的能力。）

文学造诣已经炉火纯青方先生终于向多年亦对手亦良师的文先生推荐了《金瓶梅》、《红楼梦》和《青楼梦》三部终极作品。
（将真数据和生成器生成的假数据一起送给判别器判别真假。）
文先生这些年来阅人无数，也已是文坛大佬，细细品鉴这些作品后觉得：“这些作品都是出自大师之手，无论古今。”评价第三部作品采前两部之所长，乃“清流之金瓶，烟火之红楼”也。各位文坛名宿也都公允这个评价。
（判别器无论再怎么训练，也无法区分真数据和生成器生成的假数据。而且，生成的数据足够逼真，人类也难以分辨了。）
此时方先生坦言，第三部乃是自己的拙作。众人惊呼：“已得曹先生之真传也！”
（生成器已经完美的拟合了训练数据的分布特征，GAN训练完成。）

至此，写《司马光砸锅》的小方终成一代文坛大佬，故事圆满。
```

实际上这个故事的结局还有其他版本。

如果小学时的小文就已练就一副火眼金睛，无论小方如何努力也无法取得一点能跟上小文的进步，导致小方根本不知如何着力改进，最终只得放弃文学了。反之，如果当时小文比小方还naive，连《司马光砸锅》也看不出破绽，没了鞭策和方向的小方只好接着写《司马光补锅》、《司马光打铁》、《铁匠的自我修养》......所以，要想打通“完美结局”，需要始终在整个过程中让小文同学比小方同学高明一点点，在前面不远处给方同学指明努力的方向。也就是说要想GAN能稳定的继续训练，要始终让判别器的能力强于生成器一点点。判别器太强，则梯度消失，太弱，则生成器的梯度是错误的梯度。两种情况GAN都无法正常训练。

鉴于方、文两位大佬的传说与真迹已多年鲜见于江湖，这里我们只好用MNIST手写数字数据集来演绎他们的故事了。

![](https://ai-studio-static-online.cdn.bcebos.com/bdc3eca1ff4143c3b216bc9092246f6e380e7f3276f04b1480f0d8d0f6b993c3)

下面我们用Paddle框架实现一个GAN网络，并用MNIST手写字符数据集训练。


# 三、用Paddle框架的动态图模式实现一个GAN网络

Paddle支持静态图和动态图两种编写模型的方式。

* 静态图模式（声明式编程范式）：先编译后执行的方式。用户需预先定义完整的网络结构，再编译优化网络后，才能执行获得计算结果。
* 动态图模式（命令式编程范式）：解析式的执行方式。用户无需预先定义完整的网络结构，每写一行网络代码，即可同时获得计算结果。

相比之下，静态图模式能够更方便进行全局优化，所以一般情况下执行效率更高；而动态图模式更加直观、灵活，便于调试模型。

为了更加灵活的试验网络配置，方便的观察网络各个模块的实时输出，我们选取所见即所得的动态图模式演示GAN的结构和原理。而且，即使是看中效率的工业应用场景下，动态图模式也获得越来越多的认可。毕竟多年来码农们也没因为执行效率的原因弃用更加友好高级语言，而采用汇编语言去编写应用程序。更何况，Paddle团队的小姐姐、小哥哥们正夜以继日的努力，以期在新版本中（2.0版），赋予广大用Paddle开发项目的小伙伴们“用动态图开发，用静态图部署”的能力。这样就能兼得开发和部署的效率了，真香不是？

**1.实现网络的数据读取模块**

要喂入生成器高斯分布的噪声隐变量z的维度设置为100。训练集数据使用Paddle框架内置函数paddle.dataset.mnist.train()、paddle.reader.shuffle()和paddle.batch()进行读取、打乱和划分batch。读取图片数据处理为 [N,W,H] 格式。


```python
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear
import numpy as np
import matplotlib.pyplot as plt

# 噪声维度
Z_DIM = 100
BATCH_SIZE = 128

# 读取真实图片的数据集，这里去除了数据集中的label数据，因为label在这里使用不上，这里不考虑标签分类问题。
def mnist_reader(reader):
    def r():
        for img, label in reader():
            yield img.reshape(1, 28, 28)
    return r

# 噪声生成，通过由噪声来生成假的图片数据输入。
def z_reader():
    while True:
        yield np.random.normal(0.0, 1.0, (Z_DIM, 1, 1)).astype('float32')

# 生成真实图片reader
mnist_generator = paddle.batch(
    paddle.reader.shuffle(mnist_reader(paddle.dataset.mnist.train()), 30000), batch_size=BATCH_SIZE)

# 生成假图片的reader
z_generator = paddle.batch(z_reader, batch_size=BATCH_SIZE)
```

测试下数据读取器和高斯噪声生成器。


```python
import matplotlib.pyplot as plt
%matplotlib inline

pics_tmp = next(mnist_generator())
print('一个batch图片数据的形状：batch_size =', len(pics_tmp), ', data_shape =', pics_tmp[0].shape)

plt.imshow(pics_tmp[0][0])
plt.show()

z_tmp = next(z_generator())
print('一个batch噪声z的形状：batch_size =', len(z_tmp), ', data_shape =', z_tmp[0].shape)
```

    一个batch图片数据的形状：batch_size = 128 , data_shape = (1, 28, 28)
    一个batch噪声z的形状：batch_size = 128 , data_shape = (100, 1, 1)



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_5_1.png)


**2.实现GAN的主体--生成器G和判别器D**

GAN性能的提升从生成器G和判别器D进行左右互搏、交替完善的过程得到的。所以其G网络和D网络的能力应该设计得相近，复杂度也差不多。这个项目中的生成器，采用了两个全链接层接两组上采样和转置卷积层，将输入的噪声Z逐渐转化为1×28×28的单通道图片输出。判别器的结构正好相反，先通过两组卷积和池化层将输入的图片转化为越来越小的特征图，再经过两层全链接层，输出图片是真是假的二分类结果。

生成器结构：

![](https://ai-studio-static-online.cdn.bcebos.com/625f6642177144c49bcd01b05a78e19094fe74d0f51a4cdb989648d3adcbe362)

判别器结构：

![](https://ai-studio-static-online.cdn.bcebos.com/d3a0dc774d0d41758bc674deb189a76e653815de352c4e6d951807be13baec91)



```python
# 下面分别实现了“上采样”和“转置卷积”两种方式实现的生成网络。注释掉其中一个版本可测试另一个。

# 通过上采样扩大特征图的版本
class G(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(G, self).__init__(name_scope)
        name_scope = self.full_name()
        # 第一组全连接和BN层
        self.fc1 = Linear(input_dim=100, output_dim=1024)
        self.bn1 = fluid.dygraph.BatchNorm(num_channels=1024, act='tanh')
        # 第二组全连接和BN层
        self.fc2 = Linear(input_dim=1024, output_dim=128*7*7)
        self.bn2 = fluid.dygraph.BatchNorm(num_channels=128*7*7, act='tanh')
        # 第一组卷积运算（卷积前进行上采样，以扩大特征图）
        # 注：此处使用转置卷积的效果似乎不如上采样后直接用卷积，转置卷积生成的图片噪点较多
        self.conv1 = Conv2D(num_channels=128, num_filters=64, filter_size=5, padding=2)
        self.bn3 = fluid.dygraph.BatchNorm(num_channels=64, act='tanh')
        # 第二组卷积运算（卷积前进行上采样，以扩大特征图）
        self.conv2 = Conv2D(num_channels=64, num_filters=1, filter_size=5, padding=2, act='tanh')
        
    def forward(self, z):
        z = fluid.layers.reshape(z, shape=[-1, 100])
        y = self.fc1(z)
        y = self.bn1(y)
        y = self.fc2(y)
        y = self.bn2(y)
        y = fluid.layers.reshape(y, shape=[-1, 128, 7, 7])
        # 第一组卷积前进行上采样以扩大特征图
        y = fluid.layers.image_resize(y, scale=2)
        y = self.conv1(y)
        y = self.bn3(y)
        # 第二组卷积前进行上采样以扩大特征图
        y = fluid.layers.image_resize(y, scale=2)
        y = self.conv2(y)
        return y

# 通过转置卷积扩大特征图的版本
# class G(fluid.dygraph.Layer):
#     def __init__(self, name_scope):
#         super(G, self).__init__(name_scope)
#         name_scope = self.full_name()
#         # 第一组全连接和BN层
#         self.fc1 = Linear(input_dim=100, output_dim=1024)
#         self.bn1 = fluid.dygraph.BatchNorm(num_channels=1024, act='leaky_relu')
#         # 第二组全连接和BN层
#         self.fc2 = Linear(input_dim=1024, output_dim=128*7*7)
#         self.bn2 = fluid.dygraph.BatchNorm(num_channels=128*7*7, act='leaky_relu')
#         # 第一组转置卷积运算
#         self.convtrans1 = Conv2DTranspose(128, 64, 4, stride=2, padding=1)
#         self.bn3 = fluid.dygraph.BatchNorm(64, act='leaky_relu')
#         # 第二组转置卷积运算
#         self.convtrans2 = Conv2DTranspose(64, 1, 4, stride=2, padding=1, act='leaky_relu')
        
#     def forward(self, z):
#         z = fluid.layers.reshape(z, shape=[-1, 100])
#         y = self.fc1(z)
#         y = self.bn1(y)
#         y = self.fc2(y)
#         y = self.bn2(y)
#         y = fluid.layers.reshape(y, shape=[-1, 128, 7, 7])
#         y = self.convtrans1(y)
#         y = self.bn3(y)
#         y = self.convtrans2(y)
#         return y

class D(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(D, self).__init__(name_scope)
        name_scope = self.full_name()
        # 第一组卷积池化
        self.conv1 = Conv2D(num_channels=1, num_filters=64, filter_size=3)
        self.bn1 = fluid.dygraph.BatchNorm(num_channels=64, act='relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2)
        # 第二组卷积池化
        self.conv2 = Conv2D(num_channels=64, num_filters=128, filter_size=3)
        self.bn2 = fluid.dygraph.BatchNorm(num_channels=128, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2)
        # 全连接输出层
        self.fc1 = Linear(input_dim=128*5*5, output_dim=1024)
        self.bnfc1 = fluid.dygraph.BatchNorm(num_channels=1024, act='relu')
        self.fc2 = Linear(input_dim=1024, output_dim=1)

    def forward(self, img):
        y = self.conv1(img)
        y = self.bn1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.pool2(y)
        y = fluid.layers.reshape(y, shape=[-1, 128*5*5])
        y = self.fc1(y)
        y = self.bnfc1(y)
        y = self.fc2(y)

        return y

```

测试生成器G网络和判别器D网络的前向计算结果。一个batch的数据，输出一张图片。


```python
# 测试生成网络G和判别网络D
with fluid.dygraph.guard():
    g_tmp = G('G')
    tmp_g = g_tmp(fluid.dygraph.to_variable(np.array(z_tmp))).numpy()
    print('生成器G生成图片数据的形状：', tmp_g.shape)
    plt.imshow(tmp_g[0][0])
    plt.show()

    d_tmp = D('D')
    tmp_d = d_tmp(fluid.dygraph.to_variable(tmp_g)).numpy()
    print('判别器D判别生成的图片的概率数据形状：', tmp_d.shape)

```

    生成器G生成图片数据的形状： (128, 1, 28, 28)
    判别器D判别生成的图片的概率数据形状： (128, 1)



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_9_1.png)



```python
# 显示图片，构建一个18*n大小(n=batch_size/16)的图片阵列，把预测的图片打印到note中。
import matplotlib.pyplot as plt
%matplotlib inline

def show_image_grid(images, batch_size=128, pass_id=None):
    fig = plt.figure(figsize=(8, batch_size/32))
    fig.suptitle("Pass {}".format(pass_id))
    gs = plt.GridSpec(int(batch_size/16), 16)
    gs.update(wspace=0.05, hspace=0.05)

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image[0], cmap='Greys_r')
    
    plt.show()

show_image_grid(tmp_g, BATCH_SIZE)
```


![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_10_0.png)



```python
# 拼接一个batch图像用于VisualDL可视化
def concatenate_img(input_img):
    img_arr_broadcasted = ((np.zeros([BATCH_SIZE,3,28,28]) + input_img) * 255).astype('uint8').transpose((0,2,3,1)).reshape([-1,16,28,28,3])
    # print(img_arr_broadcasted.shape)
    img_concatenated = np.concatenate(tuple(img_arr_broadcasted), axis=1)
    # print(img_concatenated.shape)
    img_concatenated = np.concatenate(tuple(img_concatenated), axis=1)
    # print(img_concatenated.shape)
    return img_concatenated

plt.figure(figsize=(12,BATCH_SIZE/32),dpi=80)
plt.imshow(concatenate_img(tmp_g))
```




    <matplotlib.image.AxesImage at 0x7f90ff8837d0>




![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_11_1.png)


**3.开始训练GAN网络**

网络的训练优化目标就是如下公式：

![](https://ai-studio-static-online.cdn.bcebos.com/d60b2999c2b44e3397117b5d6ece1eb7efb5a677e5c044b7ac56310af49f532c)

公式出自Goodfellow在2014年发表的论文[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf)。
这里简单介绍下公式的含义和如何应用到代码中。上式中等号左边的部分：

![](https://ai-studio-static-online.cdn.bcebos.com/d4f6e2307d1a4ec586f2e546031cf833e42aea25399346f093f0eedd655be207)表示的是生成样本和真实样本的差异度，可以使用二分类（真、假两个类别）的交叉商损失。

![](https://ai-studio-static-online.cdn.bcebos.com/91e096cf8e6742beac8f7de6e80df1bf6729c19df096424fa915f03f874e9bc1)
表示在生成器固定的情况下，通过最大化交叉商损失![](https://ai-studio-static-online.cdn.bcebos.com/d4f6e2307d1a4ec586f2e546031cf833e42aea25399346f093f0eedd655be207)来更新判别器D的参数。

![](https://ai-studio-static-online.cdn.bcebos.com/b9c493da756a45cf949445d7686b09583372cf9eb90c4a7eace609a3ef180dc3)
表示生成器要在判别器最大化真、假图片交叉商损失![](https://ai-studio-static-online.cdn.bcebos.com/d4f6e2307d1a4ec586f2e546031cf833e42aea25399346f093f0eedd655be207)的情况下，最小化这个交叉商损失。

等式的右边其实就是将等式左边的交叉商损失公式展开，并写成概率分布的期望形式。详细的推导请参见原论文《[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf)》。

下面是训练模型的代码，有详细的注释。大致过程是：先用真图片训练一次判别器d的参数，再用生成器g生成的假图片训练一次判别器d的参数，最后用判别器d判断生成器g生成的假图片的概率值更新一次生成器g的参数，即每轮训练先训练两次判别器d，再训练一次生成器g，使得判别器d的能力始终稍稍高于生成器g一些。


```python
from visualdl import LogWriter

def train(mnist_generator, epoch_num=10, batch_size=128, use_gpu=True, load_model=False):
    # with fluid.dygraph.guard():
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # 模型存储路径
        model_path = './output/'

        d = D('D')
        d.train()
        g = G('G')
        g.train()

        # 创建优化方法
        real_d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-4, parameter_list=d.parameters())
        fake_d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-4, parameter_list=d.parameters())
        g_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-4, parameter_list=g.parameters())
        
        # 读取上次保存的模型
        if load_model == True:
            g_para, g_opt = fluid.load_dygraph(model_path+'g')
            d_para, d_r_opt = fluid.load_dygraph(model_path+'d_o_r')
            # 上面判别器的参数已经读取到d_para了,此处无需再次读取
            _, d_f_opt = fluid.load_dygraph(model_path+'d_o_f')
            g.load_dict(g_para)
            g_optimizer.set_dict(g_opt)
            d.load_dict(d_para)
            real_d_optimizer.set_dict(d_r_opt)
            fake_d_optimizer.set_dict(d_f_opt)

        # 定义日志写入(先清空日志文件夹)
        if load_model == False:
            !rm -rf /home/aistudio/log/
        real_loss_wrt = LogWriter(logdir='./log/d_real_loss')
        fake_loss_wrt = LogWriter(logdir='./log/d_fake_loss')
        g_loss_wrt = LogWriter(logdir='./log/g_loss')
        image_wrt = LogWriter(logdir='./log/imgs')

        iteration_num = 0
        for epoch in range(epoch_num):
            for i, real_image in enumerate(mnist_generator()):
                # 丢弃不满整个batch_size的数据
                if(len(real_image) != BATCH_SIZE):
                    continue
                
                iteration_num += 1
                
                '''
                判别器d通过最小化输入真实图片时判别器d的输出与真值标签ones的交叉熵损失，来优化判别器的参数，
                以增加判别器d识别真实图片real_image为真值标签ones的概率。
                '''
                # 将MNIST数据集里的图片读入real_image，将真值标签ones用数字1初始化
                real_image = fluid.dygraph.to_variable(np.array(real_image))
                ones = fluid.dygraph.to_variable(np.ones([len(real_image), 1]).astype('float32'))
                # 计算判别器d判断真实图片的概率
                p_real = d(real_image)
                # 计算判别真图片为真的损失
                real_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_real, ones)
                real_avg_cost = fluid.layers.mean(real_cost)
                # 反向传播更新判别器d的参数
                real_avg_cost.backward()
                real_d_optimizer.minimize(real_avg_cost)
                d.clear_gradients()
                
                '''
                判别器d通过最小化输入生成器g生成的假图片g(z)时判别器的输出与假值标签zeros的交叉熵损失，
                来优化判别器d的参数，以增加判别器d识别生成器g生成的假图片g(z)为假值标签zeros的概率。
                '''
                # 创建高斯分布的噪声z，将假值标签zeros初始化为0
                z = next(z_generator())
                z = fluid.dygraph.to_variable(np.array(z))
                zeros = fluid.dygraph.to_variable(np.zeros([len(real_image), 1]).astype('float32'))
                # 判别器d判断生成器g生成的假图片的概率
                p_fake = d(g(z))
                # 计算判别生成器g生成的假图片为假的损失
                fake_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_fake, zeros)
                fake_avg_cost = fluid.layers.mean(fake_cost)
                # 反向传播更新判别器d的参数
                fake_avg_cost.backward()
                fake_d_optimizer.minimize(fake_avg_cost)
                d.clear_gradients()

                '''
                生成器g通过最小化判别器d判别生成器生成的假图片g(z)为真的概率d(fake)与真值标签ones的交叉熵损失，
                来优化生成器g的参数，以增加生成器g使判别器d判别其生成的假图片g(z)为真值标签ones的概率。
                '''
                # 生成器用输入的高斯噪声z生成假图片
                fake = g(z)
                # 计算判别器d判断生成器g生成的假图片的概率
                p_confused = d(fake)
                # 使用判别器d判断生成器g生成的假图片的概率与真值ones的交叉熵计算损失
                g_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_confused, ones)
                g_avg_cost = fluid.layers.mean(g_cost)
                # 反向传播更新生成器g的参数
                g_avg_cost.backward()
                g_optimizer.minimize(g_avg_cost)
                g.clear_gradients()
                
                # 打印输出
                if(iteration_num % 1000 == 0):
                    print('epoch =', epoch, ', batch =', i, ', real_d_loss =', real_avg_cost.numpy(), ', fake_d_loss =', fake_avg_cost.numpy(), 'g_loss =', g_avg_cost.numpy())
                    show_image_grid(fake.numpy(), BATCH_SIZE, epoch)
                
                # 写VisualDL日志
                real_loss_wrt.add_scalar(tag='loss', step=iteration_num, value=real_avg_cost.numpy())
                fake_loss_wrt.add_scalar(tag='loss', step=iteration_num, value=fake_avg_cost.numpy())
                g_loss_wrt.add_scalar(tag='loss', step=iteration_num, value=g_avg_cost.numpy())
                image_wrt.add_image(tag='numbers', img=concatenate_img(fake.numpy()), step=iteration_num)
        
        # 存储模型
        fluid.save_dygraph(g.state_dict(), model_path+'g')
        fluid.save_dygraph(g_optimizer.state_dict(), model_path+'g')
        fluid.save_dygraph(d.state_dict(), model_path+'d_o_r')
        fluid.save_dygraph(real_d_optimizer.state_dict(), model_path+'d_o_r')
        fluid.save_dygraph(d.state_dict(), model_path+'d_o_f')
        fluid.save_dygraph(fake_d_optimizer.state_dict(), model_path+'d_o_f')

# train(mnist_generator, epoch_num=10, batch_size=BATCH_SIZE, use_gpu=True, load_model=True)
train(mnist_generator, epoch_num=20, batch_size=BATCH_SIZE, use_gpu=True)

```

    epoch = 2 , batch = 63 , real_d_loss = [0.06107619] , fake_d_loss = [0.16448149] g_loss = [2.0859208]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_1.png)


    epoch = 4 , batch = 127 , real_d_loss = [0.03909737] , fake_d_loss = [0.06267369] g_loss = [3.0091727]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_3.png)


    epoch = 6 , batch = 191 , real_d_loss = [0.01486035] , fake_d_loss = [0.03086974] g_loss = [3.743864]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_5.png)


    epoch = 8 , batch = 255 , real_d_loss = [0.086999] , fake_d_loss = [0.14372507] g_loss = [2.5688832]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_7.png)


    epoch = 10 , batch = 319 , real_d_loss = [0.08311941] , fake_d_loss = [0.2403567] g_loss = [1.9954624]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_9.png)


    epoch = 12 , batch = 383 , real_d_loss = [0.14879018] , fake_d_loss = [0.29707614] g_loss = [1.7317948]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_11.png)


    epoch = 14 , batch = 447 , real_d_loss = [0.17426533] , fake_d_loss = [0.36188668] g_loss = [1.5775356]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_13.png)


    epoch = 17 , batch = 43 , real_d_loss = [0.15387994] , fake_d_loss = [0.30026525] g_loss = [1.7300353]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_15.png)


    epoch = 19 , batch = 107 , real_d_loss = [0.22448821] , fake_d_loss = [0.37466788] g_loss = [1.6089951]



![png](https://raw.githubusercontent.com/ctkindle/CV-Data-Augmentation-3-SMOTE/master/pics/output_13_17.png)


**4.用VisualDL2.0观察训练**

我们也可以使用Paddle框架的VisualDL组件更方便的观察训练过程。VisualDL是深度学习模型可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化，支持实时训练参数分析、图结构、数据样本可视化及高维数据降维呈现等诸多功能。VisualDL原生支持python的使用， 通过在模型的Python配置中添加几行代码，便可为训练过程提供丰富的可视化支持，全面支持Paddle、ONNX、Caffe等市面主流模型结构可视化，广泛支持各类用户进行可视化分析。

VisualDL使用非常便捷，使用流程只有两个步骤：

**1）将loss、image等数据写入log文件**

```
# 导入LogWriter对象
from visualdl import LogWriter
...
# 声明一个loss记录的专用log写入器
real_loss_wrt = LogWriter(logdir='./log/d_real_loss')
...
# 添加loss数据的记录
real_loss_wrt.add_scalar(tag='loss', step=iteration_num, value=real_avg_cost.numpy())
```
图片数据的写入也是类似的，上面源码中有展示。

**2）启动VisualDL服务，在浏览器打开查看页面**

首先，在终端输入：visualdl --logdir ./log --port 8081

其中 --logdir ./log 参数指定log文件的存储目录为当前目录下的log文件夹，--port 8081 参数指定visualdl服务占用的端口，如果其已被占用，可以使用其他端口，如8082、8083等。

然后，再打开这个网址（[https://aistudio.baidu.com/bdcpu3/user/76563/551962/visualdl](https://aistudio.baidu.com/bdcpu3/user/76563/551962/visualdl)）即可进入VisualDL页面查看模型训练情况。在AI Studio中，这个网址就是用visualdl替换原项目运行网址notebooks后面的内容得来的。在自己的主机上，这个网址就是运行visualdl服务的地址加上端口号，如 http://127.0.0.1:8081 。

查看生成器、判别器的loss曲线：

![](https://ai-studio-static-online.cdn.bcebos.com/e18a6014f02f4f2395a4650686a384c0c7aed13238e94c0286e5b89233f3a0e7)

查看训练过程中生成的图片：

![](https://ai-studio-static-online.cdn.bcebos.com/bb08980c37e34083b3f17453544e04e8b2cb29f6d17647178c3e06b0d57b2216)

![](https://ai-studio-static-online.cdn.bcebos.com/8de6b9509f8548f3b87f0c8b8c99ab462d1628b21bfc49cc9c2d3c85c20a10b0)

![](https://ai-studio-static-online.cdn.bcebos.com/73efdb5353a14377a7d9bfffd2e8089454302beadc2e4c049b946010f8987cf2)

VisualDL的详细介绍请参考：[GitHub项目主页](https://github.com/PaddlePaddle/VisualDL)、[VisualDL使用指南](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README.md)。

**5.玩耍经典GAN遇到的一些问题和感悟**

训练模型过程中发现，有时网络在训练几个Epoch后输出的图片的图案开始被挤压到某个边缘，然后就输出全黑的图片了，判别器判别真假图片的loss也会突然大幅下降趋向归零。后来的研究者们实现了各种GAN的变体比如WGAN，以避免这一情况的发生。后续的项目我会尝试复现那些改进的GAN变体，敬请期待。

这几天兄弟除了在笔记本上经受着各种“GAN模式崩溃”的洗礼，还目睹了一场现实版的“GAN梯度消失”。现在“小神兽”们因为疫情都在家上网课，她的母上想视察一下学习效果，结果一年级的小生成器生成的结果，让上了好些个年大xiáo的判别器很容易就挑了一大堆错误。经慈母一大翻教导后，可怜的小生成器就啥也生成不出来了。我就感慨，为啥这个非得叫“生成对抗网络”呢，就不能叫“生成和谐网络”吗？对抗这个词会不会产生误导呢？说对抗的话，判别器和生成器就应该努力提高能力以彻底击败对手。但如果真这样做的话，GAN就不work了呀？现在已经有人把原来的Discriminator叫法改为Critic以更贴切地指出，GAN中的辅助网络的主要目的不只是要鉴别图片的真伪，还要为生成器提供改进方向的评价信息。我觉得要是称它为Warm Caring Critic（循循善诱的判别器）就更好了。因为，这个辅助网络不但要提供改进信息，还要注意节奏，太快太慢都会产生问题。这样GAN就要改为GMN（Generative Mentor Networks）生成教导网络啦，多么贴切，多么和谐，哈哈哈哈。

在这里，我也试着生成一句名人名言--“GAN既是生成器的，又是判别器的，但归根结底还是生成器的。”欢迎各位大佬进行“循循善诱的判别”。

# 四、经典GAN能用于图像数据增强么？

从GAN拟合分布的理论上来说似乎不能。知乎上也有人评论GAN生成的样本直接用于分类器的数据增强是“左脚踩右脚的梯云纵”。但我抱着执念，还是想练练这颗丹。所以我将MNIST数据集按标签分了10份，看看能不能按类别生成0到9的各类标签用于数据增强。


```python
import gzip
import json
import numpy as np

# 定义数据集读取器
def load_data(mode='train', num=0, batch_size=128):
    # 数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    train_set, val_set, eval_set = data
    imgs = train_set[0]
    labels = train_set[1]
    imgs_all = [[] for i in range(10)]
    for i in range(len(imgs)):
        imgs_all[labels[i]].append(imgs[i])

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    elif mode == 'single':
        imgs = imgs_all[num]
        labels = [num] * len(imgs)

    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == batch_size:
                yield np.array(imgs_list)#, np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list)#, np.array(labels_list)

    return data_generator
```


```python
# 注意：此处调用同一 train() 函数训练，会替换上次训练保存的模型
mnist_generator_single = load_data(mode='single', num=5, batch_size=BATCH_SIZE)
train(mnist_generator_single, epoch_num=100, batch_size=BATCH_SIZE, use_gpu=True)
# train(mnist_generator_single, epoch_num=100, batch_size=BATCH_SIZE, use_gpu=True, load_model=True)

```

    loading mnist dataset from ./work/mnist.json.gz ......
    epoch = 28 , batch = 19 , real_d_loss = [0.41486874] , fake_d_loss = [0.51362956] g_loss = [1.3805386]



![png](https://raw.githubusercontent.com/ctkindle/GAN-Implementation-and-Experience/master/pics/output_18_1.png)


    epoch = 57 , batch = 4 , real_d_loss = [0.3164166] , fake_d_loss = [0.4096159] g_loss = [1.4068363]



![png](https://raw.githubusercontent.com/ctkindle/GAN-Implementation-and-Experience/master/pics/output_18_3.png)


    epoch = 85 , batch = 24 , real_d_loss = [0.22499837] , fake_d_loss = [0.29156905] g_loss = [1.6917844]



![png](https://raw.githubusercontent.com/ctkindle/GAN-Implementation-and-Experience/master/pics/output_18_5.png)


跑了几轮发现生成网络无法生成所有数字类别，2和8比较容易生成。其他的数字，有些试了10来次每次都出现“输出全黑”。

这使我意识到，训练数据的多少对GAN网络的训练很有影响。全部60000条数据有时会出现“输出全黑”。只用某一类别的数据训练GAN，其数据量是原来的十分之一，就更容易出现“输出全黑”了，而且，相同的数据量下，不同的图片内容会导致不同严重程度的“输出全黑”几率。

虽然尝试用GAN做数据增强没有成功，但在尝试的过程中也收货了不少经验和认识。经典的GAN感觉比较脆弱啊，伺候不周就会掉链子，这种一边骑车，一边调整链条的日子可不是好过的。还是赶紧试试GAN的改进版兄弟们吧。WGAN可以使GAN训练更稳定，CGAN可以在训练时指定生成的类别，infoGAN不但能指定类别，还能调整图像的特定维度的特征，比如字符倾斜度和粗细等，这个最适合用于数据增强了，以后的项目中都拿来玩耍玩耍。

**P.S.**

感谢用户名为“374771311”的同学提供的改善意见。用真正的转置卷积层替换了原来的上采样层和卷积层后，网络稳定了许多，输出全黑的概率从30%左右降到了几乎不再出现。leaky_relu激活函数的效果也比relu好。原来的网络很难训练到50轮后，现在训练200轮也未再出现“输出全黑”。只是上采样的方式生成的结果似乎边缘更光滑一些，应该是池化的原因吧。

“374771311”同学提供的在生成中使用tanh激活函数替代原来的relu方法，使得生成器不采用转置卷积扩大特征图（保持原来的上采样处理）也能够保持输出稳定，不再输出全黑。这样生成的图像边缘更光滑，噪点更少些。

欢迎各位同学“相互判别”，参与讨论，“共同生成”，一起进步～～
