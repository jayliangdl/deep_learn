# 神经网络深度学习--多分类的实现 

## 概要：

本文主要是描述如何使用神经网络解决多分类问题。具体地，我們举例介绍了多分类的应用场景；介绍其中关键的softmax激活函数；详细介绍了softmax如何执行向前传播、向后传播；详细介绍了softmax的推导过程。最后，我們对一个分为3类的数据集合尝试让程序进行学习辨识，经过一个3层的神经网络（列完整的示例代码），程序能辨识到大部分的数据，准确率达到98%喔。

## 适读人群：

对神经网络有初步了解，知道向前传播、向后传播等神经网络的关键步骤，知道如何用神经网络解决二分类问题。希望扩展了解多分类问题的解决方案，希望了解softmax激活函数的详细计算过程及推导过程。

## 正文：



周末花了大半天研究了一下多分类是如何实现的，看似不难，但却很容易让人误解，另外如果要考究推导向后传播，推导其计算是很容易出错。我在这里分享一下，希望能让其他和我一样有此困惑的同学省点时间。

对于二分类问题，例如推测一张图片是一只猫，还是不是一只猫（是猫 vs 不是猫）；或推测用户是否对某商品感兴趣（感兴趣 vs 不感兴趣），我們在最后一道输出层，用的激活函数是simgoid函数（公式：simgoid(Z) =  ![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/softmax_sigmoid.PNG)），最后一般情况下，如果simgoid(Z)>0.5,则是正向结果（例如是猫 或 感兴趣）；如果<=0.5则是反向结果(当然正向何反向结果是相对的，两者可以换转。另外0.5的阀值也是可变的)。而现实生活中，我們还有另外一种多分类问题，可选项大于2，例如预测足球比赛是主队胜、主队负或两者平手；又如经典的手写数字辨识案例，程序要猜是0~9数字中的哪一个。对于此类多分类问题，我們可以用两种方法解决：

一、	继续使用sigmoid激活函数，让分类变成特定的一类和非这特定类的其他类。例如判断病人是患哪种流感（例如分为甲、乙、丙三类），则我們的处理拆分为：甲类与其他类（乙+丙）；乙类与其他类（甲+丙）；丙类与其他类（甲+乙）等。由两个线性回归方程组成（假设非神经网络，没有隐含层）。这方案不是本文重点。

二、	使用softmax激活函数，softmax函数最后输出的是所有分类的概率值，然后看哪类的概率值最大，则归为该类。例如图片识别案例中，我們要识别图片是小狗、小猫、小兔、和其他。可选分类有4个，则经过softmax激活函数后，最后输出的是一个4维（4，1）向量，判断4个概率值中，哪个概率值最大，则判断出该图片是哪个分类：


| 经过softmax计算出概率值 |分类| 
| - | - | 
| 0.1 | 小狗 | 
| 0.1 | 小猫 | 
| 0.6 | 小兔 |
| 0.2 | 其他 |

例如上表中，我們算出数字小猫的概率是0.6，为4个分类中最大，所以我們猜这是小猫的图片。


Softmax的公式列如下：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/softmax_softmax.PNG)

i 表示总共有i个分类；


### 我們以下例说明具体计算步骤如下：

#### 步骤一（向前传播最后一步）：

当我們在向前传播算出了最后一层的Z值后（Z=WX+b），（注：最后一层Z是一个（4，1）的向量），我們依据上面提及的
![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/softmax_softmax2.PNG)
公式计算4个分类的激活值。

例如，
在训练集中某个样本我們算出Z值为：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/sample1.PNG)

则：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/softmax_softmax3.PNG)

得出：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/sample2.PNG)


注：以上只是单个样本的说明，实际上如果有m个样本，g(Z)应该是一个(4,m)的向量。如：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/sample3.PNG)


#### 步骤二（计算Cost）：
单个样本的Cost公式：
![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/cost.PNG)

（ 注：本文没有再说明为何要用此Cost公式，如有兴趣请再搜索其他文献）


其中表示图片真实分类的y值是一个（4,1）的向量，在本例中第2个分类小猫是该图片的真实分类，所以y值表示为：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/sample4.PNG)

再额外举一例，如果第3个分类小兔才是该图片的真实分类，则表示为：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/sample5.PNG)


如果我們的训练集有10张图片，它们的真实类别分别是:

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/table1.PNG)

那么y值表示为：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/sample6.PNG)



![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat.PNG)是我們训练出网络所计算得到的预估值，我們需要让![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat.PNG)尽量逼近y，![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat.PNG)与y的差距尽量小，即cost尽量小。


在本例中，由于我們有4个可选值（小狗、小猫、小兔、其他），单个样本的cost公式可再转化为：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/cost2.PNG)

由于y_1，y_3，y_4均为0所以公式可转为：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/cost3.PNG)

上面为单个样本的公式，如果扩展到m个样本，则Cost Function J为

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/cost4.PNG)

我們的目标是尽量让Cost Function J尽量小，依据上面公式，那么就是让![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat.PNG)尽量的大。如上例，就是让![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat2.PNG)尽量的大。


#### 步骤三（反向传播，计算W和b的导数）：
依据上面公式推导出：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/cost5.PNG)

我們最终可以推导出最后一层W和b的导数分别是：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/formula1.PNG)


需要特别说明的是，以上公式只是针对最后一层，针对softmax函数。另外此公式只针对一个样本，实际上我們有m个样本数据要考虑，所以![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat.PNG)，y两者都是一个(4,m)的向量，而最后用于W和b更新的W导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d1.PNG)和b导数 ![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d2.PNG) 是4个实数，分别是对应4个可选值（小狗、小猫、小兔、其他）的W、b导数。所以以上公式还需要把m个样本的结果加总，除以m，以取平均值。

然后我們依此执行对最后一层的W和b的更新：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/formula2.PNG)

注：learning_rate为学习率

注：以上计算步骤的说明均只针对神经网络中最后一道softmax层，在向前传播步骤中还有其他层的计算在softmax之前；在向后传步骤中还有其他层的计算在softmax层之后。由于这些部分不是本文重点，文中均忽略了。

如果对最后一道softmax层中W、b导数推导有兴趣的同学可以再读以下部分，如果没有兴趣的同学，可以直接跳到样例程序继续阅读。


### Softmax推导：
以下是softmax层W和b导数的推导。需要先重点说明的是，我們需要分两种情况推导，第1种情况是我們推导真实类别的W和b导数，例如上例中第2类别小猫是真实类别，我们要求出针对其的导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d22.PNG)和![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d23.PNG)；另外，第2种情况是我們推导非真实类别的W和b导数，例如上例中第1类不是真实类别，我们需要求出针对其的导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d3.PNG)和![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d4.PNG)。


另外在推导前，我們先再写出最后一层向前传播及Cost的公式，这有助于我們理解推导过程：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/formula3.PNG)

如前所述，我們最终的目标是要我們网络估算出的y ̂值最大程度与真实值y一样，所以我們要Cost function J最小，所以我们目标要计算出W和b在什么情况下，可以让J的值最小。我們用导数反向求W和b. 即我們最关键一步是计算W导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d5.PNG) 和b导数 ![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d6.PNG) ，然后执行W和b的梯度更新。

大致的思路是：
从最后一步Cost function J，我們可以计算出![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat.PNG) 对Cost的导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d7.PNG) （依据公式C）

然后地，我們又可以计算出对 ![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/y_hat.PNG) 对𝑍的导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d8.PNG)（依据公式B），

再，我們又可以计算出对Z_对W的导数 ![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d9.PNG)（依据公式A），

最后依据链式法则，将上面三个结果相乘，计算出W导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d5.PNG) 和b导数![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d6.PNG)

需要特别特别说明的是![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d7.PNG) 中i是指真实类别，在这个公式中，只有真实类别的变量。例如本例中类别2小猫是真实值，所以i=2，这点很难理解也难表达。但是只有领悟了才能明白下面推导的情况2为何公式长这个样子：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d10.PNG)

具体推导如下：
情况1（针对真实类别的W/b，我們以本文案例说明，假设类别2小猫是真实值 ）：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d11.PNG)

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d12.PNG)

情况2（针对非真是类别的W/b，我們以本文案例说明，假设类别2小猫是真实值，类别1为非真是值，下面推导计算类别1的![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d13.PNG)和![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d14.PNG)）：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d15.PNG)

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d16.PNG)

上面推导虽复杂，要区分两种不同情况，不过也可以总结成一个简单的公式，适合上面两种情况，就是：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d17.PNG)

原因是
对于第1种情况，y=1，所以和上面推导的：

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d19.PNG)

原因是
对于第1种情况，y=1，所以和上面推导的：
![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d20.PNG)
是一致的。

对于第2种情况，y=0，所以也和上面推导的：
![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/d21.PNG)
也是一致的。

### 示例：
最后我們将展示一个具体的示例，该示例展示了一个神经网络如何学习并分辨以下的数据点。
如下图，我們按一定规律产生了三种不同颜色的数据点，这些点由两个feature组成（下图X1和X2）。我們编写了一个3层的神经网络学习这些点的分布，学习过程中，我們不断检测程序学习的准确率。最后程序辨识的准确率可达98%。

这个案例是一个较简单的例子，如我們输入不同的数据，用同样的程序，让系统正确对事物进行分类处理。

例如我們输入一堆动物图片及它们的真实分类（如小猫、小狗、小兔、其他），用该套神经网络学习后，当我們输入一个新的图片，程序可辨识这是哪类动物；又如输入球队间比赛的历史记录，用同样程序学习，最终可以估算下场比赛的胜负等。例子还有很多。

![Mou icon](https://raw.githubusercontent.com/jayliangdl/jayliangdl.github.io/master/sample_program.PNG)



You can use the [editor on GitHub](https://github.com/jayliangdl/deep_learn/edit/master/index.md) to maintain and preview the content for your website in Markdown files.


Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/jayliangdl/deep_learn/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
