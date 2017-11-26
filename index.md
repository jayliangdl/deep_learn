# 神经网络深度学习--多分类的实现 

##概要：

本文主要是描述如何使用神经网络解决多分类问题。具体地，我們举例介绍了多分类的应用场景；介绍其中关键的softmax激活函数；详细介绍了softmax如何执行向前传播、向后传播；详细介绍了softmax的推导过程。最后，我們对一个分为3类的数据集合尝试让程序进行学习辨识，经过一个3层的神经网络（列完整的示例代码），程序能辨识到大部分的数据，准确率达到98%喔。

##适读人群：

对神经网络有初步了解，知道向前传播、向后传播等神经网络的关键步骤，知道如何用神经网络解决二分类问题。希望扩展了解多分类问题的解决方案，希望了解softmax激活函数的详细计算过程及推导过程。

##正文：



周末花了大半天研究了一下多分类是如何实现的，看似不难，但却很容易让人误解，另外如果要考究推导向后传播，推导其计算是很容易出错。我在这里分享一下，希望能让其他和我一样有此困惑的同学省点时间。

对于二分类问题，例如推测一张图片是一只猫，还是不是一只猫（是猫 vs 不是猫）；或推测用户是否对某商品感兴趣（感兴趣 vs 不感兴趣），我們在最后一道输出层，用的激活函数是simgoid函数（公式：simgoid(Z) = ），最后一般情况下，如果simgoid(Z)>0.5,则是正向结果（例如是猫 或 感兴趣）；如果<=0.5则是反向结果(当然正向何反向结果是相对的，两者可以换转。另外0.5的阀值也是可变的)。而现实生活中，我們还有另外一种多分类问题，可选项大于2，例如预测足球比赛是主队胜、主队负或两者平手；又如经典的手写数字辨识案例，程序要猜是0~9数字中的哪一个。对于此类多分类问题，我們可以用两种方法解决：

一、	继续使用sigmoid激活函数，让分类变成特定的一类和非这特定类的其他类。例如判断病人是患哪种流感（例如分为甲、乙、丙三类），则我們的处理拆分为：甲类与其他类（乙+丙）；乙类与其他类（甲+丙）；丙类与其他类（甲+乙）等。由两个线性回归方程组成（假设非神经网络，没有隐含层）。这方案不是本文重点。

二、	使用softmax激活函数，softmax函数最后输出的是所有分类的概率值，然后看哪类的概率值最大，则归位该类。例如图片识别案例中，我們要识别图片是小狗、小猫、小兔、和其他。可选分类有4个，则经过softmax激活函数后，最后输出的是一个4维（4，1）向量，判断4个概率值中，哪个概率值最大，则判断出该图片是哪个分类：

|经过softmax计算出概率值         | 分类   |

| 0.1                            | 小狗   |

| 0.6                            | 小猫   |

| 0.1                            | 小兔   |

| 0.2                            | 其他   |



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
