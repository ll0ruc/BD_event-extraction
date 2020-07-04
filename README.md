# BD_event-extraction
百度2020语言与智能技术竞赛：事件抽取赛道方案代码

## 说明

这个是初步参加事件抽取比赛的一个锻炼机会，由于其他事情的繁忙，只参与了第一阶段的线上评测，F1值达到0.78
这里把事件抽取当成两阶段任务，先做事件触发词检测，再做事件论元检测，每阶段都是当作一个序列标注任务
采用pipeline模型，网络的主体架构采用BERT+LSTM。


## 使用

### 准备
1、
```
pip install pytorch pytorch_pretrained_bert numpy
```
  
2、下载pytorch版本的bert_base_chinese和bert-base-chinese-vocab.txt到主目录下

### 训练、测试
```
python train.py
```

## 总结

  事件抽取在这次的任务中不需要识别出句子中的触发词，只需要识别出句子所对应的事件类型，所以不同于常见的事件抽取任务定义
  可将该问题看成论元-(事件类型-角色)的多分类任务，共65个事件类型，将<事件类型-论元>组合得到共217种类别，如<死亡-时间>、<死亡-地点>等，这样就可以做一个联合抽取的模型
  由于这些方法都没法解决论元角色覆盖问题，可以采用序列标注的混合多标签来解决这个问题，可见论文 [[Neural Architectures for Nested NER through Linearization]](https://www.aclweb.org/anthology/P19-1527.pdf)

## Reference

* Nlpcl-lab's bert_event_extraction repository [[github]](https://github.com/nlpcl-lab/bert-event-extraction)
