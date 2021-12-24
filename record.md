### baseline 选择

| model                   | hyparametre                           | params | f1    |      |
| ----------------------- | ------------------------------------- | ------ | ----- | ---- |
| distilbert-base-uncased | lr:2e-5, max_epoch: 20, drop_out: 0.1 | 66.4M  | 0.835 |      |
| robert-base             | lr:2e-5, max_epoch: 20, drop_out: 0.1 | 128M   | 0.826 |      |
| roberta-large           | lr:2e-5, max_epoch: 20, drop_out: 0.1 | 335M   | 0.821 |      |
| distilroberta-base      | lr:2e-5, max_epoch: 20, drop_out: 0.1 | 82.1M  | 0.747 |      |
| bert-base-uncased       | lr:2e-5, max_epoch: 20, drop_out: 0.1 | 190M   | 0.855 |      |
|                         |                                       |        |       |      |

baseline选择bert-base-uncased

### 预训练加成实验

| masked  | pretrian hyparameter                         | train hyparameter                      | f1    |
| ------- | -------------------------------------------- | -------------------------------------- | ----- |
| regular | masked radio:0.15 steps:19500, batch_size:32 | lr:2e-5,max_epoch=20,drop_out:0.1      | 0.864 |
| regular | masked radio:0.15 steps:19500,batch_size:32  | lr:1e-4,max_epoch=20,drop_out:0.1      | 0.852 |
| regular | masked radio:0.15 steps:50000,batch_size=32  | lr:2e-5,max_epoch=20,drop_out:0.1      | 0.860 |
| wwm     | masked_radio:0.15 steps:2000,batch_size=32   | lr:2e-5,max_epoch=20,drop_out:0.1      | 0.859 |
| wwm     | masked_radio:0.15 steps:20000,batch_size=32  | lr:2e-5,max_epoch=20,drop_out:0.1      | 0.854 |
| wwm     | masked_radio:0.15 steps:20000,batch_size=32  | lr:2e-[5](),max_epoch=20,drop_out:0.15 | 0.860 |
|         |                                              |                                        |       |

预训练加成不大，尤其是wwm相对于普通的预训练几乎是0提升的。预训练带来比较好的影响是，和官方F1计算的差值大大缩小。

### badcase分析

以某次结果为例(f1=0.86)

​		label    pred      word
12    B-CW  B-PROD     1
29  B-PROD  B-CORP     1
22   B-LOC  I-CORP     1
31  B-PROD  I-PROD     1
63  I-PROD  I-CORP     1
33  I-CORP  B-CORP     1
17   B-GRP   B-LOC     1
61  I-PROD    B-CW     1
36  I-CORP   I-LOC     1
37  I-CORP   I-PER     1
56   I-PER    B-CW     1
58   I-PER   I-GRP     1
8     B-CW  B-CORP     1
6   B-CORP    I-CW     1
5   B-CORP  I-CORP     1
49   I-GRP   I-LOC     1
50   I-GRP   I-PER     1
51   I-GRP       O     1
40    I-CW   B-LOC     2
28   B-PER       O     2
60   I-PER       O     2
46   I-GRP   B-GRP     2
62  I-PROD  B-PROD     2
57   I-PER    I-CW     2
59   I-PER  I-PROD     2
27   B-PER  B-PROD     2
55   I-LOC       O     2
25   B-PER    B-CW     2
18   B-GRP   I-GRP     2
3   B-CORP   B-PER     2
4   B-CORP  B-PROD     2
2   B-CORP   B-LOC     2
21   B-LOC   B-GRP     3
20   B-LOC  B-CORP     3
19   B-GRP       O     3
43    I-CW   I-LOC     3
48   I-GRP    I-CW     3
47   I-GRP  I-CORP     3
7   B-CORP       O     3
9     B-CW   B-GRP     3
26   B-PER   B-GRP     3
41    I-CW  I-CORP     3
10    B-CW   B-LOC     3
23   B-LOC   I-LOC     3
44    I-CW   I-PER     3
11    B-CW   B-PER     4
13    B-CW    I-CW     4
15   B-GRP  B-CORP     4
16   B-GRP    B-CW     4
30  B-PROD    B-CW     4
24   B-LOC       O     4
39    I-CW    B-CW     4
64  I-PROD    I-CW     5
69       O   B-LOC     5
0   B-CORP    B-CW     5
38  I-CORP       O     5
53   I-LOC  I-CORP     5
52   I-LOC   B-LOC     5
54   I-LOC   I-GRP     5
34  I-CORP    I-CW     6
42    I-CW   I-GRP     7
74       O   I-GRP     8
70       O   B-PER     8
68       O   B-GRP     8
66       O  B-CORP     9
1   B-CORP   B-GRP    10
72       O  I-CORP    10
75       O   I-LOC    10
65  I-PROD       O    11
76       O   I-PER    11
32  B-PROD       O    12
77       O  I-PROD    12
35  I-CORP   I-GRP    13
45    I-CW       O    18
14    B-CW       O    22
67       O    B-CW    25
73       O    I-CW    30
71       O  B-PROD    33

CW和PROD误召回和漏召回很高

CW:

1. 验证集上存在一些漏标注问题，例如my mother's curse实际上是一部作品名

2. 错误标注问题，例如television film，其实不是作品



