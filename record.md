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

| masked  | pretrian hyparameter                         | train hyparameter                  | f1    |
| ------- | -------------------------------------------- | ---------------------------------- | ----- |
| regular | masked radio:0.15 steps:19500, batch_size:32 | lr:2e-5,max_epoch=20,drop_out:0.1  | 0.864 |
| regular | masked radio:0.15 steps:19500,batch_size:32  | lr:1e-4,max_epoch=20,drop_out:0.1  | 0.852 |
| regular | masked radio:0.15 steps:50000,batch_size=32  | lr:2e-5,max_epoch=20,drop_out:0.1  | 0.860 |
| wwm     | masked_radio:0.15 steps:2000,batch_size=32   | lr:2e-5,max_epoch=20,drop_out:0.1  | 0.859 |
| wwm     | masked_radio:0.15 steps:20000,batch_size=32  | lr:2e-5,max_epoch=20,drop_out:0.1  | 0.854 |
| wwm     | masked_radio:0.15 steps:20000,batch_size=32  | lr:2e-5,max_epoch=20,drop_out:0.15 | 0.860 |
|         |                                              |                                    |       |

预训练加成不大，比较好的一点是和官方F1计算的差值大大缩小

### badcase分析

