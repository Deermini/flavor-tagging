flavor-tagging
====================
所需安装的库
--------------------
	xgboost
	gcforest
	anaconda3
	tensorflow
	
代码运行：
-------------------
	运行main.py即可。

save_model.py模块介绍：
-------------------
	run():存储模型及roc曲线
	run1():画出ROC曲线并且计算出相应的AUC值
	run3():使用经过rfe进行特征排序后的结果计算不同特征下的AUC值

实验结果：
------------------
![](https://github.com/Deermini/flavor-tagging/blob/master/picture/3.png)
		

经过特征选择后的结果：
------------------
利用特征递归消除算法对特征排序后计算得分
![](https://github.com/Deermini/flavor-tagging/blob/master/feature_selection_result/1-2.png)
![](https://github.com/Deermini/flavor-tagging/blob/master/feature_selection_result/1-3.png)



