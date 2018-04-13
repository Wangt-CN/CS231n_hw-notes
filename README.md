##——————2018.4.5——————

该repo是我在学习CS231n过程中，对所有notes和homework中包含的代码的编写，并加入了详细的comments。



##——————2018.4.8——————  
完成Note1_NN，所有部分都做了详细标注，学习了怎么导入数据，发现python挺慢的

完成assignment1中的部分KNN

其中计算distance部分，完全不用loop没有想出来，借鉴一下网上的答案，本质上用了一个数学上平方展开加上矩阵广播，比较巧妙。

学习numpy库中的：  
np.argsort（将矩阵中的元素按行或列从小到大排列，输出其对应的标号值）  
np.bincount（输入一维矩阵，输出矩阵各元素出现的次数）  
np.argmax（返回对应维度的最大索引）  
np.linalg.normal（求输入矩阵的二范数，默认）  


##——————2018.4.10——————  
完成knn.ipynb

学习numpy库中的：  
np.split(x,n), (x, [])（将行矩阵平均分成n个，或者按照list分成几份）  
np.hstack()（水平的按列对数组或者矩阵进行堆叠）  
np.vstack()（竖直的按行对数组或者矩阵进行堆叠）

制作交叉验证集时注意使用list嵌套list

时刻注意函数的输入 是什么，矩阵的形状


##——————2018.4.12——————  
SVM分类器的损失函数鼓励正确的分类的分值比其他分类的分值高出至少一个边界值。  
而softmax是计算所有类别的可能性，同时softmax分类器对于分数是永远不会满意的：正确分类总能得到更高的可能性，错误分类总能得到更低的可能性  
个人觉得SVM方法在实现的过程中还是有些绕的，特别是理论求导修正W的部分，容易弄错

寻找矩阵中的元素可采用 a[list1, list2]　的方法


linear_svm时，发现对其中dW的计算理解的还不是很到位，注意每一个样本进去，所有的W都将更新，而不仅仅更新W[:, y[i]] 

(a>0).astype(int) / (float), 让矩阵中a>0的设为1，小于0的设为0

##——————2018.4.13—————— 

在写的过程中用矩阵实现计算梯度始终不是很会，借鉴了别人的写法，但还是很难想
  counts = (margin > 0).astype(int)
  counts[range(N), y] = - np.sum(counts, axis = 1)
  dW += np.dot(X.T, counts) / N + reg * W

np.random.choice在使用时先求随机索引会方便很多  
np.argmax() 寻找矩阵中的最大值索引

svm.ipynb中源代码好像不太对，for x in results()改成for x in results.values()，访问存放的具体值

