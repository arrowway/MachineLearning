##第五章 Logistic回归

假设现在有一些数据点，我们利用一条直线对这些点进行拟合（该线称为最佳拟合直线），这个拟合过程就称作回归。

为了实现`Logistic`回归分类器，我们可以在每个特征上都乘以一个回归系数，然后把所有的结果值相加，将这个结果代入`Sigmoid`函数中，进而得到一个范围在0-1之间的数值。任何大于0.5的数据被分入1类，小于0.5即被归入0类。

###1.`Sigmoid`函数的输入记为 \(z\),由下面的公式得出：

$$ z = {\omega_0}{x_0} + {\omega _1}{x _1} + {\omega _2}{x _2} + ... + {\omega _n}{x _n}$$

###2.梯度算法的迭代公式为：
$$ \omega  = \omega  + \alpha {\nabla _\omega }f(\omega ) $$
$$ \theta _j = \theta _j + \alpha({y^i - {h _\theta} ({x^i})}) * {x _j^i} $$

###3.`Logistic`回归梯度上升优化算法

**原始数据：**  
![](http://7xt9dp.com1.z0.glb.clouddn.com/5-3.png)

**Logistic回归梯度上升优化算法-python代码:**

    def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

    def sigmoid(inX):
        return 1.0/(1+exp(-inX))

    def gradAscent(dataMatIn, classLabels):
        dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
        labelMat = mat(classLabels).transpose() #convert to NumPy matrix
        m,n = shape(dataMatrix)
        alpha = 0.001
        maxCycles = 500                         #迭代次数
        weights = ones((n,1))
        for k in range(maxCycles):              #heavy on matrix operations
            h = sigmoid(dataMatrix*weights)     #matrix mult
            error = (labelMat - h)              #vector subtraction
            weights = weights + alpha * dataMatrix.transpose()* error #matrix mult 这是梯度上升公式更新参数
        return weights                          #返回值为[w0,w1,w2]

在python提示符下，敲下如下代码：

    dataArr, labelMat = logRegres.loadDataSet()
    logRegres.gradAscent(dataArr, labelMat)
    Out[12]: 
    matrix([[ 4.12414349],
            [ 0.48007329],
            [-0.6168482 ]])

###4.分析数据：画出决策边界

** 画出数据集和Logistic回归最佳拟合直线的函数 **

    def plotBestFit(weights):              #此处weights即为上面求出的[w0,w1,w2]
       import matplotlib.pyplot as plt
       dataMat,labelMat=loadDataSet()
       dataArr = array(dataMat)
       n = shape(dataArr)[0] 
       xcord1 = []; ycord1 = []
       xcord2 = []; ycord2 = []
       for i in range(n):
           if int(labelMat[i])== 1:
               xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
           else:
               xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
       fig = plt.figure()
       ax = fig.add_subplot(111)
       ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
       ax.scatter(xcord2, ycord2, s=30, c='green')
       x = arange(-3.0, 3.0, 0.1)
       y = (-weights[0]-weights[1]*x)/weights[2] # 0=w0x0+w1x1+w2x2 (x0=1) 此处x=x1；y=x2
       ax.plot(x, y)
       plt.xlabel('X1'); plt.ylabel('X2');
       plt.show() 

在python提示符下，敲下如下代码：

    logRegres.plotBestFit(weights.getA())  #getA() 将矩阵转化为数组array

**运行结果为：**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-4.png)

###5.训练算法：随机梯度上升 

改进算法，一次仅用一个样本点来更新回归系数——**在线学习算法**

** 随机梯度上升算法 **

    def stocGradAscent0(dataMatrix, classLabels):
        m,n = shape(dataMatrix)
        alpha = 0.01
        weights = ones(n)   #initialize to all ones
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
        return weights

在python提示符下，敲下如下代码：

    weights = logRegres.stocGradAscent0(array(dataArr), labelMat)

**运行结果为：**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-5.png)

**改进的随机梯度上升算法 **

    def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

1. 步长alpha每次迭代时都会调整，可以缓解数据波动或者高频扰动。且alpha会随着迭代次数**不断减小**，但永远不会减小到0（**常数项**）——可以保证多次迭代后新数据仍然有影响。如果要处理的问题是动态变化的，可以适当加大常数项，确保新值获得更大的回归系数。当\( j << max(i) \)时，步长alpha不是严格下降的。**避免参数严格下降**也常见于模拟退火等其他优化算法中。
2. **随机选取样本值**来更新回归系数。这种方法将减少周期性的波动——具体实现方法：每次随机从样本中选出一个值，更新回归系数后将其删除，再进行下一次迭代。

**改进后回归系数变化情况**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-7.png)

**收敛速度更快，波动更小**

在python提示符下，敲下如下代码：

    dataArr, labelMat = logRegres.loadDataSet()
    weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
    logRegres.plotBestFit(weights)

**运行结果为：**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-8.png)

###6. 示例：从疝气病症预测病马的死亡率

**Logistic回归分类函数**

    def classifyVector(inX, weights):
        prob = sigmoid(sum(inX*weights))
        if prob > 0.5: return 1.0
        else: return 0.0

    def colicTest():
        frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
        trainingSet = []; trainingLabels = []
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr =[]
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
        trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)  #计算回归系数
        errorCount = 0; numTestVec = 0.0
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr =[]
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
                errorCount += 1
        errorRate = (float(errorCount)/numTestVec)
        print "the error rate of this test is: %f" % errorRate
        return errorRate

    def multiTest():           #调用colicTest()10次并求平均值
        numTests = 10; errorSum=0.0
        for k in range(numTests):
            errorSum += colicTest()
        print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

在python提示符下，敲下如下代码：

    logRegres.multiTest()

输出：

    the error rate of this test is: 0.313433
    the error rate of this test is: 0.402985
    the error rate of this test is: 0.298507
    the error rate of this test is: 0.402985
    the error rate of this test is: 0.388060
    the error rate of this test is: 0.313433
    the error rate of this test is: 0.328358
    the error rate of this test is: 0.343284
    the error rate of this test is: 0.283582
    the error rate of this test is: 0.388060
    after 10 iterations the average error rate is: 0.346269



