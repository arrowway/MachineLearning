##������ Logistic�ع�

����������һЩ���ݵ㣬��������һ��ֱ�߶���Щ�������ϣ����߳�Ϊ������ֱ�ߣ��������Ϲ��̾ͳ����ع顣

Ϊ��ʵ��`Logistic`�ع�����������ǿ�����ÿ�������϶�����һ���ع�ϵ����Ȼ������еĽ��ֵ��ӣ�������������`Sigmoid`�����У������õ�һ����Χ��0-1֮�����ֵ���κδ���0.5�����ݱ�����1�࣬С��0.5��������0�ࡣ

###1.`Sigmoid`�����������Ϊ \(z\),������Ĺ�ʽ�ó���

$$ z = {\omega_0}{x_0} + {\omega _1}{x _1} + {\omega _2}{x _2} + ... + {\omega _n}{x _n}$$

###2.�ݶ��㷨�ĵ�����ʽΪ��
$$ \omega  = \omega  + \alpha {\nabla _\omega }f(\omega ) $$
$$ \theta _j = \theta _j + \alpha({y^i - {h _\theta} ({x^i})}) * {x _j^i} $$

###3.`Logistic`�ع��ݶ������Ż��㷨

**ԭʼ���ݣ�**  
![](http://7xt9dp.com1.z0.glb.clouddn.com/5-3.png)

**Logistic�ع��ݶ������Ż��㷨-python����:**

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
        maxCycles = 500                         #��������
        weights = ones((n,1))
        for k in range(maxCycles):              #heavy on matrix operations
            h = sigmoid(dataMatrix*weights)     #matrix mult
            error = (labelMat - h)              #vector subtraction
            weights = weights + alpha * dataMatrix.transpose()* error #matrix mult �����ݶ�������ʽ���²���
        return weights                          #����ֵΪ[w0,w1,w2]

��python��ʾ���£��������´��룺

    dataArr, labelMat = logRegres.loadDataSet()
    logRegres.gradAscent(dataArr, labelMat)
    Out[12]: 
    matrix([[ 4.12414349],
            [ 0.48007329],
            [-0.6168482 ]])

###4.�������ݣ��������߽߱�

** �������ݼ���Logistic�ع�������ֱ�ߵĺ��� **

    def plotBestFit(weights):              #�˴�weights��Ϊ���������[w0,w1,w2]
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
       y = (-weights[0]-weights[1]*x)/weights[2] # 0=w0x0+w1x1+w2x2 (x0=1) �˴�x=x1��y=x2
       ax.plot(x, y)
       plt.xlabel('X1'); plt.ylabel('X2');
       plt.show() 

��python��ʾ���£��������´��룺

    logRegres.plotBestFit(weights.getA())  #getA() ������ת��Ϊ����array

**���н��Ϊ��**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-4.png)

###5.ѵ���㷨������ݶ����� 

�Ľ��㷨��һ�ν���һ�������������»ع�ϵ������**����ѧϰ�㷨**

** ����ݶ������㷨 **

    def stocGradAscent0(dataMatrix, classLabels):
        m,n = shape(dataMatrix)
        alpha = 0.01
        weights = ones(n)   #initialize to all ones
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
        return weights

��python��ʾ���£��������´��룺

    weights = logRegres.stocGradAscent0(array(dataArr), labelMat)

**���н��Ϊ��**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-5.png)

**�Ľ�������ݶ������㷨 **

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

1. ����alphaÿ�ε���ʱ������������Ի������ݲ������߸�Ƶ�Ŷ�����alpha�����ŵ�������**���ϼ�С**������Զ�����С��0��**������**���������Ա�֤��ε�������������Ȼ��Ӱ�졣���Ҫ����������Ƕ�̬�仯�ģ������ʵ��Ӵ����ȷ����ֵ��ø���Ļع�ϵ������\( j << max(i) \)ʱ������alpha�����ϸ��½��ġ�**��������ϸ��½�**Ҳ������ģ���˻�������Ż��㷨�С�
2. **���ѡȡ����ֵ**�����»ع�ϵ�������ַ��������������ԵĲ�����������ʵ�ַ�����ÿ�������������ѡ��һ��ֵ�����»ع�ϵ������ɾ�����ٽ�����һ�ε�����

**�Ľ���ع�ϵ���仯���**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-7.png)

**�����ٶȸ��죬������С**

��python��ʾ���£��������´��룺

    dataArr, labelMat = logRegres.loadDataSet()
    weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
    logRegres.plotBestFit(weights)

**���н��Ϊ��**

![](http://7xt9dp.com1.z0.glb.clouddn.com/5-8.png)

###6. ʾ������������֢Ԥ�ⲡ���������

**Logistic�ع���ຯ��**

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
        trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)  #����ع�ϵ��
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

    def multiTest():           #����colicTest()10�β���ƽ��ֵ
        numTests = 10; errorSum=0.0
        for k in range(numTests):
            errorSum += colicTest()
        print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

��python��ʾ���£��������´��룺

    logRegres.multiTest()

�����

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



