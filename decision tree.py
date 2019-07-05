#寻找最合适的分割特征
#如果不能分割数据集，该数据集作为一个叶子节点。
#对数据集进行二分割
#对分割的数据集1重复1， 2，3 步，创建右子树。
#对分割的数据集2重复1， 2，3 步，创建左子树
import pandas as pd

class Tree:
    def __init__(self, value=None, rightBranch=None, leftBranch=None, 
                 results=None, col=-1, summary=None, data=None):
        self.value = value
        self.rightBranch = rightBranch
        self.leftBranch = leftBranch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data

def DataSet():
    source = [[0, 0, 0, 0, 'N'], 
              [0, 0, 0, 1, 'N'], 
              [1, 0, 0, 0, 'Y'], 
              [2, 1, 0, 0, 'Y'], 
              [2, 2, 1, 0, 'Y'], 
              [2, 2, 1, 1, 'N'], 
              [1, 2, 1, 1, 'Y']]
    df = pd.DataFrame(source, columns = ['outlook', 'temperature', 
                                              'humidity', 'windy', 'label'])
    return df

def calculateDiffCount(df):
    #辅助计算一个特征的种类；假设输入的column为为[0, 0, 0, 1, 2, 2, 2]，
    #则输出为{0:3, 1:1, 2:3}，这样分类统计每个类别的数量
    results = df['label'].value_counts()
    return results

def gini(df):
    #计算gini值
    #rows是某个特征Ai中，特征值相同的多个样本，比如，湿度都为0的几个样本
    length = len(df.index)
    results = calculateDiffCount(df)
    imp = 0
    for x in results:
        imp += results[x]/length * results[x]/length
    return round(1 - imp,3)

def splitDatas(dataset, value, column):
    #根据条件分离数据集(splitDatas by value,column)   
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    if(isinstance(value,int) or isinstance(value,float)): #for int and float type        
            df1 = dataset[dataset[column]>value]
            df2 = dataset[dataset[column]<value]
    else:                                                 #for String type
            df1 = dataset[dataset[column].isin([value])]
            df2 = dataset[~dataset[column].isin([value])]    
    return df1, df2

def buildDecisionTree(df,evaluationFunction=gini):
    #递归建立决策树,当gain = 0 时停止递归 
    currentGain = evaluationFunction(df)
    rows_length = len(df.index)#样本数
    best_gain = 0.0
    best_spilt = None
     
    #choose the best gain
    for col in df.columns:
        col_value_set = list(set(df[col]))
        for value in col_value_set:
            df1, df2 = splitDatas(df, value, col)
            p = len(df1.index)/rows_length
            gain = currentGain - p*evaluationFunction(df1) - (1-p)*evaluationFunction(df2)
            if gain > best_gain:
                best_gain = gain
                best_spilt = [col,value]
                #根据最佳分裂点分裂成两个子树
                right = df1
                left = df2
    #记录节点不纯度和样本数
    dcY = {'impurity' : '%.3f' % currentGain, 'samples' : '%d' % rows_length}
             
    #判断是否停止
    if best_gain > 0:
        rightBranch = buildDecisionTree(right,evaluationFunction)
        leftBranch = buildDecisionTree(left,evaluationFunction)
        return Tree(col=best_spilt[0],value=best_spilt[1],
                    righBranch=rightBranch,leftBranch=leftBranch, summary=dcY)
    else:
        return Tree(results=calculateDiffCount(df),summary=dcY,data=df)
    
def prune(tree,miniGain,evaluationFunction=gini):
    #剪枝, when gain < mini Gain，合并
     
    if tree.trueBranch.results == None:prune(tree.trueBranch,miniGain,evaluationFunction)
    if tree.falseBranch.results == None:prune(tree.falseBranch,miniGain,evaluationFunction)
         
    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = len(tree.rightBranch.data)
        len2 = len(tree.leftBranch.data)
        p = float(len1)/(len1 + len2)
        gain = evaluationFunction(tree.trueBranch.data + tree.falseBranch.data) - p * evaluationFunction(tree.trueBranch.data) - (1 - p) * evaluationFunction(tree.falseBranch.data)
        if(gain < miniGain):
            tree.data = tree.trueBranch.data + tree.falseBranch.data
            tree.results = calculateDiffCount(tree.data)
            tree.trueBranch = None
            tree.falseBranch = None
