数据统计分析

基本统计特征函数

| 方法名     | 函数功能                                       | 所属库 |
| ---------- | ---------------------------------------------- | ------ |
| sum()      | 计算数据样本的总和(按列计算)                   | pandas |
| mean()     | 计算数据样本的算术平均数                       | pandas |
| var()      | 算数据样本的方差                               | pandas |
| std()      | 计算数据样本的标准差                           | pandas |
| corr()     | 计算数据样本的Spearman(Pearson)相关系数矩阵    | pandas |
| cov()      | 计算数据样本的协方差矩阵                       | pandas |
| skew()     | 样本值的偏度(三阶矩)                           | pandas |
| kurt()     | 样本值的峰度(四阶矩)                           | pandas |
| describe() | 给出样本的基本描述(基本统计量如均值、标准差等) | pandas |



拓展统计特征函数

pandas 累积计算统计特征函数

| 方法名    | 函数功能                       | 所属库 |
| --------- | ------------------------------ | ------ |
| cumsum()  | 依次给出前 L,2,..,n个数的和    | pandas |
| cumprod() | 依次给出前 1,2,”.,n个数的积    | pandas |
| cummax()  | 依次给出前1,2，.,n个数的最大值 | pandas |
| cummin()  | 依次给出前1,2，·,n个数的最小值 | pandas |

pandas滚动计算统计特征函数

| 方法名         | 函数功能                                    | 所属库 |
| -------------- | ------------------------------------------- | ------ |
| rolling_sum()  | 计算数据样本的总和(按列计算)                | pandas |
| rolling_mean() | 计算数据样本的算术平均数                    | pandas |
| rolling_var()  | 计算数据样本的方差                          | pandas |
| rolling_std()  | 计算数据样本的标准差                        | pandas |
| rolling_corr() | 计算数据样本的Spearman(Pearson)相关系数矩阵 | pandas |
| rolling_cov()  | 计算数据样本的协方差矩阵                    | pandas |
| rolling skew() | 样本值的偏度(三阶矩)                        | pandas |
| rolling kurt() | 样本值的峰度(四阶矩)                        | pandas |

统计绘图函数

Python主要统计绘图函数

| 绘图函数名         | 绘图函数功能                             | 所属工具箱        |
| ------------------ | ---------------------------------------- | ----------------- |
| plot()             | 绘制线性二维图，折线图                   | Matplotlib/pandas |
| pie()              | 绘制饼图                                 | Matplotlib/pandas |
| hist()             | 绘制二维条形直方图，可显示数据的分配情形 | Matplotlib/pandas |
| boxplot()          | 绘制样本数据的箱型图                     | pandas            |
| plot(logy = True)  | 绘制y轴的对数图形                        | pandas            |
| plot(yerr = error) | 绘制误差条形图                           | pandas            |

绘制饼图

```python
 
```



绘制二维条形直方图

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(1000)    # 1000个服从正态分布的随机数
plt.hist(x, 10)              # 分成10组绘制直方图
plt.show()
```



绘制箱型图

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x=np.randomrandn(1000)          #1000个服从正态分布的随机数
D=pd.DataFrame([x, x+1]).T       # 构造两列的DataFrame
D.plot(kind='box')               #调用Series内置的绘图方法画图，用kind参数指定箱型图(box)
plt.show()
```



使用plot(logy=True)函数进行绘图

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号
import numpy as np
import pandas as pd
x = pd.Series(np.exp(np.arange(20)))        #原始数据
plt.figure(figsize=(8,9))                   # 设置画布大小
ax1=plt.subplot(2,1,1)
x.plot(label='原始数据图', legend=True)
ax1 =plt.subplot(2,1,2)
x.plot(logy=True, label='对数数据图',legend=True)
plt.show()

```



绘制误差棒图

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simHei'] #用米正常显示中文标签
plt.rcParams['axes.unicode minus'] = False  #用米正常显示负号
import numpy as np
import pandas as pd

error = np.random.randn(10)
y = pd.Series(np.sin(np.arange(10)))
y.plot(yerr=error)
plt.show()
```



定量数据分析

```python
import pandas as pd
import numpy as np
catering_sale = 'chater.xls'# 餐饮数据
data = pd.read_excel(catering_sale,names=['date','sale'])

bins = [0,500,1000,1500,2000,2500,3000,3500,4000]
labels = ['[0,500)','[500,1000)','[1000,1500)','[1500,2000)','[2000,2500)','[2500,3000)','[3000,3500)' ,'[3500,4000)']

data['sale分层'] = pd.cut(data.sale,bins,labels=labels)
aggResult = data.groupby(by=['sale分层'])['sale'].agg({'sale': np.size})
pAggResult = round(aggResult/aggResult.sum(),2,) * 100

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6)) # 设置图框大小寸
pAggResult['sale'].plot(kind='bar',width=0.8,fontsize=10) # 绘制频率直方图
plt.rcParams['font.sans-serif'] = ['SimHei']    #用来正常显示中文标签
plt.title('季度销售额频率分布直方图’,fontsize=20')
plt.show()
```

D:\数据处理与分析\date-code\code-python\code-python\chapter3\demo\code



对比分析

```python
# 部门之间销售金额比较
import pandas as pdimport matplotlib.pyplot as pltdata=pd.read_excel("../data/dish_sale.xls")
plt.figure(figsize=(8,4))
plt.plot(data['月份'],data['A部门'],color='green',label='A部门',marker='o')
plt.plot(data['月份'],data['B部门'],color='red',label='B部门',marker='s')
plt.plot(data['月份'],data['c部门'], color='skyblue',label='C部门',marker='x')
plt.legend() # 显示图例
plt.ylabel('销售额 (万元)')
plt.show()

# B部门各年份之间销售金额的比较
data=pd.read_excel("../data/dish_sale_b.xls")
plt.figure(figsize=(8,4))
plt.plot(data['月份'],data['2012年'],color='green',label='2012年',marker='o')
plt.plot(data['月份'],data['2013年'],color='red',label='2013年',marker='s')
plt.plot (data['月份'],data['2014年'], color='skyblue',label='2014年',marker='x')
plt.legend() # 显示图例
plt.ylabel('销售额 (万元)')
plt.show()
```



统计量分析

```python
# 餐饮销量数据统计量分析
import pandas as pd

catering_sale = '../data/catering_sale.xls'                             # 餐饮数据
data = pd.read_excel(catering_sale,index_col='日期')                    # 读取数据，指定“日期”列为索引列

data = data[(data['销量'] > 400)&(data['销量'] < 5000)]                 # 过滤异常数据
statistics = data.describe()                                            #保存基本统计量

statistics.loc['range'] = statistics.loc['max']-statistics.loc['min']  # 极差
statistics.loc['var']=statistics.loc['std']/statistics.loc['mean'] # 变异系数
statistics.loc['dis'] = statistics.loc['75%']-statistics.loc['25%'] # 四分位数间距

print(statistics)
```

D:\数据处理与分析\date-code\code-python\code-python\chapter3\demo\code



周期性分析

```python
import pandas as pd
import matplotlib.pyplot as plt

df_normal = pd.read_csv("../data/user.csv")
plt.figure(figsize=(8,4))
plt.plot (df_normal["pate"],df_normal["Eletricity"])
plt.xlabel("日期")
#设置x轴刻度间隔
x_major_locator = plt.MultipleLocator(7)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.ylabel("每日电量")
plt.title("正常用户电量趋势")
plt.rcParams['font.sans-serif'] =['SimHei']# 用来正常显示中文标签
# plt.axis( equal')
plt.show()          # 展示图片

#窃电用户用电趋势分析
df_steal = pd.read_csv("../data/Steal user.csv")
plt.figure(figsize=(10,9))
plt.plot(df_steal["Date"],df_steal["Eletricity"])
plt.xlabel("日期")
plt.ylabel("日期")
#设置x轴刻度间隔
x_major_locator = plt.MultipleLocator(7)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.title("窃电用户电量趋势")
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.show() #展示图片
```



贡献度分析

```python
# 莱品盈利数据帕累托图
import pandas as pd

# 初始化参数
dish_profit='../data/catering_dish_profit.xls'          # 餐饮莱品盈利数据
data = pd.read_excel(dish_profit,index_col='菜品名')
data = data['盈利'].copy()
data.sort_values(ascending=False)

import matplotlib.pyplot as plt                         # 导入图像库
plt.rcParams['font.sans-serif'] = ['SimHei']            # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False              # 用来正常显示负号

plt.figure()
data.plot(kind='bar')
plt.ylabel('盈利(元) ')
p = 1.0*data.cumsum()/data.sum()
p.plot(color='r',secondary_y=True,style='-o',linewidth=2)
plt.annotate(format (p[6],'.4%'),xy=(6,p[6]), xytext=(6*0.9,p[6]*0.9), arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")) # 添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.ylabel('盈利(比例)')
plt.show()
```

D:\数据处理与分析\date-code\code-python\code-python\chapter3\demo\code



相关性分析

```python
# 餐饮销量数据相关性分析
import pandas as pd

catering_sale = '../data/catering_sale_all.xls'         # 餐饮数据，含有其他属性
data = pd.read_excel(catering_sale,index_col='日期')   # 读取数据，指定“日期”列为索引列

print(data.corr())                                      # 相关系数矩阵，即给出了任意两款菜式之间的相关系数

print(data.corr()['百合酱蒸凤爪'])                         #只显示“百合酱蒸凤爪”与其他菜式的相关系数

# 计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数
print (data['百合酱蒸凤爪'].corr(data['翡翠蒸香茜饺']))
```

D:\数据处理与分析\date-code\code-python\code-python\chapter3\demo\code



异常值分析

```python
#-*- coding: utf-8 -*-
import pandas as pd

catering_sale = '.\catering_sale.xls' #餐饮数据
data = pd.read_excel(catering_sale, index_col = u'日期') #读取数据，指定“日期”列为索引列

import matplotlib.pyplot as plt #导入图像库
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.figure() #建立图像
p = data.boxplot(return_type='dict') #画箱线图，直接使用DataFrame的方法
x = p['fliers'][0].get_xdata() # 'flies'即为异常值的标签
y = p['fliers'][0].get_ydata()
y.sort() #从小到大排序，该方法直接改变原对象

#用annotate添加注释
#其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
#以下参数都是经过调试的，需要具体问题具体调试。
for i in range(len(x)): 
  if i>0:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
  else:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))

plt.show() #展示箱线图

```

D:\数据处理与分析\date-code\code-python\code-python\chapter3\demo\code



一致性分析

数据不一致性是指数据的矛盾性、不相容性。直接对不一致的数据进行挖掘，可能会产生与实际相违背的挖掘结果。 
在数据挖掘过程中，不一致数据的产生主要发生在数据集成的过程中，可能是由于被挖掘数据来自于不同的数据源、对于重复存放的数据未能进行一致性更新造成的。例如，两张表中都存储了用户的电话号码，但在用户的电话号码发生改变时只更新了一张表中的数据，那么这两张表中就有了不一致的数据



缺失值分析



数据清洗

```python
#拉格朗日插值代码
import pandas as pd #导入数据分析库Pandas
from scipy.interpolate import lagrange #导入拉格朗日插值函数

inputfile = '../data/catering_sale.xls' #销量数据路径
outputfile = '../tmp/sales.xls' #输出数据路径

data = pd.read_excel(inputfile) #读入数据
data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None #过滤异常值，将其变为空值

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) #插值并返回插值结果

#逐个元素判断是否需要插值
for i in data.columns:
  for j in range(len(data)):
    if (data[i].isnull())[j]: #如果为空即插值。
      data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outputfile) #输出结果，写入文件

```

D:\数据处理与分析\date-code\code-python\code-python\chapter3\demo\code

D:\数据处理与分析\date-code\code-python\code-python\chapter4\demo\tmp\sales.xls



数据集成

```python
#-*- coding: utf-8 -*-
#线损率属性构造
import pandas as pd

#参数初始化
inputfile= '../data/electricity_data.xls' #供入供出电量数据
outputfile = '../tmp/electricity_data.xls' #属性构造后数据文件

data = pd.read_excel(inputfile) #读入数据
data[u'线损率'] = (data[u'供入电量'] - data[u'供出电量'])/data[u'供入电量']

data.to_excel(outputfile, index = False) #保存结果

```

D:\数据处理与分析\date-code\date-python\chapter4\demo\data



数据归约

