import paddle
import paddlehub as hub
from paddle.io import DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pyecharts.charts import Line, Bar, Pie
import pyecharts.options as opts
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
# 导入数据、查看数据
path = "/机器学习/k-means/附件.csv"
df = pd.read_csv(path, encoding='gbk')
print(df.shape)
print(df.info())
df.head()
# 销售日期转化为datetime，提取月、日
# 2015年没有2月29日，所以删除2月29日的两条数据
df.drop(df[df['销售日期'] == 20150229].index.tolist(),inplace = True)
# 将销售日期转换为datatime格式
df['销售日期'] = pd.to_datetime(df['销售日期'].astype(str),format = '%Y-%m-%d')
df.head()
# 提取日期中的周
df['销售周'] = df['销售日期'].dt.isocalendar().week
# 提取日期中的日
df['销售日'] = df['销售日期'].dt.day
# 删除缺失值
# 查看缺失的数据有哪些
df.isnull().sum()
df[df['销售数量'].isnull()]
# 销售数量有两个缺失值，删除缺失值
df.dropna(subset = ['销售数量'],inplace = True)
df.isnull().sum()
# 查看异常值以及去除异常值
# 销售数量有88条小于0的数据，初步判断可能是由于退货以及销售人员的操作失误导致的
print('销售金额为负数的数量为：'+str(df[df['销售数量'] <= 0].shape[0]))
df[df['销售数量'] <= 0].head()
# 销售金额有93条小于等于0的数据，初步判断可能是由于退货以及销售人员的操作失误导致的
print(df[df['销售金额'] <= 0].shape)
df[df['销售金额'] <= 0].head(2)
# 销售单价只有2条等于0的数据，应该是业务人员误操作导致,直接进行删除
df[df['商品单价'] <= 0]
# 如果是退货的话，销售金额为负，而销售金额等于0的数据应该为收银员的误操作，进行删除
df[df['销售金额'] == 0]
# 删除商品单价和销售金额为0的数据
df.drop(index = df[df['商品单价']==0].index,axis = 0,inplace = True)
df.drop(index = df[df['销售金额'] == 0].index,axis = 0,inplace = True)
# 加一个字段，判断是否退货，将销售金额小于0的数据标记为退货
df['是否退货'] = 0
df.loc[df[df['销售金额'] < 0].index,'是否退货'] = 1
df[df['销售金额'] < 0].head()
# 因为存在单价*数量不等于销售金额的情况，如下
print('单价*销量 != 销售额的数据有:',df[(df['商品单价'] * df['销售数量']) != df['销售金额']].shape[0])
df[(df['商品单价'] * df['销售数量']) != df['销售金额']].head()
# 因为存在单价*数量不等于销售金额的情况，如下
print('单价*销量 != 销售额的数据有:',df[(df['商品单价'] * df['销售数量']) != df['销售金额']].shape[0])
df[(df['商品单价'] * df['销售数量']) != df['销售金额']].head()
# 去掉不会用到的字段
col_lis = ['大类编码','中类编码','小类编码','商品编码']
for i in col_lis:
    del df[i]
df.head()
# 统计每个月的销售总额变化（可视化展示）
month_money_sum = df.groupby('销售月份')['销售金额'].sum().round(2)
line = (
    Line()
    .add_xaxis(xaxis_data = month_money_sum.index.astype(str).tolist())
    .add_yaxis(
        series_name = '月销售总额',
        y_axis = month_money_sum.values.tolist())
    .set_global_opts(title_opts = opts.TitleOpts(title = '平均日销售额变化'))
)
line.render("01平均日销售额变化.html")
# 统计每个月的日平均销售额变化（可视化展示）
day_money_sum = pd.pivot_table(df,values = '销售金额',index = '销售月份',columns = '销售日',aggfunc = 'sum')
month_money_avg = day_money_sum.loc[:,1:].mean().round(2)
line = (
    Line()
    .add_xaxis(xaxis_data = month_money_avg.index.astype(str).tolist())
    .add_yaxis(
        series_name = '平均日销售额',
        y_axis = month_money_avg.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
                opts.MarkPointItem(type_="max", name="最大值"),
                opts.MarkPointItem(type_="min", name="最小值"),
                opts.MarkPointItem(type_="average", name="平均值")
            ]
            )
        )
    .set_global_opts(title_opts = opts.TitleOpts(title = '平均日销售额变化'))
    )
line.render("02平均日销售额变化.html")
# 一月份平均每天的销售额变化
month1_money = day_money_sum.iloc[0, :].round(0)
month2_money = day_money_sum.iloc[1, :].round(0)
month3_money = day_money_sum.iloc[2, :].round(0)
month4_money = day_money_sum.iloc[3, :].round(0)
line = (
    Line()
        .add_xaxis(xaxis_data=month1_money.index.astype(str).tolist())
        .add_yaxis(
        series_name='1月份日销售额',
        y_axis=month1_money.values.tolist(),
        is_smooth=True,
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=3)
    )

        .add_yaxis(
        series_name='2月份日销售额',
        y_axis=month2_money.values.tolist(),
        is_smooth=True,
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=3)
    )

        .add_yaxis(
        series_name='3月份日销售额',
        y_axis=month3_money.values.tolist(),
        is_smooth=True,
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=3)
    )

        .add_yaxis(
        series_name='4月份日销售额',
        y_axis=month4_money.values.tolist(),
        is_smooth=True,
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=3)
    )

        .set_global_opts(title_opts=opts.TitleOpts(title='日销售额变化'))
)
line.render("03日销售额变化.html")
# 去掉2月份的数据，查看1,3,4月的平均日销售额的变化趋势
month_money_avg2 = day_money_sum.iloc[[0,2,3],:].mean().round(0)
line = (
    Line()
    .add_xaxis(xaxis_data = month_money_avg2.index.astype(str).tolist())
    .add_yaxis(
        series_name = '平均日销售额',
        y_axis = month_money_avg2.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
                opts.MarkPointItem(type_="max", name="最大值"),
                opts.MarkPointItem(type_="min", name="最小值"),
                opts.MarkPointItem(type_="average", name="平均值")
            ]
            )
        )
    .set_global_opts(title_opts = opts.TitleOpts(title = '平均日销售额变化'))
    )
line.render("04平均日销售额变化.html")

# 统计每个周的周销售总额变化（和所有周销售总额均值对比）
week_money_sum = df.groupby('销售周')['销售金额'].sum().round(0).reset_index().\
sort_values(by = '销售周',ascending = True)
line = (
    Line()
    .add_xaxis(xaxis_data = week_money_sum['销售周'].tolist())
    .add_yaxis(
        series_name = '周销售额',
        y_axis = week_money_sum['销售金额'].tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
                opts.MarkPointItem(type_="max", name="最大值"),
                opts.MarkPointItem(type_="min", name="最小值"),
                opts.MarkPointItem(type_="average", name="平均值")
            ]
            )
        )
    .set_global_opts(title_opts = opts.TitleOpts(title = '周销售额变化'))
    )
line.render("05周销售额变化.html")

# 统计所有商品类型的销售额
type_money = df.groupby('商品类型')['销售金额'].sum().round(2)
pie = (
    Pie()
    .add(
        "",
        [list(i) for i in zip(type_money.index.tolist(),type_money.values.tolist())],
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="商品类型销售额占比"),
    )
)
pie.render("06商品类型销售额占比.html")


# 统计大类名称的销售额（排名）
type_money = df.groupby('大类名称')['销售金额'].sum().round(2).sort_values(ascending = False)
bar = (
    Bar()
    .add_xaxis(type_money.index.tolist())
    .add_yaxis(
        '销售金额',
        type_money.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="商品大类销售额排名"),
    )
)
bar.render("07商品大类销售额排名.html")
# 统计中类名称的销售额（排名）
type_money = df.groupby('中类名称')['销售金额'].sum().round(2).sort_values(ascending = False).iloc[:10]
type_money
bar = (
    Bar()
    .add_xaxis(type_money.index.tolist())
    .add_yaxis(
        '销售金额',
        type_money.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="商品中类销售额前十排名"),
    )
)
bar.render("08商品中类销售额前十排名.html")
# 统计小类名称的销售额（排名）
type_money = df.groupby('小类名称')['销售金额'].sum().round(2).sort_values(ascending = False).iloc[:10]
type_money
bar = (
    Bar()
    .add_xaxis(type_money.index.tolist())
    .add_yaxis(
        '销售金额',
        type_money.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="商品小类销售额前十排名"),
    )
)
bar.render("09商品小类销售额前十排名.html")
# 统计一月销售额排名前十的小类
money_rank = df[df['销售月份'] == 201501].groupby('小类名称')['销售金额'].sum().round(2).\
sort_values(ascending = False).reset_index().iloc[:10].sort_values(by = '销售金额')
bar = (
    Bar()
    .add_xaxis(money_rank['小类名称'].tolist())
    .add_yaxis(
        '销售金额',
        money_rank['销售金额'].tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="一月份商品小类销售额前十排名"),
    )
    )
bar.render("10一月份商品小类销售额前十排名.html")
# 统计二月销售额排名前十的小类
money_rank = df[df['销售月份'] == 201502].groupby('小类名称')['销售金额'].sum().round(2).\
sort_values(ascending = False).reset_index().iloc[:10].sort_values(by = '销售金额')
bar = (
    Bar()
    .add_xaxis(money_rank['小类名称'].tolist())
    .add_yaxis(
        '销售金额',
        money_rank['销售金额'].tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="二月份商品小类销售额前十排名"),
    )
    )
bar.render("11二月份商品小类销售额前十排名.html")
# 统计三月销售额排名前十的小类
money_rank = df[df['销售月份'] == 201503].groupby('小类名称')['销售金额'].sum().round(2).\
sort_values(ascending = False).reset_index().iloc[:10].sort_values(by = '销售金额')
bar = (
    Bar()
    .add_xaxis(money_rank['小类名称'].tolist())
    .add_yaxis(
        '销售金额',
        money_rank['销售金额'].tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="三月份商品小类销售额前十排名"),
    )
    )
bar.render("12三月份商品小类销售额前十排名.html")
# 统计四月销售额排名前十的小类
money_rank = df[df['销售月份'] == 201504].groupby('小类名称')['销售金额'].sum().round(2).\
sort_values(ascending = False).reset_index().iloc[:10].sort_values(by = '销售金额')
bar = (
    Bar()
    .add_xaxis(money_rank['小类名称'].tolist())
    .add_yaxis(
        '销售金额',
        money_rank['销售金额'].tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="四月份商品小类销售额前十排名"),
    )
    )
bar.render("13四月份商品小类销售额前十排名.html")
# 根据顾客编号进行分组，计算顾客的购买次数、购买总金额、购买各类型商品的金额、客单价，生成一张顾客行为的dataframe
# 查看总计有多少顾客来超市买过东西
print('总计有',df['顾客编号'].nunique(),'名顾客来超市购买过商品.')

# 1.计算客户购买次数
# 注：这里直接用count计数的话是不合理的，因为一个客户可能一天内买过七八种商品，但实际上他的购买次数却只有这一次
df.groupby('顾客编号')['销售日期'].count().head()
# 例如顾客编号为0的这名顾客，他在2015-01-01购买了四种商品，在2015-04-25购买了一种商品，其实他只在这个超市购买过两次，应该算作两次的购买次数的。
df[df['顾客编号'] == 0]


# 所以我们这里统计次数的时候要去重一下，使用nunique()，这样统计出来的0号顾客购买次数是2就没问题了
consume_counts = df.groupby('顾客编号')['销售日期'].nunique().reset_index().rename(columns = {'销售日期':'消费次数'})
consume_counts.head()
# 2.计算客户购买总金额
consume_money = df.groupby('顾客编号')['销售金额'].sum().reset_index().rename(columns = {'销售金额':'消费总金额'})
consume_money.head()

# 3.计算客户购买各大类商品的金额
consume_type = pd.pivot_table(df,index = '顾客编号',columns = '大类名称',values = '销售金额',aggfunc = np.sum)
consume_type = consume_type.fillna(0).reset_index()
consume_type.head()
# 4.计算每个客户平均单次购买金额
# 先将上边的三个表连接起来
consume = pd.concat([consume_counts,consume_money,consume_type],axis = 1)
consume.head()
# 将数据拼接好后，我发现有三个顾客编号字段，关于如何去掉这三个重复的字段，我也是想了很久，直接del和drop都不太可行，会把三个字段都删掉
# 索引出来没有重复字段的所有列也不太好操作，于是我求助了群里的小伙伴，他们给出了如下解决方案
# 直接取反进行字段名的去重
consume = consume.loc[:,~consume.columns.duplicated()]
consume.head()
# 我想到的就是改成用merge连接，比较笨的方法，要是有十来张表，人就傻了
consume = pd.merge(pd.merge(consume_counts,consume_money,how = 'outer',on = '顾客编号'),consume_type,how = 'outer',on = '顾客编号')
consume.head()
# 下边来计算每个客户平均单次购买金额
consume['平均单次购买金额'] = consume['消费总金额'] / consume['消费次数']
consume['平均单次购买金额'] = consume['平均单次购买金额'].round(2)
consume.head()

# 现在已经生成了顾客购买行为表，下边来进行一下统计分析
# 1. 购买次数前10的顾客
consume[['顾客编号','消费次数']].sort_values(by = '消费次数',ascending = False).iloc[:10,:]

# 2. 消费金额前十的顾客
consume[['顾客编号','消费总金额']].sort_values(by = '消费总金额',ascending = False).iloc[:10,:]
# 3. 平均单次消费金额前十的顾客
consume[['顾客编号','平均单次购买金额']].sort_values(by = '平均单次购买金额',ascending = False).iloc[:10,:]

# 消费次数前十的顾客都会购买哪种商品
count_consume = consume.sort_values(by='消费次数', ascending=False).iloc[:10, 3:18].sum(). \
    round(2).sort_values(ascending=True)

bar = (
    Bar()
        .add_xaxis(count_consume.index.tolist())
        .add_yaxis(
        '销售金额',
        count_consume.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(  # 标记点配置项
            data=[
                opts.MarkPointItem(type_="max", name="最大值"),
                opts.MarkPointItem(type_="min", name="最小值"),
                opts.MarkPointItem(type_="average", name="平均值")
            ]
        )
    )
        .reversal_axis()
        .set_series_opts(label_opts=opts.LabelOpts(position='right'))
        .set_global_opts(
        title_opts=opts.TitleOpts(title="消费次数前十的顾客购买商品排名"),
    )
)
bar.render("14消费次数前十的顾客购买商品排名.html")

# 消费消费总金额前十的顾客都会购买哪种商品
count_consume = consume.sort_values(by='消费总金额', ascending=False).iloc[:10, 3:18].sum(). \
    round(2).sort_values(ascending=True)

bar = (
    Bar()
        .add_xaxis(count_consume.index.tolist())
        .add_yaxis(
        '销售金额',
        count_consume.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(  # 标记点配置项
            data=[
                opts.MarkPointItem(type_="max", name="最大值"),
                opts.MarkPointItem(type_="min", name="最小值"),
                opts.MarkPointItem(type_="average", name="平均值")
            ]
        )
    )
        .reversal_axis()
        .set_series_opts(label_opts=opts.LabelOpts(position='right'))
        .set_global_opts(
        title_opts=opts.TitleOpts(title="消费总金额前十的顾客购买商品排名"),
    )
)
bar.render("15消费总金额前十的顾客购买商品排名.html")

# 平均单次购买金额前十的顾客都会购买哪种商品
count_consume = consume.sort_values(by = '平均单次购买金额',ascending = False).iloc[:10,3:18].sum().\
round(2).sort_values(ascending = True)

bar = (
    Bar()
    .add_xaxis(count_consume.index.tolist())
    .add_yaxis(
        '销售金额',
        count_consume.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="平均单次购买金额前十的顾客购买商品排名"),
    )
    )
bar.render("16平均单次购买金额前十的顾客购买商品排名.html")

# 由于数据有次数、金额、总金额等，量纲不同，可能导致聚类结果不好，所以我们先进行数据标准化
kmean_data = consume.iloc[:,1:]
min_max = MinMaxScaler()
stand_data = min_max.fit_transform(kmean_data)
# 使用手肘法看K值
SSE = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=100)
    kmeans.fit(stand_data)
    SSE.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), SSE, marker='o')
plt.show()

# 使用轮廓系数法看k值
score = []
for i in range(2, 12):
    kmeans = KMeans(n_clusters=i, random_state=100)
    result = kmeans.fit(stand_data)
    score.append(silhouette_score(stand_data, result.labels_))

plt.figure(figsize=(10, 5))
plt.plot(range(2, 12), score, marker='o')
plt.show()

# 将数据集聚为3类,生成客户类别字段
kmeans = KMeans(n_clusters=3)
result = kmeans.fit(stand_data)
# y_kmeans = kmeans.predict(kmean_data)
consume['客户类别'] = result.labels_
consume.head()
# 求每一类客户各项的均值
consume_des = consume.groupby('客户类别').mean().round(2).iloc[:,1:]
consume_des

# 查看各类客户在各项消费的金额
consume_1 = consume_des.iloc[0,2:18].sort_values()
bar = (
    Bar()
    .add_xaxis(consume_1.index.tolist())
    .add_yaxis(
        '销售金额',
        consume_1.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="第0类客户特征"),
    )
    )
bar.render("17.第0类客户特征.html")
# 查看各类客户在各项消费的金额
consume_1 = consume_des.iloc[1,2:17].sort_values()
bar = (
    Bar()
    .add_xaxis(consume_1.index.tolist())
    .add_yaxis(
        '销售金额',
        consume_1.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="第1类客户特征"),
    )
    )
bar.render("18.第1类客户特征.html")

# 查看各类客户在各项消费的金额
consume_1 = consume_des.iloc[2,2:17].sort_values()
bar = (
    Bar()
    .add_xaxis(consume_1.index.tolist())
    .add_yaxis(
        '销售金额',
        consume_1.values.tolist(),
        markpoint_opts=opts.MarkPointOpts(    # 标记点配置项
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值"),
            opts.MarkPointItem(type_="average", name="平均值")
        ]
        )
        )
    .reversal_axis()
    .set_series_opts(label_opts = opts.LabelOpts(position = 'right'))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="第2类客户特征"),
    )
    )
bar.render('19.第二类客户特征.html')







