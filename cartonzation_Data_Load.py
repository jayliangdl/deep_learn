# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:47:47 2017

@author: ljzon
"""

import pandas as pd;
import numpy as np;

classify_volumn_define=[0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000]
classify_len_define=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
classify_wid_define=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
classify_hgt_define=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
'''
Description:classify_volumne(x)
    
    本方法用于判断给出的体积x，判断它属于哪个体积分类。
    
Arg:

    x:为产品体积

Returns:

    返回属于哪个体积分类 
'''
def classify_volumne(x):
    r=0
    for i in range(len(classify_volumn_define)-1):
        if x>classify_volumn_define[i] and x<=classify_volumn_define[i+1]:
            r=i
            break
    return r

def classify_len(x):
    r=0
    for i in range(len(classify_len_define)-1):
        if x>classify_len_define[i] and x<=classify_len_define[i+1]:
            r=i
            break
    return r

def classify_wid(x):
    r=0
    for i in range(len(classify_wid_define)-1):
        if x>classify_wid_define[i] and x<=classify_wid_define[i+1]:
            r=i
            break
    return r

def classify_hgt(x):
    r=0
    for i in range(len(classify_hgt_define)-1):
        if x>classify_hgt_define[i] and x<=classify_hgt_define[i+1]:
            r=i
            break
    return r

'''
Description:cal_order_features(data)

    本方法用于产生每张order的feature数据
    
Args:

    入参data结构如下：
          prtnum       ctnnum     ordnum ctnsze  pckqty   untlen  untwid  unthgt
    0   A110415T3  01MA041MXYA  AB6536482      M       1   7.4803  4.9213  5.1181
    1  AWHU3124TW  01MA041MY0A  AB6536532      L       2  11.8110  5.5118  7.4016
    2     A270578  01MA041MY0A  AB6536532      L       1   9.4488  7.0866  3.9370
    3   A100186T6  01MA041MYAA  AB6536544      M       1  12.5980  6.6930  8.1890
    4   A108996R1  01MA041MYAA  AB6536544      M       1   3.5433  1.1024  9.4488

    例如订单AB6536532用一个L箱子装，分别装AWHU3124TW x 2个 + A270578 x 1个
    其中单个AWHU3124TW产品的长、宽、高分别是11.8110  5.5118  7.4016

Returns:

    本方法返回以下出参(例子)：
          ordnum  pckqty volumn_type
    0  AB6517449       1          13
    1  AB6519802       1           0
    2  AB6519802       2           1
    3  AB6519802       1           3
    4  AB6522461       1           0

    上面数据中s0表示某订单包含产品体积为0~50(包含50)的件数，s1表示某订单包含产品体积为50(不包含50)~100(包含100)的件数..
    s60表示某订单包含产品体积为2950(不包含2950)~3000(包含3000)的件数
    每50一个区间.
    我们把所有在售产品按体积划分60个区间（体积为0~50为“0类型”，50~100为“1类型”...体积为2950~3000为“第59类型”），
    本方法用于统计每张订单包含每个体积类型的产品件数。
    例如上面例子数据的第1行表示订单AB6517449包含体积为“13类型”的数量为1件；
                   第2行表示订单AB6519802包含体积为“0类型”的数量为1件；
                   第3行表示订单AB6519802包含体积为“2类型”的数量为1件；
                   
'''
def cal_order_features(data):
    ##print(data[data['ordnum']==926536532])
    volumn = data['untlen']*data['untwid']*data['unthgt']##计算每个订购行的体积。
    
    df_volumn = pd.DataFrame({
        'volumn':volumn.to_dict()
    })
    data = pd.concat([data, df_volumn], axis=1)##把“体积”字段合并添加到原来列表中。
    classify_volumne_func = np.frompyfunc(lambda x:classify_volumne(x),1,1)
    classify_len_func = np.frompyfunc(lambda x:classify_len(x),1,1)
    classify_wid_func = np.frompyfunc(lambda x:classify_wid(x),1,1)
    classify_hgt_func = np.frompyfunc(lambda x:classify_hgt(x),1,1)
    ##data['classify_volumne1'] = classify_volumne_result 
    classify_volumne_result = classify_volumne_func(data['volumn']) ##对每行的体积进行分类，例如如果体积是35，由于其<50，所以分为第0类
    classify_len_result = classify_len_func(data['untlen'])
    classify_wid_result = classify_len_func(data['untwid'])
    classify_hgt_result = classify_len_func(data['unthgt'])
    data['classify_volumne_result'] = classify_volumne_result ##把分类后的数据添加到原来数据表中（增加字段classify_volumne_result）
    data['classify_len_result'] = classify_len_result
    data['classify_wid_result'] = classify_wid_result
    data['classify_hgt_result'] = classify_hgt_result
    data_volumn = data.copy()
    result_volumn = data_volumn.groupby(['ordnum','classify_volumne_result'])##
    result_volumn = result_volumn.sum()['pckqty']
    result_volumn = result_volumn.reset_index()

    data_len = data.copy()
    result_len = data_len.groupby(['ordnum','classify_len_result'])##
    result_len = result_len.sum()['pckqty']
    result_len = result_len.reset_index()
    
    data_wid = data.copy()
    result_wid = data_wid.groupby(['ordnum','classify_wid_result'])##
    result_wid = result_wid.sum()['pckqty']
    result_wid = result_wid.reset_index()
    
    data_hgt = data.copy()
    result_hgt = data_hgt.groupby(['ordnum','classify_hgt_result'])##
    result_hgt = result_hgt.sum()['pckqty']
    result_hgt = result_hgt.reset_index()
    
    result_volumn_df = pd.DataFrame({
        "ordnum":result_volumn['ordnum'],
        "type":result_volumn['classify_volumne_result'],
        "pckqty":result_volumn['pckqty'],
    })
    
    result_len_df = pd.DataFrame({
        "ordnum":result_len['ordnum'],
        "type":result_len['classify_len_result'],
        "pckqty":result_len['pckqty'],
    })
    
    result_wid_df = pd.DataFrame({
        "ordnum":result_wid['ordnum'],
        "type":result_wid['classify_wid_result'],
        "pckqty":result_wid['pckqty'],
    })
    
    result_hgt_df = pd.DataFrame({
        "ordnum":result_hgt['ordnum'],
        "type":result_hgt['classify_hgt_result'],
        "pckqty":result_hgt['pckqty'],
    })
    
    return result_volumn_df,result_len_df,result_wid_df,result_hgt_df


'''
Description: list_all_orders(data)

      本方法用于从列表中获取订单列表

    入参data结构如下：
          prtnum       ctnnum     ordnum ctnsze  pckqty   untlen  untwid  unthgt
    0   A110415T3  01MA041MXYA  AB6536482      M       1   7.4803  4.9213  5.1181
    1  AWHU3124TW  01MA041MY0A  AB6536532      L       2  11.8110  5.5118  7.4016
    2     A270578  01MA041MY0A  AB6536532      L       1   9.4488  7.0866  3.9370
    3   A100186T6  01MA041MYAA  AB6536544      M       1  12.5980  6.6930  8.1890
    4   A108996R1  01MA041MYAA  AB6536544      M       1   3.5433  1.1024  9.4488

    例如订单AB6536532用一个L箱子装，分别装AWHU3124TW x 2个 + A270578 x 1个
    其中单个AWHU3124TW产品的长、宽、高分别是11.8110  5.5118  7.4016

Returns:

    本方法返回以下出参(类型为Series)(例子)：
    0        AB6536482
    1        AB6536532
    3        AB6536544
    5        AB6536484
    6        AB6536542
    9        AB6536465
    10       AB6536486

'''
def list_all_orders(data):
    data = data.drop_duplicates(subset=['ordnum'],keep='first',inplace=False)##去掉重复的订单号记录
    orders = pd.Series(data['ordnum'])##重新对订单号产生一个Series
    return orders


'''
Description: init_orders_feature(orders,init_volumn_type)


    init_orders_feature(orders,init_volumn_type)方法用于初始化订单及其feature数据
    入参orders结构如下(类型为Series),数据例子如下：
    0        AB6536482
    1        AB6536532
    3        AB6536544
    5        AB6536484

Args:
    入参init_volumn_type结构如下(类型为numpy.ndarray),数据例子如下：
    因为我们对所有产品的体积分为60类(如体积为0~50视为第0类，体积为50~100视为第1类...体积为2950~3000视为第59类)，
    所以下面0~59的元素表示60类体积分类。
    另外60也不是不变的数字，如变量classify_volumn_define更新了，则其分类也可能变化。
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
     50 51 52 53 54 55 56 57 58 59]
 
Return:

    出参返回：
    ordnum(index)    ordnum(column)   volumn_type      pckqty
    AB6536482        AB6536482        0                    0
    AB6536482        AB6536482        1                    0
    AB6536482        AB6536482        2                    0
    ...
    AB6536482        AB6536482        59                   0

    AB6536532        AB6536532        0                    0
    AB6536532        AB6536532        1                    0
    AB6536532        AB6536532        2                    0
    ...
    AB6536532        AB6536532        59                   0

    初始化每张订单及其对应60个分类的件数(每张订单每个分类初始化件数为0)
    例如上面返回的例子中说明订单AB6536482在第0~59类中初始化件数都为0
'''
def init_orders_feature(orders,init_volumn_type):
    df1 = pd.DataFrame({
        "ordnum":orders,
        "key":0
    })
    df2 = pd.DataFrame({
        "type":init_volumn_type,
        "pckqty":0,
        "key":0
    })
    result = pd.merge(df1,df2,on=['key'])
    result = result.drop(['key'],axis=1)
    return result

'''
Description: convert_data(init_feature_df,order_feature_df)

    本方法用于整合数据.
    
Args:

    入参init_feature_df列出所有订单及其体积件数（每个元素初始化为0）。
    init_feature_df数据的例子如下：

           ordnum  pckqty  volumn_type
    0   AB6536482       0            0
    1   AB6536482       0            1
    2   AB6536482       0            2
    ...
    59  AB6536482       0            59
    
    60  AB6582395       0            0
    61  AB6582395       0            1
    62  AB6582395       0            2
    ...
    119 AB6582395       0            59
    ...
    ...
    

    入参order_feature_df列出实际订单中包含的体积分类及其件数。
    order_feature_df数据的例子如下：

          ordnum  pckqty volumn_type
     0  AB6517449       1          13
     1  AB6519802       1           0
     2  AB6519802       2           1
     3  AB6519802       1           3
     4  AB6522461       1           0
     5  AB6522461       2           1
     6  AB6522461       3           2
    例如某订单AB6517449包含第13类的体积1件。
    

Returns:

    本方法需要转换成以下格式：
          ordnum  s0  s1  s2  s3  s4  s5  s6  s7  s8 ...   s56  s57  s58  s59  
    0  AB6536484   4   1   1   1   0   0   0   0   0 ...     0    0    0    0 
    0  AB6522461   1   2   3   0   0   0   0   0   0 ...     0    0    0    0 
    ...

    每张订单一行，s0表示包含第0类体积的件数，s1表示包含第1类体积的件数，如此类推，把所有的体积类型都列出，即使包含的数量为0
    例如上面例子中订单AB6536484包含了第0类体积的件数为4件，第1类的体积的件数为1件，第2类的体积的件数为1件，第3类的体积的件数为1件
'''
def convert_data(orders,init_feature_df,order_feature_df,classify_define,prefix):
    ###把init_feature_df和order_feature_df进行合并. 
    ###因为每张订单包含某些feature只是少数，为确保数据能反映所有feature（即使为0），
    ##所以把init_feature_df(列了所有订单所有feature，只是数量为0) 和 order_feature_df（订单真正含有的feature）进行了合并
    data = pd.concat([init_feature_df,order_feature_df],axis=0)
    ##print(data[data['ordnum']==926536484])
    data=data.groupby(['ordnum','type']).sum()##对column order和volumn_type进行分类汇总
    data=data.reset_index()##去除分类汇总后的index
    data=data.reset_index(['ordnum'])
    data=pd.DataFrame({
        'ordnum':data['ordnum'],
        'type':data['type'],
        'pckqty':data['pckqty']
    })
    ##data=data.reindex(orders)
    ##print(data.T.loc[:,0:5].head(120))
    data=data.sort_index(by=['ordnum','type'])
    ##data=data[(data['ordnum']==926517449) | (data['ordnum']==926522461)]
    ##data=data[(data['ordnum']==926519802)]
    ##print(data[(data['ordnum']==926536484)])
    dataresult=pd.DataFrame(orders)
    for i in range(len(classify_define)-1):
        data0=data[(data['type']==i)]
        dataresult=pd.merge(dataresult,data0,on='ordnum')
        dataresult=dataresult.drop(['type'],axis=1)
        dataresult.rename(columns={'pckqty':prefix+str(i)},inplace=True)
    
    
    return dataresult

'''
Description: cal_order_ctn(data)

    cal_order_ctn(data)方法用于统计每张订单散箱的拆箱结果。返回结果格式如下：
    订单编号   M箱数量   S箱数量   L箱数量
    例如，订单A    1个M箱子   0个S箱子   2个L箱子
    表示订单A的装箱方式是1个M箱+2个L箱

Arg:

    data:是原始数据。内容格式如下：每行一个箱中每个产品的数据。例如某订单用2个箱子装（M类型箱子+S类型箱子），第1个箱子装A产品2个+B产品3个，第2个箱装C产品1个
         则csv文件中用三行表示，分别是
        产品A,1号箱子编号,订单编号,M,2,单个产品A长度,单个产品A宽度,单个产品A高度
        产品B,1号箱子编号,订单编号,M,3,单个产品B长度,单个产品B宽度,单个产品B高度
        产品C,2号箱子编号,订单编号,S,1,单个产品C长度,单个产品C宽度,单个产品C高度
'''
def cal_order_ctn(data):
    
    m_data = data.copy()
    l_data = data.copy()
    s_data = data.copy()
    m_data = m_data[m_data['ctnsze']=='M']##筛选出只是M箱的数据
    l_data = l_data[l_data['ctnsze']=='L']##筛选出只是l箱的数据
    s_data = s_data[s_data['ctnsze']=='S']##筛选出只是s箱的数据
    m_data = m_data.drop_duplicates(subset=['ctnnum'],keep='first',inplace=False) ##因为每个箱子可能会有多笔数据（例如一个箱子装了不同的产品），这步是去掉重复的箱子数据
    l_data = l_data.drop_duplicates(subset=['ctnnum'],keep='first',inplace=False) 
    s_data = s_data.drop_duplicates(subset=['ctnnum'],keep='first',inplace=False) 
    m_groupby = m_data.groupby(['ordnum'])##按订单号汇总
    l_groupby = l_data.groupby(['ordnum'])
    s_groupby = s_data.groupby(['ordnum'])
    ##print(m_data)
    groupby = pd.DataFrame({
        "m":m_groupby.size().to_dict(), ##m_groupby.size()是统计每张订单包含多少个不同的M箱
        "l":l_groupby.size().to_dict(), ##l_groupby.size()是统计每张订单包含多少个不同的L箱
        "s":s_groupby.size().to_dict()  ##s_groupby.size()是统计每张订单包含多少个不同的S箱
        })##产生新的DataFram，Columns为m/l/s，每张订单会有多笔数据，效果类似SQL的case when把竖的数据变成横的数据
    groupby = groupby.fillna(0)##把NAN的数据填充0
    groupby["ordnum"] = groupby.index
    return groupby


'''
Description:combind_order_feature_and_label(order_feature,order_label)
    本方法用于组合订单的feature数据和订单的标签名数据
    
Args:

    order_feature:订单的feature，数据例子如下：
   ordnum     s0  s1  s2  s3  s4  s5  s6  s7  s8 ...   s56  s57  s58  s59  
0  AB6536482  32   0   0   1   0   0   0   0   0 ...     0    0    0    0   
1  AB6536532   6   3   4   0   0   2   0   0   0 ...     0    0    0    0   
2  AB6536544   2   0   1   0   0   0   0   0   0 ...     0    0    0    0 


    order_label:订单的label(标签)，数据例子如下：
             l    m    s     ordnum
AB6517449  0.0  1.0  0.0  AB6517449
AB6519802  0.0  1.0  0.0  AB6519802
AB6522461  0.0  1.0  0.0  AB6522461
AB6534581  1.0  0.0  0.0  AB6534581
AB6534651  0.0  0.0  1.0  AB6534651
    例如第1笔数据表示的意思是订单AB6517449包含1个m类型的箱子；

Returns:

    以订单为条件关联两个入参，返回如以下的数据：    
      ordnum  s0  s1  s2  s3  s4  s5  s6  s7  s8 ...   s56  s57  s58  s59    l    m    s  
0  926536482  32   0   0   1   0   0   0   0   0 ...     0    0    0    0    0.0  1.0  0.0  
1  926536532   6   3   4   0   0   2   0   0   0 ...     0    0    0    0    0.0  1.0  1.0   
2  926536544   2   0   1   0   0   0   0   0   0 ...     0    0    0    0    0.0  1.0  0.0   
3  926536484   4   1   1   1   0   0   0   0   0 ...     0    0    0    0    0.0  1.0  0.0   
4  926536542  32  20   0   0   0   0   0   0   0 ...     0    0    0    0    0.0  0.0  1.0   
 
'''
def combind_order_feature_and_label(order_feature,order_label):
    result = pd.merge(order_feature,order_label,left_on='ordnum',right_on='ordnum')
    #####print(result.head(5))
    return result;

def tmp_lambda_func(inst_l,inst_m,inst_s,standard_l,standard_m,standard_s,label,t): 
    r=0 
    if label==0 and inst_l==standard_l and inst_m==standard_m and inst_s==standard_s: 
        r=t 
    elif label!=0:
        r=label
    return r;


def tmp_convert_data(data,standard_l,standard_m,standard_s,t): 
    func = np.frompyfunc(lambda inst_l,inst_m,inst_s,label,t:tmp_lambda_func(inst_l,inst_m,inst_s,standard_l,standard_m,standard_s,label,t),5,1) 
    label = func(data['l'],data['m'],data['s'],data['label'],t) 
    data['label']=label 
    
    '''
    data = data.drop(['l'],axis=1)
    data = data.drop(['m'],axis=1)
    data = data.drop(['s'],axis=1)
    '''
    return data

'''
Description:listAllCartonazationResult(combind_result)
    本方法用于统计所有订单中使用的装箱方案。
    例如如果所有订单中某张订单用到单独的一个l箱装，则入参combind_result中会有l=1，m=0，s=0，则返回的list结果中有一笔是l=1,m=0,s=0
       如果所有订单中某张订单用到单独的一个l+两个m箱装，则入参combind_result中会有l=1，m=2，s=0，则返回的list结果中有一笔是l=1,m=2,s=0
    如此类推，列出所有订单中使用的装箱方案。
    
Args:
    combind_result:

    s0/s1/...s58/s59为订单中产品的体积特征数据，表示在订单中含有某类型体积产品的个数，例如下面数据例子中s0为32表示体积类型0类有32个。
    l0/l1/...l14/l15为订单中产品的长度特征数据，表示在订单中含有某类型长度产品的个数。
    w0/w1/...w14/w15为订单中产品的宽度特征数据，表示在订单中含有某类型宽度产品的个数。
    h0/h1/...h14/h15为订单中产品的高度特征数据，表示在订单中含有某类型高度产品的个数。
    l表示因装该订单，用到了多少个l类型的箱子，
    m表示因装该订单，用到了多少个m类型的箱子。
    s表示因装该订单，用到了多少个s类型的箱子。
    如下面数据例子中，第1例，则整个订单用了m类型的箱子1个；第2例中用了m类型和s类型的箱子各1个
    
    例子数据如下：
      ordnum  s0  s1  s2  s3  s4  s5  s6  s7  s8 ...   h9  h10  h11  h12  h13  \
0  926536482  32   0   0   1   0   0   0   0   0 ...    0    0    0    0    0   
1  926536532   6   3   4   0   0   2   0   0   0 ...    0    0    0    0    0   
2  926536544   2   0   1   0   0   0   0   0   0 ...    0    0    0    0    0   
3  926536484   4   1   1   1   0   0   0   0   0 ...    0    0    0    0    0   
4  926536542  32  20   0   0   0   0   0   0   0 ...    0    0    0    0    0   

   h14  h15    l    m    s  
0    0    0  0.0  1.0  0.0  
1    0    0  0.0  1.0  1.0

Returns:
    
'''
def listAllCartonazationResult(combind_result):
    combind_result['count']=1
    cartonzation_result_list = combind_result.groupby(['l','m','s'])
    cartonzation_result_list = cartonzation_result_list.count()['count']
    result_list=[]
    for r in cartonzation_result_list.keys():
        result_list.append(r)
    ##print(cartonzation_result)
    result_list=[(0,0,1),(0,1,0),(1,0,0)]
    ##result_list=[(0,0,1)]
    return result_list
    

def data_load():
    srcfile = "1001.csv"
    dstfile = "1001_ctn_statics"
    '''
    org_data=pd.read_csv(srcfile)
    ##print(org_data.head())
    result_volumn_df,result_len_df,result_wid_df,result_hgt_df = cal_order_features(org_data)
    print("------volumn-------")
    print(result_volumn_df.head())
    print("")
    print("------len-------")
    print(result_len_df.head())
    print("")
    print("------wid-------")
    print(result_wid_df.head())
    print("")
    print("------hgt-------")
    print(result_hgt_df.head())
    print("")
    ##print(result[926536532])
    ##print(result[926536544])
    '''
    
    ###############################################
    
    
    org_data=pd.read_csv(srcfile)
    orders = list_all_orders(org_data) ##列出所有订单(变量orders类型为Series)
    ##init_volumn_type结构如下(类型为numpy.ndarray),数据例子如下：
    ##因为我们对所有产品的体积分为60类(如体积为0~50视为第0类，体积为50~100视为第1类...体积为2950~3000视为第59类)，
    ##所以下面0~59的元素表示60类体积分类。
    ##另外60也不是不变的数字，如变量classify_volumn_define更新了，则其分类也可能变化。
    ##[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    ## 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
    ## 50 51 52 53 54 55 56 57 58 59]
    init_volumn_type = np.arange(0,len(classify_volumn_define)-1,1)
    init_feature_df = init_orders_feature(orders,init_volumn_type)##init_feature_df列出所有订单及其体积件数（每个元素初始化为0）
    ##order_feature_df列出实际订单中包含的体积分类及其件数。
    ##order_feature_df数据的例子如下：
    
    ##      ordnum  pckqty volumn_type
    ## 0  AB6517449       1          13
    ## 1  AB6519802       1           0
    ## 2  AB6519802       2           1
    ## 3  AB6519802       1           3
    ## 4  AB6522461       1           0
    ## 5  AB6522461       2           1
    ## 6  AB6522461       3           2
    ## 例如某订单AB6517449包含第13类的体积1件。
    ##
    ##
    
    order_volumn_feature_df,order_len_feature_df,order_wid_feature_df,order_hgt_feature_df = cal_order_features(org_data)
    #####print(order_volumn_feature_df.head(10))
    ##print(order_feature_df[order_feature_df['ordnum']==926517449])
    
    order_feature_volumn = convert_data(orders,init_feature_df,order_volumn_feature_df,classify_volumn_define,prefix='s')
    order_feature_len = convert_data(orders,init_feature_df,order_len_feature_df,classify_len_define,prefix='l')
    order_feature_wid = convert_data(orders,init_feature_df,order_wid_feature_df,classify_wid_define,prefix='w')
    order_feature_hgt = convert_data(orders,init_feature_df,order_hgt_feature_df,classify_hgt_define,prefix='h')
    order_feature = pd.merge(order_feature_volumn,order_feature_len,on='ordnum')
    order_feature = pd.merge(order_feature,order_feature_wid,on='ordnum')
    order_feature = pd.merge(order_feature,order_feature_hgt,on='ordnum')
    order_feature.to_csv("converted.csv")##导出csv
    #####print(order_feature_volumn.head())
    #####print(order_feature_len.head())
    #####print(order_feature_wid.head())
    #####print(order_feature_hgt.head())
    #####print(len(order_feature),len(order_feature_len),len(order_feature_wid),len(order_feature_hgt))
    ##print(result[(result['ordnum']==926536484) | (result['ordnum']==926522461)] )
    
    ###############################################
    org_data=pd.read_csv(srcfile)
    ##print(org_data.head())
    
    order_label = cal_order_ctn(org_data)
    #####print(order_label.head())
    
    ##############################################
    combind_result = combind_order_feature_and_label(order_feature,order_label)   
    ##combind_result.to_csv('C:\\Users\\ljzon\\Downloads\\dpsdata\\combind_result.csv')
    
    ##############################################
    ##testresult = combind_result[(combind_result['l']==0) & (combind_result['m']==0) & (combind_result['s']==7)]
    ##print(testresult)
    cartonzation_result_list=listAllCartonazationResult(combind_result)
    #####print(cartonzation_result_list)
    
    
    tmp_convert_result = combind_result
    tmp_convert_result['label']=0
    t=1
    for cartonzation_result in cartonzation_result_list:
        tmp_convert_result = tmp_convert_data(tmp_convert_result,cartonzation_result[0],cartonzation_result[1],cartonzation_result[2],t)
        t=t+1
    #####print(tmp_convert_result.head(20))
    ##print(tmp_convert_result[tmp_convert_result['ordnum']==926536482])
    
    
    print('len(combind_result):'+str(len(combind_result)))
    print('len(tmp_convert_result):'+str(len(tmp_convert_result)))
    train_x = tmp_convert_result.copy()
    train_y = tmp_convert_result.copy()
    train_x = train_x.loc[0:12499,'s0':'h15'].T
    train_y = train_y.loc[0:12499,['label']].T    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print('train_y____:'+str(np.unique(train_y)))
    
    test_x = tmp_convert_result.copy()
    test_y = tmp_convert_result.copy()
    test_x = test_x.loc[12500:13499,'s0':'h15'].T
    test_y = test_y.loc[12500:13499,['label']].T
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print('test_y____:'+str(np.unique(test_y)))
    
    return train_x,train_y,test_x,test_y
    
import cartonzation_Data_Load
train_x,train_y,test_x,test_y=cartonzation_Data_Load.data_load();