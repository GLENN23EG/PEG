import pandas as pd  
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

file_path = 'C:/Users/lenovo/Desktop/干扰素机器算法/机器学习'  
csv_path = os.path.join(file_path, 'data.csv')  # 导入os模块  
df = pd.read_csv(csv_path)



# 特征变量提取及转换
x=df[["Sex","NAFLD","HBsAg","HBsAg_decline_at_week12"]]

# 分类变量提取 
y=df["Result"]
  
# 初始化随机森林分类器  
# 设置参数：n_estimators, min_impurity_decrease, max_depth, criterion  
rf=RandomForestClassifier(n_estimators=90,   
                            min_impurity_decrease=0.0,   
                            max_depth=5,   
                            criterion='entropy')  
  
# 训练模型  
rf=rf.fit(x, y)  
  
# 保存模型为.pkl文件（实际上是joblib格式，但习惯上称为.pkl）
with open("rf.pkl",'wb') as file:
    pickle.dump(rf,file)

  

