# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:53:53 2024

@author: lenovo
"""

#加载包
import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os
import shap  # 导入 SHAP 库，用于解释模型



# 加载模型
#获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'rf.pkl')   
with open(model_path, 'rb') as file:   
    model = pickle.load(file)  # 使用 pickle 加载模型文件
    
    
# Streamlit 应用开始  
st.header("Peg-IFN Therapy Efficacy Predictor")  
  
# 用户输入  
Sex = st.selectbox("Select Sex", ("Male", "Female"))  
NAFLD = st.selectbox("Select NAFLD Status", ("No", "Yes"))  
HBsAg = st.number_input("Enter HBsAg")  
HBsAg_decline_at_week12 = st.number_input("Enter HBsAg_decline_at_week12")  

  
# 将用户输入转换为模型可接受的格式  
sex_num = 0 if Sex == "Male" else 1  
nafld_num = 0 if NAFLD == "No" else 1  
input_data= pd.DataFrame([[sex_num, nafld_num,HBsAg, HBsAg_decline_at_week12]],  
                     columns=["Sex","NAFLD","HBsAg","HBsAg_decline_at_week12"])  
  
# 如果用户点击了提交按钮  
if st.button("Submit"):  
    # 获取预测结果  
    prediction = model.predict_proba(input_data)[0]  
    # 输出预测结果  
    st.write(f"**Prediction Probabilities:** {prediction}")    

## 计算 SHAP 值    
explainer = shap.Explainer(model)  # 或者使用 shap.TreeExplainer(model) 来计算树模型的 SHAP 值    
shap_values = explainer(input_data)


# 提取单个样本的 SHAP 值和期望值    
sample_shap_values = shap_values[0]  # 提取第一个样本的 SHAP 值    
expected_value = explainer.expected_value[0]  # 获取对应输出的期望值


# 创建 Explanation 对象   
explanation = shap.Explanation(        
     values=sample_shap_values[:, 0],  # 选择特定输出的 SHAP 值        
     base_values=expected_value,        
     data=input_data.iloc[0].values,        
     feature_names=input_data.columns.tolist()   
     )

 
# 保存为 HTML 文件   
shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))
    
# 在 Streamlit 中显示 HTML    
st.subheader("shap_picture")
with open("shap_force_plot.html", "r", encoding="utf-8") as f:
    html_content = f.read()
    st.components.v1.html(html_content, height=600) 

