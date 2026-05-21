import streamlit as st

st.title("🤖 智能金融问答助手")

user_input = st.text_input("请输入你要查询的股票：")

if st.button("开始分析"):
    st.write(f"正在为您分析【{user_input}】的最新研报数据...")