import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import streamlit as st

# Tiêu đề ứng dụng Streamlit
st.title('Tỷ lệ tốt nghiệp')
st.header('Dự đoán tỷ lệ tốt nghiệp dựa trên dữ liệu học sinh')
st.markdown('[GitHub của tôi](https://github.com/trung713/trumane.git)')

# Tải tệp CSV
st.divider()
uploaded_file = st.file_uploader("Tải lên tệp CSV", type="csv")

if uploaded_file is not None:
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv('graduation_rate.csv')

    # Hiển thị một vài dòng dữ liệu đầu tiên
    st.write("Dữ liệu mẫu:")
    st.write(df.head())

    # Thêm cột mục tiêu 'graduated' dựa trên 'years to graduate'
    df['graduated'] = df['years to graduate'].apply(lambda x: 1 if x <= 4 else 0)

    # Chọn các cột đặc trưng để dự đoán
    features = ['ACT composite score', 'SAT total score', 'parental level of education',
                'parental income', 'high school gpa', 'college gpa']
    X = df[features]
    y = df['graduated']

    # Tiền xử lý dữ liệu
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), ['ACT composite score', 'SAT total score', 'parental income', 'high school gpa', 'college gpa']),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['parental level of education'])
        ])

    # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuẩn bị dữ liệu cho các mô hình
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Huấn luyện mô hình Random Forest với Grid Search
    rf = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_train_preprocessed, y_train)
    best_rf = grid_rf.best_estimator_

    # Huấn luyện mô hình SVM với Grid Search
    svm = SVC(random_state=42)
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
    grid_svm.fit(X_train_preprocessed, y_train)
    best_svm = grid_svm.best_estimator_

    # Nhập thông tin người dùng
    st.divider()
    st.subheader('Nhập thông tin học sinh để dự đoán:')
    act_score = st.number_input('Điểm ACT composite score', min_value=0)
    sat_score = st.number_input('Điểm SAT total score', min_value=0)
    parental_income = st.number_input('Thu nhập của phụ huynh', min_value=0)
    high_school_gpa = st.number_input('GPA cấp 3', min_value=0.0, format="%.2f")
    college_gpa = st.number_input('GPA đại học', min_value=0.0, format="%.2f")
    parental_level_of_education = st.selectbox('Cấp học của phụ huynh', ['None', 'High School', 'Associate’s Degree', 'Bachelor’s Degree', 'Master’s Degree'])

    # Tạo DataFrame cho thông tin nhập vào
    input_data = pd.DataFrame({
        'ACT composite score': [act_score],
        'SAT total score': [sat_score],
        'parental level of education': [parental_level_of_education],
        'parental income': [parental_income],
        'high school gpa': [high_school_gpa],
        'college gpa': [college_gpa]
    })

    # Chuẩn bị dữ liệu cho dự đoán
    input_data_preprocessed = preprocessor.transform(input_data)

    if st.button('Dự đoán'):
        # Dự đoán bằng các mô hình
        rf_prediction = best_rf.predict(input_data_preprocessed)[0]
        svm_prediction = best_svm.predict(input_data_preprocessed)[0]

        st.write(f"Dự đoán của Random Forest: {'Tốt nghiệp' if rf_prediction == 1 else 'Không tốt nghiệp'}")
        st.write(f"Dự đoán của SVM: {'Tốt nghiệp' if svm_prediction == 1 else 'Không tốt nghiệp'}")

        # Đánh giá mô hình
        y_pred_rf = best_rf.predict(X_test_preprocessed)
        y_pred_svm = best_svm.predict(X_test_preprocessed)

        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)

        st.write(f'Accuracy của Random Forest: {accuracy_rf}')
        st.write(f'Accuracy của SVM: {accuracy_svm}')

        st.write("\nBáo cáo phân loại của Random Forest:")
        st.text(classification_report(y_test, y_pred_rf))

        st.write("\nBáo cáo phân loại của SVM:")
        st.text(classification_report(y_test, y_pred_svm))

        # Vẽ biểu đồ kết quả dự đoán
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Biểu đồ Random Forest
        sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_title('Ma trận nhầm lẫn Random Forest')
        ax[0].set_xlabel('Dự đoán')
        ax[0].set_ylabel('Thực tế')

        # Biểu đồ SVM
        sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues', ax=ax[1])
        ax[1].set_title('Ma trận nhầm lẫn SVM')
        ax[1].set_xlabel('Dự đoán')
        ax[1].set_ylabel('Thực tế')

        st.pyplot(fig)
else:
    st.write('graduation_rate.csv')

