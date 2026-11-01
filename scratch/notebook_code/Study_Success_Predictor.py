#==================================================# CELL#==================================================import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#==================================================# CELL#==================================================# تحميل مجموعة البيانات

file_path = "../data/student_data_extended.csv"

df = pd.read_csv(file_path)

df.head()

#==================================================# CELL#==================================================# معرفة أبعاد مجموعة البيانات
df.shape

#==================================================# CELL#==================================================# معرفة أسماء الأعمدة
df.columns

#==================================================# CELL#==================================================# معرفة معلومات عامة عن البيانات
df.info()

#==================================================# CELL#==================================================# معرفة عدد القيم المفقودة
df.isnull().sum()

#==================================================# CELL#==================================================# حساب عدد الصفوف المكررة
df.duplicated().sum()

#==================================================# CELL#==================================================# حذف الأعمدة غير المفيدة في تدريب النموذج

columns_to_drop = ['StudentID', 'Name']

existing_columns_to_drop = [
    column for column in columns_to_drop
    if column in df.columns
]

df = df.drop(columns=existing_columns_to_drop)

df.head()

#==================================================# CELL#==================================================# معرفة أبعاد مجموعة البيانات بعد حذف الأعمدة غير المفيدة
df.shape

#==================================================# CELL#==================================================# عرض الأعمدة التي تحتوي على قيم مفقودة فقط

missing_values = df.isnull().sum()

missing_values[missing_values > 0]

#==================================================# CELL#==================================================# معالجة القيم المفقودة

for column in df.columns:

    if df[column].isnull().sum() > 0:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

print("Missing values handled successfully!")

#==================================================# CELL#==================================================# التحقق من عدم وجود قيم مفقودة
df.isnull().sum()

#==================================================# CELL#==================================================# عرض الإحصائيات الوصفية للأعمدة الرقمية
df.describe(include='all')

#==================================================# CELL#==================================================# معرفة عدد الطلاب في كل تخصص

if 'Major' in df.columns:
    df['Major'].value_counts()

#==================================================# CELL#==================================================# رسم عدد الطلاب في كل تخصص

if 'Major' in df.columns:

    major_counts = df['Major'].value_counts()

    plt.figure(figsize=(10, 6))

    plt.bar(
        major_counts.index,
        major_counts.values
    )

    plt.title('Number of Students by Major')
    plt.xlabel('Major')
    plt.ylabel('Number of Students')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

#==================================================# CELL#==================================================# عرض توزيع GPA

plt.figure(figsize=(8, 5))

plt.hist(
    df['GPA'],
    bins=20,
    edgecolor='black'
)

plt.title('Distribution of GPA')
plt.xlabel('GPA')
plt.ylabel('Number of Students')

plt.tight_layout()
plt.show()

#==================================================# CELL#==================================================# عرض الإحصائيات الوصفية لمؤشر النجاح الدراسي GPA

df['GPA'].describe()

#==================================================# CELL#==================================================# مقارنة متوسط GPA بين التخصصات

if 'Major' in df.columns:

    major_gpa = (
        df.groupby('Major')['GPA']
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        major_gpa.index,
        major_gpa.values
    )

    plt.title('Average GPA by Major')
    plt.xlabel('Major')
    plt.ylabel('Average GPA')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

#==================================================# CELL#==================================================# مقارنة متوسط GPA حسب السنة الدراسية

if 'YearOfStudy' in df.columns:

    year_gpa = (
        df.groupby('YearOfStudy')['GPA']
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        year_gpa.index,
        year_gpa.values
    )

    plt.title('Average GPA by Year of Study')
    plt.xlabel('Year of Study')
    plt.ylabel('Average GPA')

    plt.tight_layout()
    plt.show()

#==================================================# CELL#==================================================# عرض توزيع GPA حسب التخصص

if 'Major' in df.columns:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x='Major',
        y='GPA'
    )

    plt.title('GPA Distribution by Major')
    plt.xlabel('Major')
    plt.ylabel('GPA')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

#==================================================# CELL#==================================================# عرض متوسط GPA حسب الاهتمام الأكاديمي

if 'AcademicInterest' in df.columns:

    interest_gpa = (
        df.groupby('AcademicInterest')['GPA']
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        interest_gpa.index,
        interest_gpa.values
    )

    plt.title('Average GPA by Academic Interest')
    plt.xlabel('Academic Interest')
    plt.ylabel('Average GPA')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

#==================================================# CELL#==================================================# معرفة الارتباط بين الأعمدة الرقمية

numeric_columns = df.select_dtypes(include=np.number).columns

correlation = df[numeric_columns].corr()

correlation

#==================================================# CELL#==================================================# رسم خريطة الارتباط للأعمدة الرقمية

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Between Numeric Variables')

plt.tight_layout()
plt.show()

#==================================================# CELL#==================================================# حساب عدد القيم الشاذة في GPA باستخدام IQR

Q1 = df['GPA'].quantile(0.25)
Q3 = df['GPA'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['GPA'] < lower_bound) |
    (df['GPA'] > upper_bound)
]

print("GPA outliers:", len(outliers))

#==================================================# CELL#==================================================# عرض أسماء الأعمدة النهائية
df.columns.tolist()

#==================================================# CELL#==================================================# تحديد المدخلات والهدف

target_column = 'GPA'

feature_columns = [
    column for column in df.columns
    if column != target_column
]

X = df[feature_columns]

y = df[target_column]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(feature_columns)

#==================================================# CELL#==================================================from sklearn.model_selection import train_test_split

# تقسيم البيانات إلى تدريب واختبار

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

#==================================================# CELL#==================================================# تحديد الأعمدة الرقمية والفئوية

numeric_features = X.select_dtypes(include=np.number).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

#==================================================# CELL#==================================================from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# تجهيز البيانات الفئوية قبل تدريب النماذج

preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            categorical_features
        )
    ],
    remainder='passthrough'
)

print("Data preprocessor created successfully!")

#==================================================# CELL#==================================================from sklearn.ensemble import RandomForestRegressor

# إنشاء نموذج Random Forest

random_forest_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'model',
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# تدريب النموذج

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest model trained successfully!")

#==================================================# CELL#==================================================from sklearn.ensemble import GradientBoostingRegressor

# إنشاء نموذج Gradient Boosting

gradient_boosting_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'model',
            GradientBoostingRegressor(
                random_state=42
            )
        )
    ]
)

# تدريب النموذج

gradient_boosting_model.fit(
    X_train,
    y_train
)

print("Gradient Boosting model trained successfully!")

#==================================================# CELL#==================================================# إنشاء تنبؤات نموذج Random Forest

rf_predictions = random_forest_model.predict(X_test)

print("Random Forest predictions completed successfully!")

#==================================================# CELL#==================================================# إنشاء تنبؤات نموذج Gradient Boosting

gb_predictions = gradient_boosting_model.predict(X_test)

print("Gradient Boosting predictions completed successfully!")

#==================================================# CELL#==================================================from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# حساب نتائج نموذج Random Forest

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R²:", rf_r2)

#==================================================# CELL#==================================================# حساب نتائج نموذج Gradient Boosting

gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = mean_squared_error(
    y_test,
    gb_predictions
) ** 0.5

gb_r2 = r2_score(
    y_test,
    gb_predictions
)

print("Gradient Boosting MAE:", gb_mae)
print("Gradient Boosting RMSE:", gb_rmse)
print("Gradient Boosting R²:", gb_r2)

#==================================================# CELL#==================================================# مقارنة النماذج

comparison_results = pd.DataFrame({
    'Model': [
        'Random Forest',
        'Gradient Boosting'
    ],
    'MAE': [
        rf_mae,
        gb_mae
    ],
    'RMSE': [
        rf_rmse,
        gb_rmse
    ],
    'R2 Score': [
        rf_r2,
        gb_r2
    ]
})

comparison_results = comparison_results.sort_values(
    by='R2 Score',
    ascending=False
)

comparison_results

#==================================================# CELL#==================================================# اختيار النموذج الأفضل

if rf_r2 >= gb_r2:

    model = random_forest_model
    best_model_name = "Random Forest"
    y_pred = rf_predictions

else:

    model = gradient_boosting_model
    best_model_name = "Gradient Boosting"
    y_pred = gb_predictions

print("Best Model:", best_model_name)

#==================================================# CELL#==================================================# حساب المقاييس النهائية للنموذج الأفضل

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print("Best Model:", best_model_name)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

#==================================================# CELL#==================================================# مقارنة القيم الحقيقية بالقيم المتوقعة

evaluation_results = pd.DataFrame({
    'Actual GPA': y_test.values,
    'Predicted GPA': y_pred
})

evaluation_results.head(10)

#==================================================# CELL#==================================================# رسم القيم الحقيقية مقابل القيم المتوقعة

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.title('Actual vs Predicted GPA')

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    'r--'
)

plt.tight_layout()
plt.show()

#==================================================# CELL#==================================================# حساب متوسط نسبة الخطأ بالنسبة المئوية

non_zero_actual = y_test != 0

percentage_error = (
    np.abs(
        (
            y_test[non_zero_actual] -
            y_pred[non_zero_actual]
        ) /
        y_test[non_zero_actual]
    )
    * 100
)

print(
    "Average Percentage Error:",
    percentage_error.mean()
)

#==================================================# CELL#==================================================# عرض مثال من القيم الموجودة في البيانات

for column in categorical_features:

    print(
        f"{column}:",
        df[column].dropna().unique()[:10]
    )

#==================================================# CELL#==================================================# إنشاء بيانات طالب جديدة للتجربة

# تم أخذ نموذج بيانات صالح من نفس مجموعة البيانات
# ويمكن تعديل القيم لاحقًا عند بناء واجهة الويب

new_student = X.iloc[[0]].copy()

new_student

#==================================================# CELL#==================================================# التنبؤ بقيمة GPA للطالب الجديد

predicted_gpa = model.predict(
    new_student
)[0]

print(
    "Predicted GPA:",
    round(predicted_gpa, 2)
)

#==================================================# CELL#==================================================# تحويل قيمة GPA المتوقعة إلى مستوى نجاح دراسي

def get_success_level(gpa):

    if gpa >= 3.5:
        return "Excellent"

    elif gpa >= 3.0:
        return "Very Good"

    elif gpa >= 2.5:
        return "Good"

    elif gpa >= 2.0:
        return "Average"

    else:
        return "Needs Improvement"

success_level = get_success_level(
    predicted_gpa
)

print(
    "Predicted Study Success Level:",
    success_level
)

#==================================================# CELL#==================================================import joblib

# حفظ النموذج الأفضل

model_path = "../models/study_success_model.pkl"

joblib.dump(
    model,
    model_path
)

print("Model saved successfully!")

