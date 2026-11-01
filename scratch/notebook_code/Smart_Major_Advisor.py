#==================================================# CELL#==================================================import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#==================================================# CELL#==================================================file_path = "../data/Dataset Project 404.xlsx"

df = pd.read_excel(file_path)

df.head()

#==================================================# CELL#==================================================# معرفة أبعاد مجموعة البيانات
df.shape

#==================================================# CELL#==================================================#معرفة أسماء الأعمدة
df.columns

#==================================================# CELL#==================================================#معرفة معلومات عامة عن البيانات
df.info()

#==================================================# CELL#==================================================#معرفة عدد القيم المفقودة
df.isnull().sum()

#==================================================# CELL#==================================================# حساب عدد الصفوف المكررة
df.duplicated().sum()

#==================================================# CELL#==================================================# عرض الأعمدة التي تحتوي على قيم مفقودة فقط
missing_values = df.isnull().sum()

missing_values[missing_values > 0]

#==================================================# CELL#==================================================# حذف الأعمدة غير المفيدة في تدريب النموذج
df = df.drop(columns=['Sr.No.', 'Course'])

#==================================================# CELL#==================================================# معرفة أبعاد مجموعة البيانات بعد حذف الأعمدة غير المفيدة
df.shape

#==================================================# CELL#==================================================# عرض القيم الفريدة في العمود P6
df['P6'].unique()

#==================================================# CELL#==================================================# عرض القيم الفريدة في العمود P8
df['P8'].unique()

#==================================================# CELL#==================================================# تعويض القيم المفقودة في P6 بأكثر قيمة متكررة
df['P6'] = df['P6'].fillna(df['P6'].mode()[0])

# تعويض القيم المفقودة في P8 بأكثر قيمة متكررة
df['P8'] = df['P8'].fillna(df['P8'].mode()[0])

#==================================================# CELL#==================================================# التحقق من عدم وجود قيم مفقودة
df.isnull().sum()

#==================================================# CELL#==================================================# معرفة عدد الوظائف المختلفة في مجموعة البيانات
df['Job profession'].nunique()

#==================================================# CELL#==================================================# عرض عدد السجلات الموجودة لكل وظيفة
df['Job profession'].value_counts()

#==================================================# CELL#==================================================# عرض عدد السجلات لأفضل 15 وظيفة
job_counts = df['Job profession'].value_counts().head(15)

plt.figure(figsize=(12, 6))

plt.bar(
    job_counts.index,
    job_counts.values
)

plt.title('Number of Records for Top 15 Job Professions')
plt.xlabel('Job Profession')
plt.ylabel('Number of Records')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

#==================================================# CELL#==================================================# عرض الإحصائيات الوصفية لدرجات المهارات
skill_columns = [
    'Linguistic',
    'Musical',
    'Bodily',
    'Logical - Mathematical',
    'Spatial-Visualization',
    'Interpersonal',
    'Intrapersonal',
    'Naturalist'
]

df[skill_columns].describe()

#==================================================# CELL#==================================================# حساب متوسط كل مهارة
skill_means = df[skill_columns].mean()

# رسم متوسط درجات المهارات
plt.figure(figsize=(10, 6))

plt.bar(
    skill_means.index,
    skill_means.values
)

plt.title('Average Score for Each Skill')
plt.xlabel('Skills')
plt.ylabel('Average Score')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

#==================================================# CELL#==================================================# حساب الارتباط بين المهارات
correlation = df[skill_columns].corr()

correlation

#==================================================# CELL#==================================================import seaborn as sns
import matplotlib.pyplot as plt

# رسم خريطة الارتباط بين المهارات
plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Between Skills')
plt.show()

#==================================================# CELL#==================================================# عرض توزيع درجات مهارة Logical - Mathematical

plt.figure(figsize=(8, 5))

plt.hist(
    df['Logical - Mathematical'],
    bins=10,
    edgecolor='black'
)

plt.title('Distribution of Logical - Mathematical Skill')
plt.xlabel('Score')
plt.ylabel('Number of Students')

plt.show()

#==================================================# CELL#==================================================# مقارنة توزيع درجات جميع المهارات

plt.figure(figsize=(12, 6))

df[skill_columns].boxplot()

plt.title('Distribution of Scores Across Skills')
plt.xlabel('Skills')
plt.ylabel('Score')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

#==================================================# CELL#==================================================# حساب عدد القيم الشاذة في كل مهارة

for column in skill_columns:
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]
    
    print(column, ":", len(outliers), "outliers")

#==================================================# CELL#==================================================df.columns.tolist()

#==================================================# CELL#==================================================# تحديد المدخلات والهدف

X = df[skill_columns]

y = df['Job profession']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

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

#==================================================# CELL#==================================================from sklearn.ensemble import RandomForestClassifier

# إنشاء نموذج Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# تدريب النموذج
model.fit(X_train, y_train)

print("Model trained successfully!")

#==================================================# CELL#==================================================# اختبار النموذج على بيانات الاختبار
y_pred = model.predict(X_test)

print("Predictions completed successfully!")

#==================================================# CELL#==================================================from sklearn.metrics import accuracy_score

# حساب دقة النموذج
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

#==================================================# CELL#==================================================from sklearn.metrics import classification_report

# عرض تقرير شامل لتقييم أداء النموذج
print(classification_report(y_test, y_pred))

#==================================================# CELL#==================================================from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# إنشاء مصفوفة الالتباس
cm = confusion_matrix(y_test, y_pred)

# عرض مصفوفة الالتباس
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

plt.figure(figsize=(20, 20))

disp.plot(
    xticks_rotation=90,
    values_format='d'
)

plt.title("Confusion Matrix")
plt.show()

#==================================================# CELL#==================================================import pandas as pd

new_student = pd.DataFrame(
    [[15, 10, 12, 19, 18, 14, 13, 11]],
    columns=skill_columns
)

# التنبؤ بالمهنة المناسبة

prediction = model.predict(new_student)

print("Predicted Job Profession:", prediction[0])

#==================================================# CELL#==================================================# عرض أفضل 3 مهن محتملة مع نسب الاحتمالية

import numpy as np
import pandas as pd

# الحصول على احتمالية كل مهنة
probabilities = model.predict_proba(new_student)[0]

# الحصول على أسماء المهن
professions = model.classes_

# إنشاء DataFrame للنتائج
results = pd.DataFrame({
    'Job Profession': professions,
    'Probability': probabilities
})

# ترتيب النتائج من الأعلى إلى الأقل
results = results.sort_values(
    by='Probability',
    ascending=False
)

# عرض أفضل 3 مهن
top_3 = results.head(3)

# تحويل الاحتمالية إلى نسبة مئوية
top_3 = top_3.copy()
top_3['Probability (%)'] = top_3['Probability'] * 100

print(top_3[['Job Profession', 'Probability (%)']])

#==================================================# CELL#==================================================import joblib

# حفظ النموذج المدرّب
joblib.dump(model, "../models/smart_major_model.pkl")

print("Model saved successfully!")

