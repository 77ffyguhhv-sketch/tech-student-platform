#==================================================# CELL#==================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("Libraries imported successfully!")


#==================================================# CELL#==================================================# تحديد مسارات البيانات بشكل متوافق مع تشغيل الـNotebook من مجلد notebooks

from pathlib import Path

BASE_DIR = Path.cwd().resolve()

possible_data_dirs = [
    BASE_DIR.parent / "data",
    BASE_DIR / "data",
]

DATA_DIR = next((p for p in possible_data_dirs if p.exists()), BASE_DIR.parent / "data")

courses_path = DATA_DIR / "Courses_and_Learning_Material.csv"
roadmap_path = DATA_DIR / "Learning_Pathway_Index.csv"

print("Data directory:", DATA_DIR)
print("Courses file exists:", courses_path.exists())
print("Roadmap file exists:", roadmap_path.exists())

#==================================================# CELL#==================================================courses_df = pd.read_csv(courses_path)
roadmap_df = pd.read_csv(roadmap_path)

print("Courses Dataset Shape:", courses_df.shape)
print("Roadmap Dataset Shape:", roadmap_df.shape)
print("\nData loaded successfully!")

#==================================================# CELL#==================================================
print("Courses Dataset Preview:")
display(courses_df.head())

print("\nRoadmap Dataset Preview:")
display(roadmap_df.head())


#==================================================# CELL#==================================================
print("Courses Dataset Shape:", courses_df.shape)
print("Roadmap Dataset Shape:", roadmap_df.shape)


#==================================================# CELL#==================================================
print("Courses Dataset Columns:")
print(courses_df.columns.tolist())

print("\nRoadmap Dataset Columns:")
print(roadmap_df.columns.tolist())


#==================================================# CELL#==================================================
print("Courses Dataset Information:")
courses_df.info()

print("\n" + "=" * 60)

print("\nRoadmap Dataset Information:")
roadmap_df.info()


#==================================================# CELL#==================================================
print("Missing Values in Courses Dataset:")
display(courses_df.isnull().sum().to_frame("Missing Values"))

print("\nMissing Values in Roadmap Dataset:")
display(roadmap_df.isnull().sum().to_frame("Missing Values"))


#==================================================# CELL#==================================================
print("Duplicate Rows in Courses Dataset:", courses_df.duplicated().sum())
print("Duplicate Rows in Roadmap Dataset:", roadmap_df.duplicated().sum())


#==================================================# CELL#==================================================
courses_df = courses_df.drop_duplicates().copy()
roadmap_df = roadmap_df.drop_duplicates().copy()

print("Courses Dataset Shape After Removing Duplicates:", courses_df.shape)
print("Roadmap Dataset Shape After Removing Duplicates:", roadmap_df.shape)


#==================================================# CELL#==================================================# تنظيف بيانات الدورات والمسارات بطريقة آمنة

def clean_dataframe(df):
    df = df.copy()

    # تنظيف الأعمدة النصية
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()

    # تنظيف الأعمدة الرقمية فقط
    for column in df.select_dtypes(include=[np.number]).columns:
        median_value = df[column].median()
        if pd.isna(median_value):
            median_value = 0
        df[column] = df[column].fillna(median_value)

    return df

courses_df = clean_dataframe(courses_df)
roadmap_df = clean_dataframe(roadmap_df)

print("Data cleaning completed successfully!")
print("Remaining missing values in Courses:", courses_df.isnull().sum().sum())
print("Remaining missing values in Roadmap:", roadmap_df.isnull().sum().sum())

#==================================================# CELL#==================================================
print("Remaining Missing Values in Courses Dataset:", courses_df.isnull().sum().sum())
print("Remaining Missing Values in Roadmap Dataset:", roadmap_df.isnull().sum().sum())

print("Duplicate Rows in Courses Dataset:", courses_df.duplicated().sum())
print("Duplicate Rows in Roadmap Dataset:", roadmap_df.duplicated().sum())


#==================================================# CELL#==================================================
print("Number of Courses Records:", len(courses_df))
print("Number of Roadmap Records:", len(roadmap_df))

print("\nNumber of Courses Columns:", len(courses_df.columns))
print("Number of Roadmap Columns:", len(roadmap_df.columns))


#==================================================# CELL#==================================================
print("Unique Values in Courses Dataset Text Columns:\n")

for column in courses_df.select_dtypes(include="object").columns:
    print(f"{column}: {courses_df[column].nunique()}")


#==================================================# CELL#==================================================
print("Unique Values in Roadmap Dataset Text Columns:\n")

for column in roadmap_df.select_dtypes(include="object").columns:
    print(f"{column}: {roadmap_df[column].nunique()}")


#==================================================# CELL#==================================================
# اختيار أول عمود نصي مناسب للرسم

text_columns = courses_df.select_dtypes(include="object").columns.tolist()

if text_columns:
    selected_column = text_columns[0]
    top_values = courses_df[selected_column].value_counts().head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(top_values.index.astype(str), top_values.values)

    plt.title(f"Top 10 Values in {selected_column}")
    plt.xlabel(selected_column)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No text columns available for visualization.")


#==================================================# CELL#==================================================# الإحصائيات الوصفية للأعمدة المتاحة
courses_df.describe(include="all")

#==================================================# CELL#==================================================
prepared_courses = courses_df.copy()

course_text_columns = prepared_courses.select_dtypes(include="object").columns.tolist()

print("Course Text Columns:")
print(course_text_columns)


#==================================================# CELL#==================================================
prepared_courses["combined_features"] = (
    prepared_courses[course_text_columns]
    .astype(str)
    .agg(" ".join, axis=1)
)

prepared_courses[["combined_features"]].head()


#==================================================# CELL#==================================================
prepared_roadmap = roadmap_df.copy()

roadmap_text_columns = prepared_roadmap.select_dtypes(include="object").columns.tolist()

print("Roadmap Text Columns:")
print(roadmap_text_columns)


#==================================================# CELL#==================================================
prepared_roadmap["combined_features"] = (
    prepared_roadmap[roadmap_text_columns]
    .astype(str)
    .agg(" ".join, axis=1)
)

prepared_roadmap[["combined_features"]].head()


#==================================================# CELL#==================================================
course_vectorizer = TfidfVectorizer(stop_words="english")

course_vectors = course_vectorizer.fit_transform(
    prepared_courses["combined_features"]
)

print("Course recommendation system prepared successfully!")
print("Course Vectors Shape:", course_vectors.shape)


#==================================================# CELL#==================================================
roadmap_vectorizer = TfidfVectorizer(stop_words="english")

roadmap_vectors = roadmap_vectorizer.fit_transform(
    prepared_roadmap["combined_features"]
)

print("Learning pathway recommendation system prepared successfully!")
print("Roadmap Vectors Shape:", roadmap_vectors.shape)


#==================================================# CELL#==================================================
# عرض عدد الكلمات التي تعلمها كل نظام

print("Number of Course Features:", len(course_vectorizer.get_feature_names_out()))
print("Number of Roadmap Features:", len(roadmap_vectorizer.get_feature_names_out()))


#==================================================# CELL#==================================================
def recommend_courses(user_interest, top_n=10):

    user_vector = course_vectorizer.transform([user_interest])

    similarity_scores = cosine_similarity(
        user_vector,
        course_vectors
    ).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    recommendations = prepared_courses.iloc[top_indices].copy()

    recommendations["Similarity Score"] = similarity_scores[top_indices]

    return recommendations


#==================================================# CELL#==================================================
def recommend_learning_path(user_interest, top_n=5):

    user_vector = roadmap_vectorizer.transform([user_interest])

    similarity_scores = cosine_similarity(
        user_vector,
        roadmap_vectors
    ).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    recommendations = prepared_roadmap.iloc[top_indices].copy()

    recommendations["Similarity Score"] = similarity_scores[top_indices]

    return recommendations


#==================================================# CELL#==================================================
user_interest = "Machine Learning"

recommended_paths = recommend_learning_path(
    user_interest,
    top_n=5
)

print("Recommended Learning Paths:")
display(
    recommended_paths.drop(
        columns=["combined_features"],
        errors="ignore"
    )
)


#==================================================# CELL#==================================================
recommended_courses = recommend_courses(
    user_interest,
    top_n=10
)

print("Recommended Courses and Learning Materials:")
display(
    recommended_courses.drop(
        columns=["combined_features"],
        errors="ignore"
    )
)


#==================================================# CELL#==================================================
def generate_learning_roadmap(user_interest, path_top_n=5, course_top_n=10):

    print("=" * 70)
    print("SMART LEARNING ROADMAP GENERATOR")
    print("=" * 70)

    print(f"\nLearning Goal: {user_interest}")

    print("\n1. RECOMMENDED LEARNING PATHS")
    print("-" * 70)

    paths = recommend_learning_path(
        user_interest,
        top_n=path_top_n
    )

    display(
        paths.drop(
            columns=["combined_features"],
            errors="ignore"
        )
    )

    print("\n2. RECOMMENDED COURSES AND LEARNING MATERIALS")
    print("-" * 70)

    courses = recommend_courses(
        user_interest,
        top_n=course_top_n
    )

    display(
        courses.drop(
            columns=["combined_features"],
            errors="ignore"
        )
    )

    return paths, courses


#==================================================# CELL#==================================================
# تجربة إنشاء خارطة تعلم جديدة

new_learning_goal = "Python Programming"

paths, courses = generate_learning_roadmap(
    new_learning_goal
)


#==================================================# CELL#==================================================
new_learning_goal = "Data Science"

paths, courses = generate_learning_roadmap(
    new_learning_goal
)


#==================================================# CELL#==================================================
os.makedirs("../models", exist_ok=True)

print("Models directory ready!")


#==================================================# CELL#==================================================
joblib.dump(
    {
        "vectorizer": course_vectorizer,
        "vectors": course_vectors,
        "data": prepared_courses
    },
    "../models/learning_course_recommender.pkl"
)

print("Course recommendation system saved successfully!")


#==================================================# CELL#==================================================
joblib.dump(
    {
        "vectorizer": roadmap_vectorizer,
        "vectors": roadmap_vectors,
        "data": prepared_roadmap
    },
    "../models/learning_path_recommender.pkl"
)

print("Learning pathway recommendation system saved successfully!")


#==================================================# CELL#==================================================
saved_course_system = joblib.load(
    "../models/learning_course_recommender.pkl"
)

saved_roadmap_system = joblib.load(
    "../models/learning_path_recommender.pkl"
)

print("Saved systems loaded successfully!")
print("Learning_Roadmap_Generator.ipynb completed successfully!")


