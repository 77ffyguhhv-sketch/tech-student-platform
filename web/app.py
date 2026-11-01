import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# Get the base directory (parent of this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Dictionary to hold the loaded models and datasets
models = {}
datasets = {}

# 1. Load Smart Major Advisor Model
try:
    major_model_path = os.path.join(BASE_DIR, 'models', 'smart_major_model.pkl')
    models['smart_major'] = joblib.load(major_model_path)
    print("Loaded Smart Major model.")
except Exception as e:
    print(f"Error loading Smart Major model: {e}")
    models['smart_major'] = None

# 2. Load Study Success Model
try:
    success_model_path = os.path.join(BASE_DIR, 'models', 'study_success_model.pkl')
    models['study_success'] = joblib.load(success_model_path)
    print("Loaded Study Success model.")
except Exception as e:
    print(f"Error loading Study Success model: {e}")
    models['study_success'] = None

# 3. Load Smart Device Advisor Model and Dataset
try:
    device_model_path = os.path.join(BASE_DIR, 'models', 'smart_device_advisor_model_FIXED.pkl')
    models['device_advisor'] = joblib.load(device_model_path)
    print("Loaded Smart Device Advisor model dict.")
except Exception as e:
    print(f"Error loading Smart Device Advisor model: {e}")
    models['device_advisor'] = None

try:
    device_csv_path = os.path.join(BASE_DIR, 'data', 'mobile_recommendation_system_dataset.csv')
    df_mobile = pd.read_csv(device_csv_path)
    # Clean price and rating
    df_mobile['price_clean'] = df_mobile['price'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    df_mobile['price_clean'] = pd.to_numeric(df_mobile['price_clean'], errors='coerce')
    df_mobile['ratings_clean'] = pd.to_numeric(df_mobile['ratings'], errors='coerce')
    datasets['mobile'] = df_mobile
    print("Loaded and cleaned Mobile dataset.")
except Exception as e:
    print(f"Error loading Mobile dataset: {e}")
    datasets['mobile'] = None

# 4. Load Learning Roadmap Models
try:
    course_recommender_path = os.path.join(BASE_DIR, 'models', 'learning_course_recommender.pkl')
    models['course_recommender'] = joblib.load(course_recommender_path)
    print("Loaded Course Recommender model dict.")
except Exception as e:
    print(f"Error loading Course Recommender model: {e}")
    models['course_recommender'] = None

try:
    roadmap_recommender_path = os.path.join(BASE_DIR, 'models', 'learning_path_recommender.pkl')
    models['roadmap_recommender'] = joblib.load(roadmap_recommender_path)
    print("Loaded Roadmap Recommender model dict.")
except Exception as e:
    print(f"Error loading Roadmap Recommender model: {e}")
    models['roadmap_recommender'] = None

# HTML Template Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/major-advisor')
def major_advisor():
    return render_template('major_advisor.html')

@app.route('/study-success')
def study_success():
    return render_template('study_success.html')

@app.route('/device-advisor')
def device_advisor():
    return render_template('device_advisor.html')

@app.route('/learning-roadmap')
def learning_roadmap():
    return render_template('learning_roadmap.html')

JOB_ARABIC_MAP = {
    'Actor / Actress': 'ممثل / ممثلة',
    'Actuary': 'خبير إكتواري (مقيم مخاطر مالية)',
    'Anthropologist': 'عالم إنسان (أنثروبولوجي)',
    'Archeologist': 'عالم آثار',
    'Artist': 'فنان / رسام',
    'Astronomer': 'عالم فلك',
    'Astronomer\n': 'عالم فلك',
    'Athlete': 'رياضي محترف',
    'Audiologist': 'أخصائي سمعيات',
    'Banking': 'مختص أعمال مصرفية وبنوك',
    'Broadcaster': 'مذيع / إعلامي',
    'Business Analyst': 'محلل أعمال',
    'Business manager': 'مدير أعمال',
    'Chartered Accountant': 'محاسب قانوني معتمد',
    'Chief financial officer': 'مدير مالية تنفيذي (CFO)',
    'Company secretary': 'أمين سر شركة',
    'Computer analyst': 'محلل نظم حاسوب',
    'Computer programmer': 'مبرمج حاسوب',
    'Consultant': 'مستشار إداري / تقني',
    'Counselor': 'مستشار نفسي / توجيهي',
    'Criminologist': 'عالم أدلة وجرائم (جنائي)',
    'Dancer': 'مصمم استعراضي / راقص',
    'Database designer': 'مصمم ومطور قواعد بيانات',
    'Economist': 'خبير اقتصادي',
    'Editor': 'محرر صحفي / لغوي',
    'Engineer': 'مهندس (مختلف التخصصات)',
    'Fashion Designer': 'مصمم أزياء',
    'Financial Advisor': 'مستشار مالي',
    'Geologist': 'عالم جيولوجيا (طبقات الأرض)',
    'Graphic Designer': 'مصمم جرافيك',
    'Historian': 'مؤرخ / عالم تاريخ',
    'Interior Decorator': 'مصمم ديكور داخلي',
    'Internal auditor': 'مدقق حسابات داخلي',
    'Journalist': 'صحفي / مراسل',
    'Language Teacher': 'معلم لغات',
    'Lawyer': 'محامي / مستشار قانوني',
    'Leader': 'قائد تنفيذي / إداري',
    'Librarian': 'أخصائي مكتبات ومعلومات',
    'Logistics manager': 'مدير سلاسل إمداد ولوجستيات',
    'Manager': 'مدير عام',
    'Marine Biologist': 'عالم أحياء بحرية',
    'Marine Biologist\n': 'عالم أحياء بحرية',
    'Marketing': 'أخصائي تسويق',
    'Mathematician': 'عالم رياضيات',
    'Mechanic': 'فني ميكانيكا',
    'Medical': 'طبيب / أخصائي طبي',
    'Middle, Higher School Teacher and Professors': 'مدرس ثانوي / أستاذ جامعي',
    'Militry': 'ضابط عسكري / خدمة عسكرية',
    'Music teacher': 'معلم موسيقى',
    'Nature photographer': 'مصور طبيعة وحياة برية',
    'Para Medical (physiotherapy, occupational theropy, audio and speech language theropy, nursing)': 'أخصائي خدمات طبية مساعدة (تمريض/علاج طبيعي)',
    'Para Militry (https://en.wikipedia.org/wiki/Paramilitary_forces_of_India)': 'قوات أمنية وحماية مدنية',
    'Pharmacist': 'صيدلي',
    'Philosopher': 'فيلسوف / مفكر',
    'Physical Therapist': 'أخصائي علاج طبيعي',
    'Physician': 'طبيب بشري',
    'Physicist': 'عالم فيزياء',
    'Pilot': 'طيار مدني / عسكري',
    'Poet': 'شاعر / أديب',
    'Police Force (Spies, CBI officials, CID, Detectives)': 'ضابط شرطة / محقق جنائي',
    'Politician': 'سياسي / دبلوماسي',
    'Pre Primary Teacher (2020 NEP and Mental Health)': 'معلم رياض أطفال وطفولة مبكرة',
    'Primary Teacher': 'معلم المرحلة الابتدائية',
    'Psychologist': 'أخصائي علم نفس',
    'Receptionist': 'موظف استقبال / خدمات عملاء',
    'Recording engineer': 'مهندس تسجيل صوتي',
    'Research analyst': 'محلل بحوث ودراسات',
    'Sales Representative': 'ممثل مبيعات',
    'Social Worker': 'أخصائي خدمة اجتماعية',
    'Sound editor': 'محرر ومخرج صوتيات',
    'Stock Broker': 'وسيط مالي / تداول أسهم',
    'Technician': 'فني تقني',
    'Veterinarian': 'طبيب بيطري',
    'Writer': 'كاتب / صانع محتوى'
}

# API Endpoints
# 1. API: Smart Major Advisor Prediction
@app.route('/api/predict-major', methods=['POST'])
def predict_major():
    if not models['smart_major']:
        return jsonify({'error': 'Smart Major Advisor model is not loaded.'}), 500
    
    try:
        data = request.json
        skill_cols = [
            'Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical', 
            'Spatial-Visualization', 'Interpersonal', 'Intrapersonal', 'Naturalist'
        ]
        
        input_data = {}
        for col in skill_cols:
            val = data.get(col)
            if val is None:
                return jsonify({'error': f'Missing value for skill: {col}'}), 400
            input_data[col] = [float(val)]
            
        input_df = pd.DataFrame(input_data)
        
        # Predict primary profession
        prediction = str(models['smart_major'].predict(input_df)[0]).strip()
        prediction_ar = JOB_ARABIC_MAP.get(prediction, JOB_ARABIC_MAP.get(prediction + '\n', prediction))
        
        # Predict top 3 probabilities
        probabilities = models['smart_major'].predict_proba(input_df)[0]
        classes = models['smart_major'].classes_
        
        results = pd.DataFrame({
            'profession': classes,
            'probability': probabilities
        }).sort_values(by='probability', ascending=False)
        
        top_3 = results.head(3).copy()
        top_3['percentage'] = round(top_3['probability'] * 100, 1)
        
        predictions_list = []
        for idx, row in top_3.iterrows():
            prof_name = str(row['profession']).strip()
            predictions_list.append({
                'profession': prof_name,
                'profession_ar': JOB_ARABIC_MAP.get(prof_name, JOB_ARABIC_MAP.get(prof_name + '\n', prof_name)),
                'percentage': row['percentage']
            })
        
        # Return results
        return jsonify({
            'recommended_profession': prediction,
            'recommended_profession_ar': prediction_ar,
            'top_predictions': predictions_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2. API: Study Success Prediction
@app.route('/api/predict-success', methods=['POST'])
def predict_success():
    if not models['study_success']:
        return jsonify({'error': 'Study Success Predictor model is not loaded.'}), 500
    
    try:
        data = request.json
        feature_cols = [
            'AcademicInterest', 'ExtracurricularActivities', 'Skills', 
            'Location', 'YearOfStudy', 'Major', 'Languages', 
            'ClubMemberships', 'ResearchInterests'
        ]
        
        # Build dataframe
        input_data = {}
        for col in feature_cols:
            val = data.get(col)
            if val is None:
                return jsonify({'error': f'Missing value for field: {col}'}), 400
            input_data[col] = [str(val)]
            
        input_df = pd.DataFrame(input_data)
        
        # Predict GPA
        predicted_gpa = models['study_success'].predict(input_df)[0]
        predicted_gpa = round(float(predicted_gpa), 2)
        
        # Get success level
        def get_success_level(gpa):
            if gpa >= 3.5:
                return {"en": "Excellent", "ar": "ممتاز", "class": "success-excellent"}
            elif gpa >= 3.0:
                return {"en": "Very Good", "ar": "جيد جداً", "class": "success-vgood"}
            elif gpa >= 2.5:
                return {"en": "Good", "ar": "جيد", "class": "success-good"}
            elif gpa >= 2.0:
                return {"en": "Average", "ar": "مقبول", "class": "success-average"}
            else:
                return {"en": "Needs Improvement", "ar": "يحتاج تحسين", "class": "success-poor"}
                
        level = get_success_level(predicted_gpa)
        
        return jsonify({
            'predicted_gpa': predicted_gpa,
            'success_level': level['en'],
            'success_level_ar': level['ar'],
            'badge_class': level['class']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 3. API: Smart Device Price Predictor (Custom Specs)
@app.route('/api/predict-device', methods=['POST'])
def predict_device():
    if not models['device_advisor']:
        return jsonify({'error': 'Smart Device Advisor model is not loaded.'}), 500
    
    try:
        data = request.json
        text_features = data.get('text_features', '').strip()
        rating = data.get('rating')
        
        if not text_features or rating is None:
            return jsonify({'error': 'Please provide text description and rating.'}), 400
            
        input_df = pd.DataFrame({
            'text_features': [text_features],
            'rating': [float(rating)]
        })
        
        model_obj = models['device_advisor']['model']
        predicted_price = model_obj.predict(input_df)[0]
        
        return jsonify({
            'predicted_price': round(float(predicted_price), 2),
            'currency': 'SAR (ر.س)'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import re

def translate_device_text(text):
    if not text or not isinstance(text, str):
        return ''
    mappings = [
        (r'\bAPPLE\b', 'أبل'), (r'\biPhone\b', 'آيفون'), (r'\bSAMSUNG\b', 'سامسونج'),
        (r'\bGalaxy\b', 'جالاكسي'), (r'\bREDMI\b', 'ريدمي'), (r'\bXIAOMI\b', 'شاومي'),
        (r'\bOPPO\b', 'أوبو'), (r'\bVIVO\b', 'فيفو'), (r'\bREALME\b', 'ريلمي'),
        (r'\bONEPLUS\b', 'وان بلس'), (r'\bPOCO\b', 'بوكو'), (r'\bNOKIA\b', 'نوكيا'),
        (r'\bINFINIX\b', 'إنفينيكس'), (r'\bTECNO\b', 'تكنو'), (r'\bMOTOROLA\b', 'موتورولا'),
        (r'\bMOTO\b', 'موتورولا'), (r'\bHONOR\b', 'هونر'), (r'\bHUAWEI\b', 'هواوي'),
        (r'\bGoogle\b', 'جوجل'), (r'\bPixel\b', 'بيكسل'), (r'\bPro Max\b', 'برو ماكس'),
        (r'\bPro\b', 'برو'), (r'\bMax\b', 'ماكس'), (r'\bPlus\b', 'بلس'),
        (r'\bUltra\b', 'ألترا'), (r'\bLite\b', 'لايت'), (r'\bMini\b', 'ميني'),
        (r'\bNote\b', 'نوت'), (r'\bPower\b', 'باور'), (r'\bPrime\b', 'برايم'),
        (r'\bGB\b', 'جيجابايت'), (r'\bTB\b', 'تيرابايت'), (r'\bRAM\b', 'رام'),
        (r'\bROM\b', 'تخزين'), (r'\bSnapdragon\b', 'سناب دراجون'), (r'\bBionic\b', 'بايونيك'),
        (r'\bAndroid\b', 'أندرويد'), (r'\bSuper AMOLED\b', 'شاشة سوبر أموليد'),
        (r'\bAMOLED\b', 'شاشة أموليد'), (r'\bOLED\b', 'شاشة أوليد'), (r'\bLCD\b', 'شاشة إل سي دي'),
        (r'\bDisplay\b', 'شاشة'), (r'\bCamera\b', 'كاميرا'), (r'\bBattery\b', 'بطارية'),
        (r'\bmAh\b', 'مللي أمبير'), (r'\bMP\b', 'ميجابكسل'), (r'\bDual SIM\b', 'شريحتين اتصالات'),
        (r'\b5G\b', '5G (الجيل الخامس)'), (r'\b4G\b', '4G (الجيل الرابع)'),
        (r'\bOnyx Black\b', 'أسود أونيكس'), (r'\bPhantom Black\b', 'أسود فانتوم'),
        (r'\bMidnight Black\b', 'أسود منتصف الليل'), (r'\bGraphite Black\b', 'أسود جرافيت'),
        (r'\bBlack\b', 'أسود'), (r'\bPearl White\b', 'أبيض لؤلؤي'),
        (r'\bStarburst White\b', 'أبيض ناصع'), (r'\bWhite\b', 'أبيض'),
        (r'\bBlue Void\b', 'أزرق فاقع'), (r'\bSierra Blue\b', 'أزرق سييرا'),
        (r'\bPacific Blue\b', 'أزرق باسيفيك'), (r'\bSky Blue\b', 'أزرق سماوي'),
        (r'\bBlue\b', 'أزرق'), (r'\bAurora Green\b', 'أخضر أورورا'),
        (r'\bMidnight Green\b', 'أخضر منتصف الليل'), (r'\bGreen\b', 'أخضر'),
        (r'\bRed\b', 'أحمر'), (r'\bGold\b', 'ذهبي'), (r'\bSilver\b', 'فضي'),
        (r'\bGraphite\b', 'جرافيت'), (r'\bGray\b', 'رمادي'), (r'\bGrey\b', 'رمادي'),
        (r'\bPurple\b', 'بنفسجي'), (r'\bYellow\b', 'أصفر'), (r'\bPink\b', 'وردي'),
        (r'\bOrange\b', 'برتقالي')
    ]
    res = text
    for pattern, replacement in mappings:
        res = re.sub(pattern, replacement, res, flags=re.IGNORECASE)
    return res

# 4. API: Smart Device Advisor Recommendation
@app.route('/api/recommend-devices', methods=['POST'])
def recommend_devices_api():
    if datasets['mobile'] is None:
        return jsonify({'error': 'Device database is not loaded.'}), 500
        
    try:
        data = request.json
        budget = float(data.get('budget', 50000))
        min_rating = float(data.get('min_rating', 0.0))
        keyword = data.get('keyword', '').strip()
        
        df_m = datasets['mobile'].copy()
        
        # Filter by budget
        df_m = df_m[df_m['price_clean'] <= budget]
        
        # Filter by rating
        df_m = df_m[df_m['ratings_clean'] >= min_rating]
        
        # Filter by keyword if provided
        if keyword:
            df_m = df_m[
                df_m['name'].str.contains(keyword, case=False, na=False) |
                df_m['corpus'].str.contains(keyword, case=False, na=False)
            ]
            
        # Sort by ratings (desc) and price (asc)
        df_m = df_m.sort_values(by=['ratings_clean', 'price_clean'], ascending=[False, True])
        
        # Get top 5
        top_devices = df_m.head(5).copy()
        
        # Handle empty results
        if top_devices.empty:
            return jsonify({'devices': []})
            
        # Convert to dictionary list
        devices_list = []
        for idx, row in top_devices.iterrows():
            name_en = str(row['name'])
            specs_en = str(row.get('corpus', ''))
            devices_list.append({
                'name': name_en,
                'name_ar': translate_device_text(name_en),
                'price': float(row['price_clean']),
                'rating': float(row['ratings_clean']),
                'imgURL': row.get('imgURL', ''),
                'specs': specs_en,
                'specs_ar': translate_device_text(specs_en)
            })
            
        return jsonify({'devices': devices_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 5. API: Learning Roadmap Generator
@app.route('/api/generate-roadmap', methods=['POST'])
def generate_roadmap():
    if not models['course_recommender'] or not models['roadmap_recommender']:
        return jsonify({'error': 'Learning Recommendation System models are not loaded.'}), 500
        
    try:
        data = request.json
        user_interest = data.get('interest', '').strip()
        path_top_n = int(data.get('path_count', 5))
        course_top_n = int(data.get('course_count', 10))
        
        if not user_interest:
            return jsonify({'error': 'Please enter a learning interest.'}), 400
            
        # 1. Generate Paths
        roadmap_dict = models['roadmap_recommender']
        roadmap_vectorizer = roadmap_dict['vectorizer']
        roadmap_vectors = roadmap_dict['vectors']
        prepared_roadmap = roadmap_dict['data']
        
        user_vector_path = roadmap_vectorizer.transform([user_interest])
        similarity_scores_path = cosine_similarity(user_vector_path, roadmap_vectors).flatten()
        top_indices_path = similarity_scores_path.argsort()[::-1][:path_top_n]
        
        paths_df = prepared_roadmap.iloc[top_indices_path].copy()
        
        # 2. Generate Courses
        course_dict = models['course_recommender']
        course_vectorizer = course_dict['vectorizer']
        course_vectors = course_dict['vectors']
        prepared_courses = course_dict['data']
        
        user_vector_course = course_vectorizer.transform([user_interest])
        similarity_scores_course = cosine_similarity(user_vector_course, course_vectors).flatten()
        top_indices_course = similarity_scores_course.argsort()[::-1][:course_top_n]
        
        courses_df = prepared_courses.iloc[top_indices_course].copy()
        
        # Build paths JSON
        paths_list = []
        for idx, row in paths_df.iterrows():
            # If duration is NaN, substitute with default
            dur = row.get('Duration')
            if pd.isna(dur): dur = 'N/A'
            paths_list.append({
                'module_code': row.get('Module_Code', ''),
                'title': row.get('Course_Learning_Material', ''),
                'source': row.get('Source', ''),
                'level': row.get('Course_Level', ''),
                'type': row.get('Type_Free_Paid', ''),
                'module': row.get('Module', ''),
                'duration': str(dur),
                'difficulty': row.get('Difficulty_Level', ''),
                'link': row.get('Links', '')
            })
            
        # Build courses JSON
        courses_list = []
        for idx, row in courses_df.iterrows():
            dur = row.get('Duration')
            if pd.isna(dur): dur = 'N/A'
            courses_list.append({
                'module_code': row.get('Module_Code', ''),
                'title': row.get('Course_Learning_Material', ''),
                'source': row.get('Source', ''),
                'level': row.get('Course_Level', ''),
                'type': row.get('Type_Free_Paid', ''),
                'duration': str(dur),
                'prerequisites': row.get('Prerequisites', ''),
                'prework': row.get('Prework', ''),
                'link': row.get('Course_Learning_Material_Link', '')
            })
            
        return jsonify({
            'paths': paths_list,
            'courses': courses_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
