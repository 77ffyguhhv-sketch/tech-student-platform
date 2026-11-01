document.addEventListener('DOMContentLoaded', () => {
    // === INITIALIZATION ===
    initTheme();
    initLanguage();
    initSliders();
    initForms();
});

// === THEME MANAGER ===
function initTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;

    // Load saved theme or default to dark (since we want a modern premium look)
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        themeToggle.innerHTML = '☀️'; // light icon
    } else {
        document.body.classList.remove('dark-theme');
        themeToggle.innerHTML = '🌙'; // dark icon
    }

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark-theme');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        themeToggle.innerHTML = isDark ? '☀️' : '🌙';
    });
}

// === LANGUAGE MANAGER ===
let currentLang = 'en';

function initLanguage() {
    const langToggle = document.getElementById('lang-toggle');
    if (!langToggle) return;

    // Load saved language or default to en
    currentLang = localStorage.getItem('lang') || 'en';
    applyLanguage(currentLang);

    langToggle.addEventListener('click', () => {
        currentLang = currentLang === 'en' ? 'ar' : 'en';
        localStorage.setItem('lang', currentLang);
        applyLanguage(currentLang);
    });
}

function applyLanguage(lang) {
    const langToggle = document.getElementById('lang-toggle');
    if (langToggle) {
        langToggle.innerText = lang === 'en' ? 'العربية' : 'English';
    }

    // Set document directions
    if (lang === 'ar') {
        document.documentElement.dir = 'rtl';
        document.documentElement.lang = 'ar';
        document.body.classList.add('rtl');
    } else {
        document.documentElement.dir = 'ltr';
        document.documentElement.lang = 'en';
        document.body.classList.remove('rtl');
    }

    // Translate all elements with data-en/data-ar
    document.querySelectorAll('[data-en]').forEach(el => {
        el.innerText = el.getAttribute(`data-${lang}`);
    });

    // Translate all input placeholders with data-placeholder-en/data-placeholder-ar
    document.querySelectorAll('[data-placeholder-en]').forEach(el => {
        el.placeholder = el.getAttribute(`data-placeholder-${lang}`);
    });
}

// === SLIDERS VALUE SYNC ===
function initSliders() {
    document.querySelectorAll('.range-slider').forEach(slider => {
        const valueDisplay = slider.parentElement.querySelector('.slider-val');
        
        // Initial sync
        if (valueDisplay) {
            valueDisplay.innerText = slider.value;
        }

        // On input change
        slider.addEventListener('input', (e) => {
            if (valueDisplay) {
                valueDisplay.innerText = e.target.value;
            }
        });
    });
}

// === FORM SUBMISSIONS & API CALLS ===
function initForms() {
    // 1. Smart Major Advisor Form
    const majorForm = document.getElementById('major-advisor-form');
    if (majorForm) {
        majorForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading(true);
            
            const payload = {
                'Linguistic': document.getElementById('Linguistic').value,
                'Musical': document.getElementById('Musical').value,
                'Bodily': document.getElementById('Bodily').value,
                'Logical - Mathematical': document.getElementById('Logical-Mathematical').value,
                'Spatial-Visualization': document.getElementById('Spatial-Visualization').value,
                'Interpersonal': document.getElementById('Interpersonal').value,
                'Intrapersonal': document.getElementById('Intrapersonal').value,
                'Naturalist': document.getElementById('Naturalist').value
            };

            try {
                const response = await fetch('/api/predict-major', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                showLoading(false);

                if (result.error) {
                    alert(result.error);
                    return;
                }

                // Show Results Area
                const resultsWrapper = document.getElementById('results-wrapper');
                resultsWrapper.style.display = 'block';

                // Recommended profession text
                const profTitle = document.getElementById('recommended-profession');
                const profEn = result.recommended_profession;
                const profAr = result.recommended_profession_ar || result.recommended_profession;
                profTitle.setAttribute('data-en', profEn);
                profTitle.setAttribute('data-ar', profAr);
                profTitle.innerText = currentLang === 'ar' ? profAr : profEn;

                // Probability breakdown
                const probList = document.getElementById('probability-list');
                probList.innerHTML = '';

                result.top_predictions.forEach(item => {
                    const pct = item.percentage;
                    const nameEn = item.profession;
                    const nameAr = item.profession_ar || item.profession;
                    
                    const itemHtml = `
                        <div class="prob-item">
                            <div class="prob-labels">
                                <span data-en="${nameEn}" data-ar="${nameAr}">${currentLang === 'ar' ? nameAr : nameEn}</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill" style="width: 0%"></div>
                            </div>
                        </div>
                    `;
                    probList.insertAdjacentHTML('beforeend', itemHtml);
                    
                    // Trigger animations
                    setTimeout(() => {
                        const fills = probList.querySelectorAll('.prob-fill');
                        fills[fills.length - 1].style.width = `${pct}%`;
                    }, 50);
                });

                // Scroll to results
                resultsWrapper.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                showLoading(false);
                alert('Connection error occurred.');
                console.error(err);
            }
        });
    }

    // 2. Study Success Predictor Form
    const successForm = document.getElementById('study-success-form');
    if (successForm) {
        successForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading(true);

            // Get comma-separated list of selected checkboxes for Skills, Languages, ClubMemberships
            const getSelectedValues = (name) => {
                const checked = Array.from(document.querySelectorAll(`input[name="${name}"]:checked`));
                return checked.map(el => el.value).join(', ');
            };

            const payload = {
                'AcademicInterest': document.getElementById('AcademicInterest').value,
                'ExtracurricularActivities': document.getElementById('ExtracurricularActivities').value,
                'Skills': getSelectedValues('Skills') || 'None',
                'Location': document.getElementById('Location').value,
                'YearOfStudy': document.getElementById('YearOfStudy').value,
                'Major': document.getElementById('Major').value,
                'Languages': getSelectedValues('Languages') || 'English',
                'ClubMemberships': getSelectedValues('ClubMemberships') || 'None',
                'ResearchInterests': document.getElementById('ResearchInterests').value
            };

            try {
                const response = await fetch('/api/predict-success', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                showLoading(false);

                if (result.error) {
                    alert(result.error);
                    return;
                }

                const resultsWrapper = document.getElementById('results-wrapper');
                resultsWrapper.style.display = 'block';

                // Display GPA and Success Category
                document.getElementById('gpa-value').innerText = result.predicted_gpa;
                
                const categoryBadge = document.getElementById('success-category');
                categoryBadge.setAttribute('data-en', result.success_level);
                categoryBadge.setAttribute('data-ar', result.success_level_ar);
                categoryBadge.innerText = currentLang === 'en' ? result.success_level : result.success_level_ar;
                
                // Clear old classes
                categoryBadge.className = 'result-badge';
                categoryBadge.classList.add(result.badge_class);

                resultsWrapper.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                showLoading(false);
                alert('Connection error occurred.');
                console.error(err);
            }
        });
    }

    // 3. Smart Device Advisor Forms (Double features: Recommendation & Pricing)
    const deviceRecForm = document.getElementById('device-rec-form');
    if (deviceRecForm) {
        deviceRecForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading(true);

            const payload = {
                'budget': document.getElementById('budget').value,
                'min_rating': document.getElementById('min_rating').value,
                'keyword': document.getElementById('keyword').value
            };

            try {
                const response = await fetch('/api/recommend-devices', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                showLoading(false);

                if (result.error) {
                    alert(result.error);
                    return;
                }

                const resultsWrapper = document.getElementById('results-wrapper');
                resultsWrapper.style.display = 'block';

                const deviceContainer = document.getElementById('device-list');
                deviceContainer.innerHTML = '';

                if (!result.devices || result.devices.length === 0) {
                    deviceContainer.innerHTML = `
                        <div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-secondary);" 
                             data-en="No devices found matching your requirements. Try raising your budget or lowering the rating requirement."
                             data-ar="لم يتم العثور على أجهزة تطابق متطلباتك. جرب زيادة الميزانية أو تقليل التقييم المطلوب.">
                            No devices found matching your requirements. Try raising your budget or lowering the rating requirement.
                        </div>
                    `;
                    // Refresh translations for newly inserted element
                    applyLanguage(currentLang);
                } else {
                    result.devices.forEach(dev => {
                        const nameEn = dev.name;
                        const nameAr = dev.name_ar || dev.name;
                        const specsEn = dev.specs || 'N/A';
                        const specsAr = dev.specs_ar || dev.specs || 'غير متوفر';

                        const deviceHtml = `
                            <div class="device-result-card">
                                <span class="device-rating">⭐ ${dev.rating}</span>
                                <h4 class="device-title" data-en="${nameEn}" data-ar="${nameAr}">${currentLang === 'ar' ? nameAr : nameEn}</h4>
                                <div class="device-price">${dev.price.toLocaleString()} ر.س</div>
                                <div class="device-specs">
                                    <strong data-en="Specs / Features:" data-ar="المواصفات والمميزات:">${currentLang === 'ar' ? 'المواصفات والمميزات:' : 'Specs / Features:'}</strong><br>
                                    <span data-en="${specsEn}" data-ar="${specsAr}">${currentLang === 'ar' ? specsAr : specsEn}</span>
                                </div>
                            </div>
                        `;
                        deviceContainer.insertAdjacentHTML('beforeend', deviceHtml);
                    });
                }

                resultsWrapper.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                showLoading(false);
                alert('Connection error occurred.');
                console.error(err);
            }
        });
    }

    const devicePredictForm = document.getElementById('device-predict-form');
    if (devicePredictForm) {
        devicePredictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading(true);

            const payload = {
                'text_features': document.getElementById('predict_text').value,
                'rating': document.getElementById('predict_rating').value
            };

            try {
                const response = await fetch('/api/predict-device', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                showLoading(false);

                if (result.error) {
                    alert(result.error);
                    return;
                }

                const resultsWrapper = document.getElementById('results-wrapper-predict');
                resultsWrapper.style.display = 'block';

                document.getElementById('predicted-price-val').innerText = `${result.predicted_price.toLocaleString()} ر.س`;

                resultsWrapper.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                showLoading(false);
                alert('Connection error occurred.');
                console.error(err);
            }
        });
    }

    // 4. Learning Roadmap Generator Form
    const roadmapForm = document.getElementById('learning-roadmap-form');
    if (roadmapForm) {
        roadmapForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoading(true);

            const payload = {
                'interest': document.getElementById('interest').value,
                'path_count': document.getElementById('path_count').value,
                'course_count': document.getElementById('course_count').value
            };

            try {
                const response = await fetch('/api/generate-roadmap', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                showLoading(false);

                if (result.error) {
                    alert(result.error);
                    return;
                }

                const resultsWrapper = document.getElementById('results-wrapper');
                resultsWrapper.style.display = 'block';

                // Populate Pathways Timeline
                const timelineContainer = document.getElementById('roadmap-timeline');
                timelineContainer.innerHTML = '';

                if (!result.paths || result.paths.length === 0) {
                    timelineContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No pathway modules recommended.</p>';
                } else {
                    result.paths.forEach((path, idx) => {
                        const sideClass = idx % 2 === 0 ? 'timeline-item-left' : 'timeline-item-right';
                        const pathHtml = `
                            <div class="timeline-item ${sideClass}">
                                <div class="timeline-content">
                                    <span class="timeline-badge">${path.level}</span>
                                    <div class="timeline-title">${path.title}</div>
                                    <div style="font-size:0.9rem; color:var(--text-secondary); margin:0.25rem 0;">
                                        <strong>Module:</strong> ${path.module}
                                    </div>
                                    <div class="timeline-meta">
                                        <span>🏢 ${path.source}</span>
                                        <span>⏱️ ${path.duration}</span>
                                        <span>📈 ${path.difficulty}</span>
                                        <span>💰 ${path.type}</span>
                                    </div>
                                    ${path.link ? `<a href="${path.link}" target="_blank" class="course-link">🔗 Learn More</a>` : ''}
                                </div>
                            </div>
                        `;
                        timelineContainer.insertAdjacentHTML('beforeend', pathHtml);
                    });
                }

                // Populate Courses Grid
                const coursesContainer = document.getElementById('courses-list');
                coursesContainer.innerHTML = '';

                if (!result.courses || result.courses.length === 0) {
                    coursesContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary); grid-column: 1/-1;">No additional course materials found.</p>';
                } else {
                    result.courses.forEach(course => {
                        const courseHtml = `
                            <div class="course-card">
                                <span class="course-badge">${course.type}</span>
                                <h4 class="course-title">${course.title}</h4>
                                <div class="course-details">
                                    <span>🏢 <strong>Source:</strong> ${course.source}</span>
                                    <span>⏱️ <strong>Duration:</strong> ${course.duration}</span>
                                    <span>🎓 <strong>Level:</strong> ${course.level}</span>
                                    <span>⚠️ <strong>Prerequisites:</strong> ${course.prerequisites}</span>
                                    <span>⚙️ <strong>Prework:</strong> ${course.prework}</span>
                                </div>
                                ${course.link ? `<a href="${course.link}" target="_blank" class="course-link" style="margin-top:1rem;">🔗 Go to Course</a>` : ''}
                            </div>
                        `;
                        coursesContainer.insertAdjacentHTML('beforeend', courseHtml);
                    });
                }

                resultsWrapper.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                showLoading(false);
                alert('Connection error occurred.');
                console.error(err);
            }
        });
    }
}

// === UTILITIES ===
function showLoading(show) {
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
        loadingEl.style.display = show ? 'flex' : 'none';
    }
}
