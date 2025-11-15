// Global Variables
let currentUser = null;
let userStats = {};

// ====== MESSAGE FUNCTION ======
function showMessage(message, type, pageType) {
    const messageBoxId = pageType + 'Message';
    const messageBox = document.getElementById(messageBoxId);
    
    if (!messageBox) {
        console.error('Message box not found:', messageBoxId);
        return;
    }
    
    messageBox.textContent = message;
    messageBox.className = 'message-box show ' + type;
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        messageBox.classList.remove('show');
    }, 3000);
}

// ====== NOTIFICATION TOAST ======
function showNotification(message, type) {
    const toast = document.getElementById('notificationToast');
    
    toast.textContent = message;
    toast.className = 'notification-toast show ' + type;
    
    // Auto-hide after 4 seconds
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// ====== PAGE NAVIGATION ======
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');
}

function showLogin() {
    showPage('loginPage');
}

function showRegister() {
    showPage('registerPage');
}

function showDashboard() {
    if (!currentUser) {
        showLogin();
        return;
    }
    showPage('dashboardPage');
    updateDashboard();
}

function showPrediction() {
    if (!currentUser) {
        showLogin();
        return;
    }
    showPage('predictionPage');
}

function showProfile() {
    if (!currentUser) {
        showLogin();
        return;
    }
    showPage('profilePage');
    updateProfile();
}

// ====== AUTHENTICATION ======
function handleLogin() {
    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value.trim();

    if (!email || !password) {
        showMessage('⚠️ Please fill all fields!', 'error', 'login');
        return;
    }

    // Check if account exists
    if (!userStats[email]) {
        showMessage('❌ Account not registered! Please create an account first.', 'error', 'login');
        return;
    }

    // Check if password matches
    if (userStats[email].password !== password) {
        showMessage('❌ Incorrect password! Please try again.', 'error', 'login');
        return;
    }

    // Login successful
    const name = userStats[email].name;
    currentUser = { name, email };

    // Clear form
    document.getElementById('loginEmail').value = '';
    document.getElementById('loginPassword').value = '';

    console.log('Login successful:', currentUser);
    showMessage('✓ Login Successful! Welcome ' + name + '!', 'success', 'login');
    
    // Redirect after 1 second
    setTimeout(() => {
        showDashboard();
    }, 1000);
}

function handleRegister() {
    const name = document.getElementById('registerName').value.trim();
    const email = document.getElementById('registerEmail').value.trim();
    const password = document.getElementById('registerPassword').value;
    const confirm = document.getElementById('registerConfirm').value;

    if (!name || !email || !password || !confirm) {
        showMessage('⚠️ Please fill all fields!', 'error', 'register');
        return;
    }

    if (password !== confirm) {
        showMessage('❌ Passwords do not match!', 'error', 'register');
        return;
    }

    // Check if email already registered
    if (userStats[email]) {
        showMessage('❌ This email is already registered! Please use a different email.', 'error', 'register');
        return;
    }

    // Store user info with password
    userStats[email] = {
        password: password,
        name: name,
        total: 0,
        positive: 0,
        negative: 0,
        joined: new Date().toLocaleDateString()
    };

    // Clear form
    document.getElementById('registerName').value = '';
    document.getElementById('registerEmail').value = '';
    document.getElementById('registerPassword').value = '';
    document.getElementById('registerConfirm').value = '';

    console.log('Account created successfully! User must now login.');
    
    // Show success notification
    showNotification('✓ Account Successfully Created!\n\nNow please login with your email and password.', 'success');
    
    // Redirect to login after 2 seconds
    setTimeout(() => {
        showLogin();
    }, 2000);
}

function handleLogout() {
    currentUser = null;
    showLogin();
}

// ====== SENTIMENT ANALYSIS ======
async function analyzeReview() {
    const review = document.getElementById('reviewInput').value.trim();
    const model = document.querySelector('input[name="model"]:checked').value;

    if (!review) {
        alert('Enter a review!');
        return;
    }

    // LSTM check removed - let the backend handle availability

    const resultContainer = document.getElementById('resultContainer');
    resultContainer.innerHTML = '<p style="text-align: center; padding: 3rem; color: #666;"><i class="fas fa-spinner fa-spin"></i> Analyzing...</p>';
    document.getElementById('resultsSection').classList.remove('hidden');

    try {
        // Call the actual Flask API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                review: review,
                model: model === 'lr' ? 'logistic_regression' : 'lstm'
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            generateResults(data, model);
        } else {
            resultContainer.innerHTML = `<p style="text-align: center; padding: 3rem; color: #e74c3c;">Error: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultContainer.innerHTML = '<p style="text-align: center; padding: 3rem; color: #e74c3c;">Network error. Please try again.</p>';
    }
}

function generateResults(apiData, selectedModel) {
    let html = '';
    let totalPositive = 0;
    let totalNegative = 0;
    const resultContainer = document.getElementById('resultContainer');

    // Process predictions from API response
    const predictions = apiData.predictions;

    if (selectedModel === 'lr' && predictions.logistic_regression) {
        const result = predictions.logistic_regression;
        const isPositive = result.sentiment === 'Positive';
        const confidence = result.confidence;
        
        if (isPositive) totalPositive++;
        else totalNegative++;
        
        html += `
            <div class="result-card">
                <div class="result-header">
                    <h3>Logistic Regression</h3>
                    <div class="sentiment-badge ${isPositive ? 'sentiment-positive' : 'sentiment-negative'}">
                        ${isPositive ? '😊 POSITIVE' : '😞 NEGATIVE'}
                    </div>
                </div>
                <div class="result-details">
                    <div class="detail-item">
                        <h4>Sentiment</h4>
                        <div class="value">${result.sentiment}</div>
                    </div>
                    <div class="detail-item">
                        <h4>Confidence</h4>
                        <div class="value">${confidence.toFixed(1)}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                </div>
                ${result.error ? `<div class="error-message">Error: ${result.error}</div>` : ''}
            </div>
        `;
    }

    if (selectedModel === 'lstm' && predictions.lstm) {
        const result = predictions.lstm;
        const isPositive = result.sentiment === 'Positive';
        const confidence = result.confidence;
        
        if (isPositive) totalPositive++;
        else totalNegative++;
        
        html += `
            <div class="result-card">
                <div class="result-header">
                    <h3>LSTM Network</h3>
                    <div class="sentiment-badge ${isPositive ? 'sentiment-positive' : 'sentiment-negative'}">
                        ${isPositive ? '😊 POSITIVE' : '😞 NEGATIVE'}
                    </div>
                </div>
                <div class="result-details">
                    <div class="detail-item">
                        <h4>Sentiment</h4>
                        <div class="value">${result.sentiment}</div>
                    </div>
                    <div class="detail-item">
                        <h4>Confidence</h4>
                        <div class="value">${confidence.toFixed(1)}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%</div>
                        </div>
                    </div>
                </div>
                ${result.error ? `<div class="error-message">Error: ${result.error}</div>` : ''}
            </div>
        `;
    }

    // Show error if no valid predictions
    if (!html) {
        html = '<p style="text-align: center; padding: 3rem; color: #e74c3c;">No predictions available. Please try again.</p>';
    }

    resultContainer.innerHTML = html;

    // Update stats
    if (currentUser) {
        userStats[currentUser.email].total++;
        userStats[currentUser.email].positive += totalPositive;
        userStats[currentUser.email].negative += totalNegative;
        console.log('Updated stats:', userStats[currentUser.email]);
    }
}

function clearForm() {
    document.getElementById('reviewInput').value = '';
    document.getElementById('resultsSection').classList.add('hidden');
}

// ====== DASHBOARD UPDATE ======
function updateDashboard() {
    if (currentUser) {
        document.getElementById('userName').textContent = currentUser.name;
        const stats = userStats[currentUser.email] || { total: 0, positive: 0, negative: 0 };
        document.getElementById('totalAnalyses').textContent = stats.total;
        document.getElementById('positiveCount').textContent = stats.positive;
        document.getElementById('negativeCount').textContent = stats.negative;
    }
}

// ====== PROFILE UPDATE ======
function updateProfile() {
    if (currentUser) {
        const stats = userStats[currentUser.email] || {
            total: 0,
            joined: new Date().toLocaleDateString()
        };
        document.getElementById('profileName').textContent = currentUser.name;
        document.getElementById('profileEmail').textContent = currentUser.email;
        document.getElementById('profileAnalyses').textContent = stats.total;
        document.getElementById('profileJoined').textContent = stats.joined;
    }
}

// ====== TOGGLE PASSWORD ======
function togglePassword(icon, fieldId) {
    const field = document.getElementById(fieldId);
    
    if (field.type === 'password') {
        field.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        field.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
}

// ====== CHECK MODEL AVAILABILITY ======
async function checkModelAvailability() {
    try {
        const response = await fetch('/api/models-status');
        const data = await response.json();
        
        // Update LSTM option based on availability
        const lstmOption = document.querySelector('.model-option:nth-child(2)');
        const lstmInput = document.getElementById('modelLSTM');
        const lstmLabel = document.querySelector('label[for="modelLSTM"]');
        
        if (!data.lstm_model) {
            lstmOption.style.opacity = '0.5';
            lstmOption.style.pointerEvents = 'none';
            lstmInput.disabled = true;
            lstmLabel.querySelector('div:nth-child(2)').textContent = 'Currently Unavailable';
            
            // Add info message if not already present
            if (!document.querySelector('.model-info')) {
                const infoDiv = document.createElement('div');
                infoDiv.className = 'model-info';
                infoDiv.style.cssText = 'font-size: 0.85rem; color: #666; margin-top: 0.5rem; text-align: center;';
                infoDiv.innerHTML = '💡 LSTM model temporarily disabled due to TensorFlow compatibility issues';
                lstmOption.parentNode.parentNode.appendChild(infoDiv);
            }
        } else {
            // Enable LSTM if available
            lstmOption.style.opacity = '1';
            lstmOption.style.pointerEvents = 'auto';
            lstmInput.disabled = false;
            lstmLabel.querySelector('div:nth-child(2)').textContent = 'Deep Learning';
        }
        
        console.log('Model availability checked:', data);
    } catch (error) {
        console.error('Failed to check model availability:', error);
    }
}

// ====== PAGE LOAD ======
document.addEventListener('DOMContentLoaded', function() {
    console.log('MovieSent App Loaded Successfully!');
    checkModelAvailability();
    showLogin();
});