import pandas as pd
import numpy as np
import config

def generate_student_data(n_samples=config.N_SAMPLES):
    np.random.seed(config.RANDOM_STATE)
    
    # Feature 1: Study Hours (0 to 10)
    study_hours = np.random.randint(0, 11, n_samples)
    
    # Feature 2: Previous Exam Score (0 to 100)
    prev_score = np.random.randint(30, 101, n_samples)
    
    # Feature 3: Attendance Percentage (50 to 100)
    attendance = np.random.randint(50, 101, n_samples)
    
    # Logic for Target (Pass=1, Fail=0)
    # Passed if (Study > 4 AND Prev_Score > 40) OR (Attendance > 85)
    # Adding some noise to make it not perfectly separable
    y = []
    for i in range(n_samples):
        score = (study_hours[i] * 5) + (prev_score[i] * 0.5) + (attendance[i] * 0.2)
        noise = np.random.randint(-5, 6)
        final_score = score + noise
        
        # Threshold for passing
        if final_score > 75:
            y.append(1)
        else:
            y.append(0)
            
    data = pd.DataFrame({
        config.FEATURES[0]: study_hours,
        config.FEATURES[1]: prev_score,
        config.FEATURES[2]: attendance,
        config.TARGET: y
    })
    
    # Ensure all columns are integers
    data = data.astype(int)
    
    return data
