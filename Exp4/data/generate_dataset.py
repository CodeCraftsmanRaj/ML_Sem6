import numpy as np
import pandas as pd
from config import N_SAMPLES, RANDOM_STATE

def generate_data():
    np.random.seed(RANDOM_STATE)

    Age = np.random.randint(18, 60, N_SAMPLES)
    Annual_Income = np.random.randint(15000, 150000, N_SAMPLES)
    Spending_Score = np.random.randint(1, 100, N_SAMPLES)

    Purchase = []
    for i in range(N_SAMPLES):
        if (Age[i] < 40 and Spending_Score[i] > 50) or \
           (Annual_Income[i] > 80000 and Spending_Score[i] > 40):
            Purchase.append(1)
        else:
            Purchase.append(0)

    df = pd.DataFrame({
        "Age": Age,
        "Annual_Income": Annual_Income,
        "Spending_Score": Spending_Score,
        "Purchase": Purchase
    })

    return df
