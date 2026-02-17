import numpy as np
import pandas as pd

def generate_data(n_samples=400, random_state=42):
    np.random.seed(random_state)

    Age = np.random.randint(18, 60, n_samples)
    Annual_Income = np.random.randint(15000, 150000, n_samples)
    Spending_Score = np.random.randint(1, 100, n_samples)

    # Purchase logic (custom rule-based pattern)
    Purchase = []
    for i in range(n_samples):
        if (Age[i] < 40 and Spending_Score[i] > 50) or (Annual_Income[i] > 80000 and Spending_Score[i] > 40):
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
