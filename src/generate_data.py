import pandas as pd
import numpy as np

np.random.seed(42)

n_transactions = 1000000

data = {
    "transaction_id": range(1, n_transactions + 1),
    "amount": np.random.uniform(5, 1000, n_transactions),
    "country": np.random.choice(
        ["Morocco", "France", "Spain", "Italy", "Germany"],
        n_transactions
    ),
    "payment_method": np.random.choice(
        ["card", "paypal", "bank_transfer"],
        n_transactions
    ),
    "device": np.random.choice(
        ["mobile", "desktop", "tablet"],
        n_transactions
    ),
    "hour": np.random.randint(0, 24, n_transactions),
}

df = pd.DataFrame(data)
# Calculate fraud risk
fraud_probability = (
    0.001
    + (df["amount"] > 800) * 0.015
    + (df["hour"] < 5) * 0.010
    + (df["payment_method"] == "bank_transfer") * 0.005
    + (df["device"] == "tablet") * 0.003
)

# Limit probability between 0 and 0.50
fraud_probability = np.clip(fraud_probability, 0, 0.05)

# Generate fraud label
df["is_fraud"] = np.random.binomial(1, fraud_probability)

print(df.head())
print()
print("Number of transactions:", len(df))
df.to_csv("data/raw/transactions.csv", index=False)

print("Data saved successfully!")