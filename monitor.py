import pandas as pd
import numpy as np
import sys # NEW: Allows us to trigger a system alarm

print("Starting MLOps Drift Monitor...")

# 1. Original Training Data
original_data = pd.DataFrame({
    'Amount': [15.0, 45.0, 12.0, 50.0],
    'Age': [25.0, 42.0, 19.0, 35.0]
})

# 2. Read live data from C++ API
new_data = pd.read_csv("transactions_log.csv", names=["Amount", "Age", "Prediction"])

avg_old_amount = original_data['Amount'].mean()
avg_new_amount = new_data['Amount'].mean()

print(f"Original Average Transaction: ${avg_old_amount}")
print(f"Today's Average Transaction: ${avg_new_amount}")

# 3. The Automation Trigger
if avg_new_amount > (avg_old_amount * 2):
    print("\n🚨 ALARM: MASSIVE DATA DRIFT DETECTED! 🚨")
    print("Sending failure code to GitHub to trigger retraining...")
    sys.exit(1) # This forces GitHub to start the retraining step!
else:
    print("\n✅ Data looks normal. AI is healthy.")
    sys.exit(0)

