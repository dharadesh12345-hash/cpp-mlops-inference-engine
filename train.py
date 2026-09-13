from sklearn.ensemble import RandomForestClassifier
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("Training model...")

# 1. Create fake training data: [Transaction Amount, User Age]
X_train = np.array([
    [15.0, 25.0],   # Normal
    [45.0, 42.0],   # Normal
    [12.0, 19.0],   # Normal
    [5000.0, 22.0]  # FRAUD! (Young person, massive transaction)
], dtype=np.float32)

# 0 = Normal, 1 = Fraud
y_train = np.array([0, 0, 0, 1])

# 2. Train a basic Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 3. Convert the Python model into an ONNX file (The Bridge)
# We tell ONNX to expect 2 float numbers as input
initial_type = [('float_input', FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 4. Save the file so C++ can read it later!
with open("fraud_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("SUCCESS: Model trained and saved as 'fraud_model.onnx'!")
