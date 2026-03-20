from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load models
model = pickle.load(open("model.pkl","rb"))
Q = np.load("rl_model.npy")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    tau = data["tau"]
    K = data["K"]

    ml_pred = model.predict([[tau, K]])

    # RL suggestion (simple)
    state = int(tau) % 10
    action = np.argmax(Q[state])

    return jsonify({
        "Kp": float(ml_pred[0][0]),
        "Ki": float(ml_pred[0][1]),
        "RL_action": int(action)
    })

if __name__ == "__main__":
    app.run(debug=True)