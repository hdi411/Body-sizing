import os
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import json
import re

logging.basicConfig(level=logging.INFO)

# ── Size chart ────────────────────────────────────────────────────────────────
size_chart = {
    "XX-Small": {"chest": (78.7, 81.3),  "waist": (64.8, 67.3)},
    "X-Small":  {"chest": (83.8, 86.4),  "waist": (69.8, 72.4)},
    "Small":    {"chest": (88.9, 94.0),  "waist": (74.9, 80.0)},
    "Medium":   {"chest": (97.0, 102.9), "waist": (81.9, 87.9)},
    "Large":    {"chest": (105.9, 111.8),"waist": (90.0, 95.0)},
    "X-Large":  {"chest": (114.3, 119.4),"waist": (97.8, 102.9)},
    "XX-Large": {"chest": (121.9, 127.0),"waist": (106.7, 111.8)},
}

def find_suitable_size(predicted: dict, chart: dict) -> str:
    waist = predicted.get("waist", 0)
    chest = predicted.get("chest", 0)
    for size, ranges in chart.items():
        w_min, w_max = ranges["waist"]
        c_min, c_max = ranges["chest"]
        if w_min <= waist <= w_max and c_min <= chest <= c_max:
            return size
    # Fallback: find closest waist match
    closest = min(chart.items(), key=lambda x: abs((x[1]["waist"][0] + x[1]["waist"][1]) / 2 - waist))
    return closest[0] + " (approximate)"

# ── GPT-4o Vision ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT = """You are a body measurement expert. The user has provided a front photo and a side photo of themselves, plus their height.

User height: {height_cm} cm

Analyze both photos carefully and estimate the following body measurements in centimeters:
- ankle
- arm_length
- bicep
- calf
- chest (bust circumference)
- forearm
- hip (hip circumference at widest point)
- leg_length
- shoulder_breadth
- thigh
- waist (circumference at narrowest point)
- wrist

Use the height as the scale reference to calculate real-world centimeter values.

Return ONLY a valid JSON object with measurement names as keys and numeric cm values as values. No explanation, no markdown, just JSON.

Example format:
{{"ankle": 22.5, "arm_length": 58.0, "bicep": 28.0, "calf": 35.0, "chest": 90.0, "forearm": 24.0, "hip": 95.0, "leg_length": 80.0, "shoulder_breadth": 38.0, "thigh": 52.0, "waist": 72.0, "wrist": 15.0}}"""

def analyze_with_gpt4o(front_b64: str, side_b64: str, height_cm: float) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT.format(height_cm=height_cm)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{front_b64}", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{side_b64}",  "detail": "high"}},
                ],
            }
        ],
    )

   raw = response.choices[0].message.content.strip()
    # Extract JSON object from anywhere in the response
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        raise ValueError("No JSON found in GPT response")
    return json.loads(match.group())
# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body received"}), 400
        if "front_image" not in data or "side_image" not in data:
            return jsonify({"error": "Both front_image and side_image are required"}), 400
        if "height" not in data:
            return jsonify({"error": "height (in cm) is required"}), 400

        height_cm = float(data["height"])
        front_b64 = data["front_image"]
        side_b64  = data["side_image"]

        logging.info(f"Received request — height: {height_cm} cm")

        measurements = analyze_with_gpt4o(front_b64, side_b64, height_cm)
        size = find_suitable_size(measurements, size_chart)

        return jsonify({
            "Predicted Measurements": {k: round(float(v), 1) for k, v in measurements.items()},
            "Suggested Size": size,
        })

    except json.JSONDecodeError as e:
        logging.error(f"GPT returned invalid JSON: {e}")
        return jsonify({"error": "GPT response could not be parsed. Try again."}), 500
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable before running.")
    app.run(debug=True, host="0.0.0.0", port=5000)
