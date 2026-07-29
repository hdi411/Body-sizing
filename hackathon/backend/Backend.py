import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
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
    closest = min(chart.items(), key=lambda x: abs((x[1]["waist"][0] + x[1]["waist"][1]) / 2 - waist))
    return closest[0] + " (approximate)"

# ── Claude Vision ─────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

PROMPT = """You are a clothing size assistant helping a user find their correct clothing size.

The user's height is {height_cm} cm. Using the two photos (front and side view) and the height as a scale reference, estimate these measurements in centimeters:
ankle, arm_length, bicep, calf, chest, forearm, hip, leg_length, shoulder_breadth, thigh, waist, wrist

Return ONLY a JSON object. No explanation, no markdown.

Example: {{"ankle": 22.5, "arm_length": 58.0, "bicep": 28.0, "calf": 35.0, "chest": 90.0, "forearm": 24.0, "hip": 95.0, "leg_length": 80.0, "shoulder_breadth": 38.0, "thigh": 52.0, "waist": 72.0, "wrist": 15.0}}"""

def detect_media_type(b64_string: str) -> str:
    # Detect image type from first bytes
    import base64
    header = base64.b64decode(b64_string[:16])
    if header[:4] == b'\x89PNG':
        return "image/png"
    elif header[:2] in (b'\xff\xd8', b'\xff\xe0', b'\xff\xe1'):
        return "image/jpeg"
    return "image/jpeg"  # default

def analyze_with_claude(front_b64: str, side_b64: str, height_cm: float) -> dict:
    front_media = detect_media_type(front_b64)
    side_media  = detect_media_type(side_b64)

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": front_media, "data": front_b64}
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": side_media, "data": side_b64}
                },
                {
                    "type": "text",
                    "text": PROMPT.format(height_cm=height_cm)
                }
            ]
        }]
    )

    raw = response.content[0].text.strip()
    logging.info(f"Claude raw response: {raw[:300]}")

    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        raise ValueError(f"No JSON in response: {raw[:100]}")
    return json.loads(match.group())

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

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

        measurements = analyze_with_claude(front_b64, side_b64, height_cm)
        size = find_suitable_size(measurements, size_chart)

        return jsonify({
            "Predicted Measurements": {k: round(float(v), 1) for k, v in measurements.items()},
            "Suggested Size": size,
        })

    except json.JSONDecodeError as e:
        logging.error(f"Claude returned invalid JSON: {e}")
        return jsonify({"error": "Response could not be parsed. Try again."}), 500
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable before running.")
    app.run(debug=True, host="0.0.0.0", port=5000)
