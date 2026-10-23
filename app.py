import streamlit as st
import cv2
import os
import tempfile
import subprocess
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Animal Behaviour Risk Prediction",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 AI-Based Animal Behaviour and Risk Prediction System")
st.write("YOLOv8 + ByteTrack + Random Forest + Explainable AI")

# ============================================================
# 2. MODEL PATHS AND SETTINGS
# ============================================================

MODEL_PATH = "models/best.pt"
TRACKER_PATH = "models/bytetrack_custom.yaml"
RF_PATH = "models/generalized_behavior_random_forest_windows_.pkl"

OUTPUT_DIR = "outputs"
VIDEO_DIR = "input_videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

FFMPEG_PATH = shutil.which("ffmpeg")

CLASS_NAMES = {
    0: "Human",
    1: "Dog",
    2: "Elephant",
    3: "Monkey"
}

RISK_MAP = {
    "Calm": "Low",
    "Alert": "Medium",
    "Stressed": "High",
    "Aggressive": "High"
}

ALERT_MAP = {
    "Low": "NORMAL - LOW RISK",
    "Medium": "WARNING - MONITOR ANIMAL",
    "High": "HIGH RISK ALERT"
}

REQUIRED_FILES = {
    "YOLO model": MODEL_PATH,
    "ByteTrack tracker": TRACKER_PATH,
    "Random Forest model": RF_PATH
}

for name, path in REQUIRED_FILES.items():
    if not os.path.exists(path):
        st.error(f"Missing {name}: {path}")
        st.stop()

# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():
    yolo_model = YOLO(MODEL_PATH)
    package = joblib.load(RF_PATH)

    random_forest = package["model"]
    animal_encoder = package["animal_encoder"]
    numeric_features = list(package["numeric_features"])
    numeric_medians = package.get("numeric_medians", {})

    return (
        yolo_model,
        random_forest,
        animal_encoder,
        numeric_features,
        numeric_medians
    )


model, rf, animal_encoder, numeric_features, numeric_medians = (
    load_models()
)

# The encoded animal class is a separate input feature in this pipeline.
FEATURE_COLUMNS = list(numeric_features)

if "animal_class_encoded" not in FEATURE_COLUMNS:
    FEATURE_COLUMNS.append("animal_class_encoded")

# ============================================================
# 4. SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=0.80,
    value=0.20,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "YOLO Image Size",
    [640, 960, 1280],
    index=1
)

st.sidebar.write("Supported Classes")
st.sidebar.write("Human")
st.sidebar.write("Dog")
st.sidebar.write("Elephant")
st.sidebar.write("Monkey")

st.sidebar.caption(
    "Risk predictions are estimates from the trained model."
)

# ============================================================
# 5. VIDEO INPUT
# ============================================================

st.subheader("Video Input")

input_mode = st.radio(
    "Choose video source",
    ["Built-in / Project Video", "Upload Input Video"],
    horizontal=True
)

video_path = None
video_source_name = None
temporary_input_path = None

if input_mode == "Built-in / Project Video":

    supported_extensions = (".mp4", ".avi", ".mov", ".mkv")

    project_videos = sorted(
        os.path.join(VIDEO_DIR, name)
        for name in os.listdir(VIDEO_DIR)
        if name.lower().endswith(supported_extensions)
    )

    if not project_videos:
        st.warning(
            "No built-in video found. Place your video inside "
            "the input_videos folder and run the dashboard again."
        )
        st.stop()

    # Automatically select the first video alphabetically.
    # There is no video-selection dropdown.
    video_path = project_videos[0]
    video_source_name = os.path.basename(video_path)

    st.info(f"Built-in project video: {video_source_name}")

else:

    uploaded_video = st.file_uploader(
        "Upload Animal Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is None:
        st.info("Upload a video to continue.")
        st.stop()

    suffix = os.path.splitext(uploaded_video.name)[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(uploaded_video.getbuffer())
    temp_file.close()

    temporary_input_path = temp_file.name
    video_path = temporary_input_path
    video_source_name = uploaded_video.name

# ============================================================
# 6. READ VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("Unable to open the selected video.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

if not np.isfinite(fps) or fps <= 0:
    fps = 25.0

st.info(
    f"Video: {video_source_name} | "
    f"Resolution: {width} x {height} | "
    f"FPS: {fps:.2f} | "
    f"Frames: {total_frames}"
)

# ============================================================
# 7. START ANALYSIS
# ============================================================

if not st.button(
    "▶ Start Complete Analysis",
    type="primary",
    use_container_width=True
):
    st.stop()

tracking_video = os.path.join(
    OUTPUT_DIR, "tracking_output.mp4"
)

final_raw_video = os.path.join(
    OUTPUT_DIR, "final_behavior_raw.mp4"
)

final_video = os.path.join(
    OUTPUT_DIR, "animal_behavior_risk_final.mp4"
)

results_csv = os.path.join(
    OUTPUT_DIR, "animal_behavior_risk_results.csv"
)

# Remove outputs from the previous analysis.
for output_file in [
    tracking_video,
    final_raw_video,
    final_video,
    results_csv
]:
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except OSError:
            pass

# ============================================================
# 8. YOLO DETECTION AND BYTETRACK TRACKING
# ============================================================

st.subheader("1. Detection and Tracking")

cap = cv2.VideoCapture(video_path)

writer = cv2.VideoWriter(
    tracking_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    st.error("Unable to create the tracking video.")
    st.stop()

tracking_data = []
frame_number = 0

progress = st.progress(0)
status = st.empty()

# Reset tracking state before processing this video.
model.predictor = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        tracker=TRACKER_PATH,
        persist=True,
        conf=confidence,
        imgsz=image_size,
        classes=[0, 1, 2, 3],
        verbose=False
    )

    result = results[0]
    plotted = result.plot()

    if plotted.shape[:2] != (height, width):
        plotted = cv2.resize(plotted, (width, height))

    writer.write(plotted)

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            # Fallback IDs when tracking IDs are unavailable.
            ids = np.arange(len(xyxy))

        for box, class_id, track_id in zip(xyxy, classes, ids):

            x1, y1, x2, y2 = box

            tracking_data.append({
                "frame": frame_number,
                "track_id": int(track_id),
                "class_id": int(class_id),
                "class_name": CLASS_NAMES.get(
                    int(class_id), "Unknown"
                ),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2)
            })

    progress.progress(
        min(frame_number / max(total_frames, 1), 1.0)
    )

    status.write(
        f"Processing frame {frame_number} of {total_frames}"
    )

cap.release()
writer.release()

progress.empty()
status.empty()

if not tracking_data:
    st.error("No objects were detected in the video.")
    st.stop()

df = pd.DataFrame(tracking_data)

st.success("Detection and tracking completed.")

# ============================================================
# 9. BEHAVIOUR FEATURE EXTRACTION
# ============================================================

st.subheader("2. Behaviour Feature Extraction")

group_columns = ["class_name", "track_id"]

df = df.sort_values(
    group_columns + ["frame"]
).reset_index(drop=True)

df["dx"] = (
    df.groupby(group_columns)["center_x"]
    .diff()
    .fillna(0)
)

df["dy"] = (
    df.groupby(group_columns)["center_y"]
    .diff()
    .fillna(0)
)

df["distance_pixels"] = np.sqrt(
    df["dx"] ** 2 + df["dy"] ** 2
)

df["speed_pixels_per_second"] = (
    df["distance_pixels"] * fps
)

df["acceleration"] = (
    df.groupby(group_columns)["speed_pixels_per_second"]
    .diff()
    .fillna(0)
    .abs()
    * fps
)

df["direction"] = np.degrees(
    np.arctan2(df["dy"], df["dx"])
)

df["direction_change"] = (
    df.groupby(group_columns)["direction"]
    .diff()
    .abs()
    .fillna(0)
)

# ============================================================
# 10. HUMAN-ANIMAL DISTANCE
# ============================================================

st.subheader("3. Human-Animal Distance")

human_df = df[df["class_id"] == 0].copy()

animal_df = df[
    df["class_id"].isin([1, 2, 3])
].copy()

if animal_df.empty:
    st.warning("No Dog, Elephant or Monkey was detected.")
    st.stop()

human_positions = defaultdict(list)

for _, row in human_df.iterrows():
    human_positions[int(row["frame"])].append(
        (row["center_x"], row["center_y"])
    )

distance_values = []

for _, row in animal_df.iterrows():

    positions = human_positions.get(
        int(row["frame"]), []
    )

    if not positions:
        distance_values.append(np.nan)
    else:
        distances = [
            np.sqrt(
                (row["center_x"] - hx) ** 2
                + (row["center_y"] - hy) ** 2
            )
            for hx, hy in positions
        ]

        distance_values.append(min(distances))

animal_df["human_distance"] = distance_values

# ============================================================
# 11. CREATE BEHAVIOUR WINDOWS
# ============================================================

st.subheader("4. Behaviour Prediction")

window_frames = max(int(fps), 1)
step_frames = max(int(fps * 0.25), 1)
minimum_detections = 5

windows = []

for (animal_class, track_id), group in animal_df.groupby(
    ["class_name", "track_id"]
):

    group = group.sort_values("frame")

    if len(group) < minimum_detections:
        continue

    first_frame = int(group["frame"].min())
    last_frame = int(group["frame"].max())

    start_frame = first_frame

    while start_frame <= last_frame:

        end_frame = start_frame + window_frames - 1

        window = group[
            (group["frame"] >= start_frame)
            & (group["frame"] <= end_frame)
        ]

        if len(window) >= minimum_detections:

            windows.append({
                "animal_class": animal_class,
                "track_id": int(track_id),
                "start_frame": start_frame,
                "end_frame": end_frame,

                "mean_speed": window[
                    "speed_pixels_per_second"
                ].mean(),

                "max_speed": window[
                    "speed_pixels_per_second"
                ].max(),

                "speed_std": window[
                    "speed_pixels_per_second"
                ].std(),

                "mean_acceleration": window[
                    "acceleration"
                ].mean(),

                "mean_direction_change": window[
                    "direction_change"
                ].mean(),

                "mean_human_distance": window[
                    "human_distance"
                ].mean(),

                "minimum_human_distance": window[
                    "human_distance"
                ].min(),

                "distance_std": window[
                    "human_distance"
                ].std(),

                "human_detected_ratio": window[
                    "human_distance"
                ].notna().mean()
            })

        start_frame += step_frames

prediction_df = pd.DataFrame(windows)

if prediction_df.empty:
    st.warning(
        "No usable behaviour windows were created. "
        "Try another video or check the detections."
    )
    st.stop()

# ============================================================
# 12. PREPARE FEATURES FOR THE RANDOM FOREST
# ============================================================

for feature in numeric_features:
    if feature not in prediction_df.columns:
        prediction_df[feature] = np.nan

prediction_df[numeric_features] = (
    prediction_df[numeric_features]
    .replace([np.inf, -np.inf], np.nan)
)

for feature in numeric_features:

    if feature in numeric_medians:
        fill_value = numeric_medians[feature]
    else:
        fill_value = 0.0

    prediction_df[feature] = (
        prediction_df[feature].fillna(fill_value)
    )

supported_animals = list(animal_encoder.classes_)

prediction_df = prediction_df[
    prediction_df["animal_class"].isin(supported_animals)
].copy()

if prediction_df.empty:
    st.warning("No supported animal class was found.")
    st.stop()

prediction_df["animal_class_encoded"] = (
    animal_encoder.transform(
        prediction_df["animal_class"].astype(str)
    )
)

# Keep the exact feature order expected by the model.
X = prediction_df[FEATURE_COLUMNS].copy()

# ============================================================
# 13. RANDOM FOREST BEHAVIOUR PREDICTION
# ============================================================

prediction_df["behavior_prediction"] = rf.predict(X)

if hasattr(rf, "predict_proba"):
    prediction_df["prediction_confidence"] = (
        rf.predict_proba(X).max(axis=1)
    )
else:
    prediction_df["prediction_confidence"] = 0.0

prediction_df["risk_level"] = (
    prediction_df["behavior_prediction"].map(RISK_MAP)
)

prediction_df["alert"] = (
    prediction_df["risk_level"].map(ALERT_MAP)
)

prediction_df.to_csv(results_csv, index=False)

st.success(
    "Behaviour, risk and alert prediction completed."
)

# ============================================================
# 14. SUMMARY
# ============================================================

st.subheader("5. Analysis Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Behaviour Windows", len(prediction_df))

with c2:
    st.metric(
        "Calm",
        int((prediction_df["behavior_prediction"] == "Calm").sum())
    )

with c3:
    st.metric(
        "Medium Risk",
        int((prediction_df["risk_level"] == "Medium").sum())
    )

with c4:
    st.metric(
        "High Risk",
        int((prediction_df["risk_level"] == "High").sum())
    )

st.write("### Behaviour Distribution")
st.bar_chart(
    prediction_df["behavior_prediction"].value_counts()
)

st.write("### Risk Distribution")
st.bar_chart(
    prediction_df["risk_level"].value_counts()
)

high_risk = prediction_df[
    prediction_df["risk_level"] == "High"
]

medium_risk = prediction_df[
    prediction_df["risk_level"] == "Medium"
]

if len(high_risk) > 0:
    st.error(
        f"🚨 HIGH RISK ALERT: "
        f"{len(high_risk)} high-risk windows detected."
    )

if len(medium_risk) > 0:
    st.warning(
        f"⚠️ WARNING: "
        f"{len(medium_risk)} medium-risk windows detected."
    )

# ============================================================
# 15. DISPLAY BEHAVIOUR AND RISK RESULTS
# ============================================================

st.subheader("6. Behaviour and Risk Results")

display_df = prediction_df[
    [
        "animal_class",
        "track_id",
        "start_frame",
        "end_frame",
        "behavior_prediction",
        "prediction_confidence",
        "risk_level",
        "alert"
    ]
].copy()

display_df["prediction_confidence"] = (
    display_df["prediction_confidence"] * 100
).round(1)

display_df.columns = [
    "Animal",
    "Track ID",
    "Start Frame",
    "End Frame",
    "Behaviour",
    "Confidence (%)",
    "Risk",
    "Alert"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 16. EXPLAINABLE AI (XAI) USING SHAP
# ============================================================

st.subheader(
    "7. Explainable AI (XAI): Why was this behaviour predicted?"
)

st.write(
    "SHAP explains which input features influenced the Random Forest "
    "prediction. You can inspect overall feature importance and the "
    "contribution of features for one selected prediction."
)

st.info(
    "SHAP explains the model's prediction. It does not prove an animal's "
    "actual emotional state or intention. Distances and speeds in this "
    "system are measured in pixels, not metres."
)


def friendly_feature_name(feature):
    """Convert technical feature names into readable labels."""

    labels = {
        "mean_speed": "Average movement speed",
        "max_speed": "Maximum movement speed",
        "speed_std": "Variation in movement speed",
        "mean_acceleration": "Average acceleration",
        "mean_direction_change": "Movement direction changes",
        "mean_human_distance": "Average distance from human",
        "minimum_human_distance": "Minimum distance from human",
        "distance_std": "Variation in human distance",
        "human_detected_ratio": "Human presence ratio",
        "animal_class_encoded": "Encoded animal class"
    }

    return labels.get(
        feature,
        feature.replace("_", " ").title()
    )


def get_shap_contributions(model_object, input_row, predicted_label):
    """
    Return SHAP values for the selected predicted class.

    Supports common SHAP output formats for classification models.
    """

    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed. Run: python -m pip install shap"

    try:
        explainer = shap.TreeExplainer(model_object)
        explanation = explainer(input_row)

        values = np.asarray(explanation.values)
        model_classes = list(
            getattr(model_object, "classes_", [])
        )

        if predicted_label in model_classes:
            class_index = model_classes.index(predicted_label)
        else:
            class_index = 0

        if values.ndim == 3:
            # Usually samples x features x classes.
            if class_index >= values.shape[2]:
                class_index = 0

            contributions = values[0, :, class_index]

        elif values.ndim == 2:
            # One sample x features.
            contributions = values[0, :]

        elif values.ndim == 1:
            contributions = values

        else:
            # Compatibility with older SHAP versions.
            old_values = explainer.shap_values(input_row)

            if isinstance(old_values, list):
                class_index = min(
                    class_index,
                    len(old_values) - 1
                )
                contributions = np.asarray(
                    old_values[class_index]
                )[0]

            else:
                old_values = np.asarray(old_values)

                if old_values.ndim == 3:
                    contributions = old_values[
                        0, :, min(class_index, old_values.shape[2] - 1)
                    ]
                elif old_values.ndim == 2:
                    contributions = old_values[0]
                else:
                    return None, "Unsupported SHAP output format."

        contributions = np.asarray(contributions).reshape(-1)

        if len(contributions) != len(input_row.columns):
            # Try the older SHAP interface if necessary.
            old_values = explainer.shap_values(input_row)

            if isinstance(old_values, list):
                class_index = min(
                    class_index,
                    len(old_values) - 1
                )
                contributions = np.asarray(
                    old_values[class_index]
                )[0].reshape(-1)

            else:
                old_values = np.asarray(old_values)

                if old_values.ndim == 3:
                    contributions = old_values[
                        0, :, min(class_index, old_values.shape[2] - 1)
                    ].reshape(-1)

                elif old_values.ndim == 2:
                    contributions = old_values[0].reshape(-1)

                else:
                    return None, "Could not match SHAP values to features."

        if len(contributions) != len(input_row.columns):
            return None, (
                "The number of SHAP values does not match the "
                "number of model input features."
            )

        contribution_df = pd.DataFrame({
            "Feature": list(input_row.columns),
            "Input value": input_row.iloc[0].to_numpy(),
            "SHAP contribution": contributions
        })

        contribution_df["Absolute contribution"] = (
            contribution_df["SHAP contribution"].abs()
        )

        contribution_df = contribution_df.sort_values(
            "Absolute contribution",
            ascending=False
        )

        return contribution_df, None

    except Exception as exc:
        return None, (
            f"SHAP could not explain this prediction: {exc}"
        )


# Select a prediction window to explain.
if not prediction_df.empty:

    xai_choices = prediction_df.reset_index(drop=True)

    selected_window_index = st.selectbox(
        "Select a prediction window to explain",
        options=list(range(len(xai_choices))),
        format_func=lambda i: (
            f"{xai_choices.loc[i, 'animal_class']} | "
            f"Track {int(xai_choices.loc[i, 'track_id'])} | "
            f"Frames {int(xai_choices.loc[i, 'start_frame'])}-"
            f"{int(xai_choices.loc[i, 'end_frame'])} | "
            f"{xai_choices.loc[i, 'behavior_prediction']}"
        )
    )

    selected_row = xai_choices.iloc[selected_window_index]
    selected_label = selected_row["behavior_prediction"]

    st.markdown("### Selected Prediction")

    xai_col1, xai_col2, xai_col3 = st.columns(3)

    with xai_col1:
        st.metric("Animal", selected_row["animal_class"])

    with xai_col2:
        st.metric("Predicted Behaviour", selected_label)

    with xai_col3:
        st.metric("Risk Level", selected_row["risk_level"])

    st.write(
        "Model confidence: "
        f"{float(selected_row['prediction_confidence']) * 100:.1f}%"
    )

    # Use the same row and feature order as the Random Forest prediction.
    selected_X = X.iloc[[selected_window_index]].copy()

    # --------------------------------------------
    # Individual prediction explanation
    # --------------------------------------------

    shap_df, shap_error = get_shap_contributions(
        rf,
        selected_X,
        selected_label
    )

    if shap_df is not None and not shap_df.empty:

        shap_df["Feature"] = shap_df["Feature"].map(
            friendly_feature_name
        )

        st.markdown("### Feature Contributions")

        st.write(
            "Features with larger absolute SHAP values had a stronger "
            "influence on this model prediction. Positive and negative "
            "values represent different directions of contribution "
            "for the selected class."
        )

        chart_df = shap_df.head(10).sort_values(
            "SHAP contribution"
        )

        st.bar_chart(
            chart_df.set_index("Feature")[
                ["SHAP contribution"]
            ]
        )

        st.markdown("### SHAP Explanation Table")

        st.dataframe(
            shap_df[
                [
                    "Feature",
                    "Input value",
                    "SHAP contribution"
                ]
            ].round(4),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Explanation in Simple Words")

        for _, contribution in shap_df.head(4).iterrows():

            value = float(contribution["SHAP contribution"])
            feature = contribution["Feature"]

            if value > 0:
                effect = (
                    f"contributed positively towards the model's "
                    f"**{selected_label}** output"
                )
            elif value < 0:
                effect = (
                    f"contributed negatively towards the model's "
                    f"**{selected_label}** output"
                )
            else:
                effect = (
                    "had approximately zero contribution "
                    "to this class output"
                )

            st.write(
                f"- **{feature}** {effect} "
                f"(SHAP value: {value:.4f})."
            )

        st.caption(
            "SHAP values describe the trained model's output relative to "
            "its baseline. They are not causal explanations."
        )

    else:
        st.warning(shap_error)

    # --------------------------------------------
    # Overall feature importance
    # --------------------------------------------

    st.markdown("### Overall Feature Importance")

    with st.expander(
        "Show overall SHAP feature importance",
        expanded=False
    ):

        try:
            import shap

            # Limit the sample size to reduce explanation time.
            sample_size = min(len(X), 100)
            X_sample = X.iloc[:sample_size].copy()

            explainer = shap.TreeExplainer(rf)
            sample_explanation = explainer(X_sample)

            sample_values = np.asarray(
                sample_explanation.values
            )

            if sample_values.ndim == 3:
                # Average absolute contributions over samples and classes.
                global_importance = np.abs(
                    sample_values
                ).mean(axis=(0, 2))

            elif sample_values.ndim == 2:
                global_importance = np.abs(
                    sample_values
                ).mean(axis=0)

            else:
                global_importance = None

            if (
                global_importance is not None
                and len(global_importance) == len(X.columns)
            ):

                importance_df = pd.DataFrame({
                    "Feature": [
                        friendly_feature_name(name)
                        for name in X.columns
                    ],
                    "Mean Absolute SHAP Value": global_importance
                }).sort_values(
                    "Mean Absolute SHAP Value",
                    ascending=False
                ).head(10)

                st.write(
                    "This chart shows which input features have the "
                    "strongest average influence across the sampled "
                    "prediction windows."
                )

                st.bar_chart(
                    importance_df.set_index("Feature")
                )

                st.dataframe(
                    importance_df.round(4),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.warning(
                    "The installed SHAP version returned an unexpected "
                    "shape for the overall feature importance."
                )

        except Exception as exc:
            st.warning(
                "Unable to calculate overall SHAP importance: "
                f"{exc}"
            )

else:
    st.info("No prediction windows are available to explain.")

# ============================================================
# 17. CREATE FINAL ANNOTATED VIDEO
# ============================================================

st.subheader("8. Final Annotated Video")

frame_predictions = {}

for _, row in prediction_df.iterrows():

    confidence_value = float(
        row["prediction_confidence"]
    )

    for frame in range(
        int(row["start_frame"]),
        int(row["end_frame"]) + 1
    ):

        key = (
            frame,
            int(row["track_id"])
        )

        old = frame_predictions.get(key)

        if (
            old is None
            or confidence_value > old["confidence"]
        ):

            frame_predictions[key] = {
                "behavior": row["behavior_prediction"],
                "risk": row["risk_level"],
                "alert": row["alert"],
                "confidence": confidence_value
            }

tracks_by_frame = defaultdict(list)

for _, row in df.iterrows():
    tracks_by_frame[int(row["frame"])].append(row)

cap = cv2.VideoCapture(tracking_video)

writer = cv2.VideoWriter(
    final_raw_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    st.error("Unable to create the final annotated video.")
    st.stop()

frame_number = 0
progress = st.progress(0)
status = st.empty()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    for track in tracks_by_frame.get(frame_number, []):

        # Behaviour and risk predictions are for animal classes.
        if int(track["class_id"]) == 0:
            continue

        key = (
            frame_number,
            int(track["track_id"])
        )

        prediction = frame_predictions.get(key)

        if prediction is None:
            continue

        x1 = int(track["x1"])
        y1 = int(track["y1"])
        x2 = int(track["x2"])
        y2 = int(track["y2"])

        label = (
            f"ID {int(track['track_id'])} | "
            f"{prediction['behavior']} | "
            f"{prediction['risk']} | "
            f"{prediction['confidence'] * 100:.0f}%"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            prediction["alert"],
            (x1, min(height - 15, y2 + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    writer.write(frame)

    progress.progress(
        min(frame_number / max(total_frames, 1), 1.0)
    )

    status.write(
        f"Creating final video: {frame_number} / {total_frames}"
    )

cap.release()
writer.release()

progress.empty()
status.empty()

st.success("Final annotated video created.")

# ============================================================
# 18. VIDEO CONVERSION
# ============================================================

if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):

    if height > width:
        scale_filter = "scale=-2:1080"
    else:
        scale_filter = "scale=1920:-2"

    ffmpeg_command = [
        FFMPEG_PATH,
        "-y",
        "-i", final_raw_video,
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "1",
        "-an",
        final_video
    ]

    try:
        process = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0 or not os.path.exists(final_video):
            st.warning(
                "FFmpeg conversion failed. Using the raw annotated video."
            )
            final_video = final_raw_video

    except Exception as exc:
        st.warning(
            f"Video conversion failed. Using the raw video. Details: {exc}"
        )
        final_video = final_raw_video

else:
    final_video = final_raw_video

# ============================================================
# 19. DISPLAY AND DOWNLOAD RESULTS
# ============================================================

st.subheader("9. Final Results")

st.write("### Final Video")
st.video(final_video)

with open(final_video, "rb") as video_file:
    st.download_button(
        "Download Final Video",
        data=video_file,
        file_name="animal_behavior_risk_final.mp4",
        mime="video/mp4",
        use_container_width=True
    )

with open(results_csv, "rb") as csv_file:
    st.download_button(
        "Download Results CSV",
        data=csv_file,
        file_name="animal_behavior_risk_results.csv",
        mime="text/csv",
        use_container_width=True
    )

st.success("🎉 Complete analysis finished successfully.")

# ============================================================
# 20. CLEAN UP TEMPORARY UPLOAD
# ============================================================

if (
    temporary_input_path
    and os.path.exists(temporary_input_path)
):
    try:
        os.remove(temporary_input_path)
    except OSError:
        pass