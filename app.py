import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import os
from datetime import datetime

# ── Model & Data ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")  # medium = better accuracy

@st.cache_data
def load_data():
    return pd.read_csv("food_calories.csv")

COCO_TO_FOOD = {
    "pizza"    : "pizza",
    "apple"    : "apple",
    "banana"   : "banana",
    "orange"   : "orange",
    "sandwich" : "sandwich",
    "hot dog"  : "hot dog",
    "cake"     : "cake",
    "donut"    : "donut",
    "carrot"   : "carrot",
    "broccoli" : "broccoli",
    "wine glass": "wine",
    "cup"      : "coffee",
    "bowl"     : "soup",
    "bottle"   : "juice",
}

HEALTHY_SWAPS = {
    "burger"  : "Try grilled chicken wrap — saves ~200 kcal",
    "pizza"   : "Try whole wheat veggie pizza — saves ~150 kcal",
    "donut"   : "Try a banana instead — saves ~180 kcal",
    "cake"    : "Try Greek yogurt with berries — saves ~250 kcal",
    "hot dog" : "Try a turkey sandwich — saves ~100 kcal",
    "wine"    : "Try sparkling water with lemon — saves ~125 kcal",
}

# ── Image Enhancement ─────────────────────────────────────────────
def enhance_image(img_array):
    kernel   = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp    = cv2.filter2D(img_array, -1, kernel)
    lab      = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l        = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    return enhanced

# ── Portion Estimator ─────────────────────────────────────────────
def estimate_portion(box, img_w, img_h):
    x1, y1, x2, y2 = box
    ratio = ((x2-x1) * (y2-y1)) / (img_w * img_h)
    if ratio < 0.10:
        return 0.5, "Small"
    elif ratio < 0.30:
        return 1.0, "Medium"
    else:
        return 1.5, "Large"

# ── UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="🍽️ Food Calorie Estimator", layout="wide")
st.title("🍽️ Food Calorie Estimator")
st.write("Upload a food image — AI detects every item, estimates portions, calculates total calories.")

# Sidebar — Personal Profile
st.sidebar.header("👤 Your Profile")
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)
height = st.sidebar.number_input("Height (cm)", 140, 210, 170)
age    = st.sidebar.number_input("Age", 10, 80, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
conf_threshold = st.sidebar.slider("Detection Confidence", 0.10, 0.90, 0.20, 0.05)

# BMR Calculation
if gender == "Male":
    bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
else:
    bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)
daily_goal = round(bmr * 1.55)
st.sidebar.metric("🎯 Your Daily Calorie Goal", f"{daily_goal} kcal")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

model = load_model()
df    = load_data()

# Upload
uploaded_files = st.file_uploader(
    "📤 Upload food image(s)",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if uploaded_files:
    all_foods      = []
    grand_total    = 0
    grand_protein  = 0
    grand_carbs    = 0
    grand_fat      = 0

    for uploaded_file in uploaded_files:
        image     = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_array = enhance_image(img_array)
        img_h, img_w = img_array.shape[:2]

        with st.spinner(f"🔍 Detecting food in {uploaded_file.name}..."):
            results = model(img_array, conf=conf_threshold, iou=0.4, agnostic_nms=True)
            result  = results[0]

        labels = [result.names[int(c)] for c in result.boxes.cls]
        confs  = [float(x) for x in result.boxes.conf]
        boxes  = result.boxes.xyxy.tolist()

        food_items = []
        for label, conf, box in zip(labels, confs, boxes):
            if label in COCO_TO_FOOD:
                food  = COCO_TO_FOOD[label]
                match = df[df["food"] == food]
                if not match.empty:
                    row            = match.iloc[0]
                    pf, plabel     = estimate_portion(box, img_w, img_h)
                    cal  = round(float(row["calories_per_serving"]) * pf, 1)
                    pro  = round(float(row["protein_g"])            * pf, 1)
                    carb = round(float(row["carbs_g"])              * pf, 1)
                    fat  = round(float(row["fat_g"])                * pf, 1)
                    food_items.append({
                        "Food"           : food.title(),
                        "Portion"        : plabel,
                        "Confidence"     : f"{conf:.0%}",
                        "Calories (kcal)": cal,
                        "Protein (g)"    : pro,
                        "Carbs (g)"      : carb,
                        "Fat (g)"        : fat,
                    })
                    grand_total   += cal
                    grand_protein += pro
                    grand_carbs   += carb
                    grand_fat     += fat
                    all_foods.append(food)

        # Display per image
        st.markdown(f"### 📸 {uploaded_file.name}")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            annotated     = result.plot(line_width=2)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detection Result", use_column_width=True)
        with col2:
            if food_items:
                if len(food_items) > 1:
                    st.info(f"🍱 Mixed plate — {len(food_items)} items detected!")
                st.dataframe(pd.DataFrame(food_items), use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No food items recognized in this image.")

    # ── Grand Total ───────────────────────────────────────────────
    if all_foods:
        st.markdown("---")
        st.markdown("## 📊 Total Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔥 Total Calories", f"{grand_total:.0f} kcal")
        m2.metric("💪 Protein",        f"{grand_protein:.1f}g")
        m3.metric("🍞 Carbs",          f"{grand_carbs:.1f}g")
        m4.metric("🧈 Fat",            f"{grand_fat:.1f}g")

        # Progress vs daily goal
        progress = min(grand_total / daily_goal, 1.0)
        st.progress(progress)
        st.caption(f"{grand_total:.0f} / {daily_goal} kcal — {progress*100:.0f}% of your daily goal")

        if grand_total < 300:
            st.success("💚 Light meal — well within your goal!")
        elif grand_total < 600:
            st.warning("💛 Moderate meal — you're on track!")
        else:
            st.error("🔴 High calorie meal — consider a lighter next meal!")

        # Healthy swaps
        st.markdown("### 💡 Healthier Swap Suggestions")
        shown = False
        for food in set(all_foods):
            if food in HEALTHY_SWAPS:
                st.info(f"**{food.title()}** → {HEALTHY_SWAPS[food]}")
                shown = True
        if not shown:
            st.success("✅ Great choices! Your meal looks balanced.")

        # Download report
        report_df = pd.DataFrame([{
            "Food"           : f["Food"],
            "Portion"        : f["Portion"],
            "Calories (kcal)": f["Calories (kcal)"],
            "Protein (g)"    : f["Protein (g)"],
            "Carbs (g)"      : f["Carbs (g)"],
            "Fat (g)"        : f["Fat (g)"],
        } for uploaded_file in uploaded_files
          for f in ([] if not uploaded_files else
          [{
            "Food"           : food.title(),
            "Portion"        : "-",
            "Calories (kcal)": 0,
            "Protein (g)"    : 0,
            "Carbs (g)"      : 0,
            "Fat (g)"        : 0,
          }])])

        st.download_button(
            label    = "📥 Download Meal Report",
            data     = pd.DataFrame([{
                "Food": f, "Total Calories": grand_total,
                "Protein": grand_protein, "Carbs": grand_carbs, "Fat": grand_fat
            } for f in all_foods]).to_csv(index=False),
            file_name= f"meal_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime     = "text/csv"
        )

        # Save to history
        st.session_state.history.append({
            "Time"    : datetime.now().strftime("%H:%M"),
            "Calories": grand_total,
            "Foods"   : ", ".join([f.title() for f in all_foods])
        })

    # ── Meal History Chart ────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 📈 Today's Calorie History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("Time")["Calories"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)