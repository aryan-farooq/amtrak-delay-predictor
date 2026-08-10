import json
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Wolverine ETA Predictor",
    page_icon="🚆",
    layout="centered"
)


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).parent


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    point_model = joblib.load(
        BASE_DIR / "point_model.pkl"
    )

    lower_model = joblib.load(
        BASE_DIR / "lower_model.pkl"
    )

    upper_model = joblib.load(
        BASE_DIR / "upper_model.pkl"
    )

    return (
        point_model,
        lower_model,
        upper_model
    )


point_model, lower_model, upper_model = load_models()


# ==========================================
# LOAD MODEL INFORMATION
# ==========================================

with open(
    BASE_DIR / "model_info.json",
    "r"
) as file:

    model_info = json.load(file)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_delay(
    det_delay,
    der_delay
):

    input_data = pd.DataFrame({
        "det_delay": [det_delay],
        "der_delay": [der_delay]
    })

    prediction = point_model.predict(
        input_data
    )[0]

    lower = lower_model.predict(
        input_data
    )[0]

    upper = upper_model.predict(
        input_data
    )[0]

    return (
        prediction,
        lower,
        upper
    )


# ==========================================
# TIME FUNCTION
# ==========================================

def add_delay_to_time(
    scheduled_time,
    delay_minutes
):

    scheduled_datetime = datetime.combine(
        datetime.today(),
        scheduled_time
    )

    # Round prediction to nearest whole minute
    rounded_delay = round(float(delay_minutes))

    predicted_datetime = (
        scheduled_datetime
        + timedelta(minutes=rounded_delay)
    )

    return predicted_datetime.strftime(
        "%-I:%M %p"
    )


# ==========================================
# HEADER
# ==========================================

st.title("🚆 Wolverine ETA Predictor")

st.markdown(
    """
    **Train 351 · Detroit → Dearborn → Ann Arbor**

    Machine-learning ETA estimates using observed
    upstream train delays.
    """
)

st.caption(
    "Independent data science project. "
    "Not affiliated with Amtrak."
)

st.divider()


# ==========================================
# INPUT SECTION
# ==========================================

st.markdown(
    """
    ### Route

    **Detroit**  →  **Dearborn**  →  **Ann Arbor**
    
    Observed　　　Observed　　　　Predicted
    """
)

st.subheader("Current train status")

col1, col2 = st.columns(2)

with col1:

    det_delay = st.number_input(
        "Detroit departure delay",
        min_value=-30,
        max_value=300,
        value=10,
        step=1,
        help="Enter minutes late. Negative values mean early."
    )


with col2:

    der_delay = st.number_input(
        "Dearborn departure delay",
        min_value=-30,
        max_value=300,
        value=12,
        step=1,
        help="Enter minutes late. Negative values mean early."
    )


scheduled_arrival = st.time_input(
    "Scheduled Ann Arbor arrival",
    value=datetime.strptime(
        "7:14 AM",
        "%I:%M %p"
    ).time()
)


# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button(
    "Predict Ann Arbor arrival",
    type="primary",
    use_container_width=True
):

    prediction, lower, upper = predict_delay(
        det_delay,
        der_delay
    )

    predicted_time = add_delay_to_time(
        scheduled_arrival,
        prediction
    )

    lower_time = add_delay_to_time(
        scheduled_arrival,
        lower
    )

    upper_time = add_delay_to_time(
        scheduled_arrival,
        upper
    )


    # ======================================
    # MAIN RESULT
    # ======================================

   st.divider()

st.subheader("Prediction")

with st.container(border=True):

    st.caption("PREDICTED ANN ARBOR ARRIVAL")

    st.markdown(
        f"# {predicted_time}"
    )

    st.markdown(
        f"**{prediction:.1f} minutes late**"
    )

    st.markdown(
        f"Likely arrival: "
        f"**{lower_time} – {upper_time}**"
    )

    st.caption(
        "Approximately 80% historical prediction interval"
    )


    # ======================================
    # CURRENT STATUS
    # ======================================

    st.subheader("Observed train status")

    status1, status2 = st.columns(2)

    status1.metric(
        "Detroit",
        f"{det_delay} min"
    )

    status2.metric(
        "Dearborn",
        f"{der_delay} min"
    )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()

st.subheader("Historical model performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Mean absolute error",
    f"{model_info['historical_test_mae_minutes']:.1f} min"
)

metric2.metric(
    "Within 5 min",
    f"{model_info['within_5_minutes_percent']:.1f}%"
)

metric3.metric(
    "Interval coverage",
    f"{model_info['interval_coverage_percent']:.1f}%"
)


with st.expander(
    "How does this model work?"
):

    st.write(
        """
        The model uses historical Amtrak Wolverine
        performance to learn how delays propagate
        from Detroit and Dearborn to Ann Arbor.

        The primary prediction is generated with
        Gradient Boosting. Separate quantile models
        estimate lower and upper arrival-delay bounds.

        Historical testing showed that upstream train
        performance was substantially more useful than
        calendar information such as weekday or month.
        """
    )


with st.expander(
    "Important limitation"
):

    st.write(
        """
        Rare operational events can cause large delays
        after the train leaves Dearborn. Historical
        testing found that these events are difficult
        to predict using Detroit and Dearborn delay
        information alone.

        This tool should therefore be treated as an
        experimental estimate rather than an official
        Amtrak arrival time.
        """
    )
