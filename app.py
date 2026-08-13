import json
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import pandas as pd
import streamlit as st

from live_data import get_train_351_status

# Optional debug function
try:
    from live_data import debug_train_ids
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


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


try:

    point_model, lower_model, upper_model = load_models()

except Exception as exc:

    st.error(
        "The prediction models could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ==========================================
# LOAD MODEL INFORMATION
# ==========================================

try:

    with open(
        BASE_DIR / "model_info.json",
        "r"
    ) as file:

        model_info = json.load(file)

except Exception as exc:

    st.error(
        "model_info.json could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_delay(
    det_delay,
    der_delay
):

    input_data = pd.DataFrame({
        "det_delay": [float(det_delay)],
        "der_delay": [float(der_delay)]
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
        float(prediction),
        float(lower),
        float(upper)
    )


# ==========================================
# TIME FUNCTIONS
# ==========================================

def format_time(dt):

    # Streamlit runs on Linux, where %-I works.
    # The fallback makes the function safer elsewhere.
    try:
        return dt.strftime("%-I:%M %p")

    except ValueError:
        return dt.strftime("%I:%M %p").lstrip("0")


def add_delay_to_time(
    scheduled_time,
    delay_minutes
):

    scheduled_datetime = datetime.combine(
        datetime.today(),
        scheduled_time
    )

    # Passenger-facing ETA rounded to nearest minute
    rounded_delay = round(
        float(delay_minutes)
    )

    predicted_datetime = (
        scheduled_datetime
        + timedelta(
            minutes=rounded_delay
        )
    )

    return format_time(
        predicted_datetime
    )


# ==========================================
# HEADER
# ==========================================

st.title(
    "🚆 Wolverine ETA Predictor"
)

st.markdown(
    """
    **Train 351 · Detroit → Dearborn → Ann Arbor**

    Machine-learning ETA estimates using observed
    upstream train delays.
    """
)

st.caption(
    "Independent data science project. "
    "Not affiliated with or operated by Amtrak."
)

st.divider()


# ==========================================
# ROUTE
# ==========================================

st.markdown(
    """
    ### Route

    **Detroit** → **Dearborn** → **Ann Arbor**

    Observed → Observed → Predicted
    """
)

st.divider()


# ==========================================
# CURRENT TRAIN STATUS
# ==========================================

st.subheader(
    "Current train status"
)

data_source = st.radio(
    "Data source",
    [
        "Live Amtrak data",
        "Manual entry"
    ],
    horizontal=True
)


det_delay = None
der_delay = None


# ==========================================
# LIVE MODE
# ==========================================

if data_source == "Live Amtrak data":

    with st.spinner(
        "Checking realtime Train 351 data..."
    ):

        live_status = (
            get_train_351_status()
        )


    # --------------------------------------
    # Complete live-feed failure
    # --------------------------------------

    if live_status.get("error"):

        st.warning(
            "Live train data could not be loaded."
        )

        st.caption(
            live_status["error"]
        )

        st.info(
            "You can switch to **Manual entry** "
            "to use the predictor."
        )


    # --------------------------------------
    # Feed is too old
    # --------------------------------------

    elif (
        live_status.get(
            "feed_age_minutes"
        ) is not None
        and
        live_status[
            "feed_age_minutes"
        ] > 15
    ):

        st.warning(
            "⚠️ The realtime feed appears "
            "to be stale."
        )

        st.write(
            "Feed age: "
            f"**{live_status['feed_age_minutes']} "
            "minutes**"
        )

        st.info(
            "Switch to **Manual entry** "
            "for a prediction."
        )


    # --------------------------------------
    # Train not in feed
    # --------------------------------------

    elif not live_status.get(
        "train_found",
        False
    ):

        st.info(
            "Train 351 is not currently active in the "
            "realtime feed. Train 351 normally operates "
            "in the morning. Try Live mode while the train "
            "is running, or use Manual entry to test the model."
        )


    # --------------------------------------
    # Live train exists
    # --------------------------------------

    else:

        st.success(
            "● Live Train 351 data connected"
        )

        feed_age = live_status.get(
            "feed_age_minutes"
        )

        if feed_age is not None:

            st.caption(
                f"Realtime feed updated "
                f"{feed_age:.1f} minutes ago."
            )


        det_delay = live_status.get(
            "det_delay"
        )

        der_delay = live_status.get(
            "der_delay"
        )


        live_col1, live_col2 = (
            st.columns(2)
        )


        with live_col1:

            if det_delay is None:

                st.metric(
                    "Detroit",
                    "Waiting..."
                )

                st.caption(
                    "No observed departure yet."
                )

            else:

                st.metric(
                    "Detroit departure",
                    f"{det_delay:.0f} min late"
                )


        with live_col2:

            if der_delay is None:

                st.metric(
                    "Dearborn",
                    "Waiting..."
                )

                st.caption(
                    "No observed departure yet."
                )

            else:

                st.metric(
                    "Dearborn departure",
                    f"{der_delay:.0f} min late"
                )


        # Tell user why button might be disabled

        if det_delay is None:

            st.info(
                "Waiting for Train 351 to depart "
                "Detroit before live prediction "
                "data becomes available."
            )

        elif der_delay is None:

            st.info(
                "Detroit data is available. "
                "Waiting for the Dearborn departure "
                "before making the full prediction."
            )


# ==========================================
# MANUAL MODE
# ==========================================

else:

    manual_col1, manual_col2 = (
        st.columns(2)
    )


    with manual_col1:

        det_delay = st.number_input(
            "Detroit departure delay",
            min_value=-30,
            max_value=300,
            value=0,
            step=1,
            help=(
                "Enter minutes late. "
                "Use a negative number "
                "if the train departed early."
            )
        )


    with manual_col2:

        der_delay = st.number_input(
            "Dearborn departure delay",
            min_value=-30,
            max_value=300,
            value=0,
            step=1,
            help=(
                "Enter minutes late. "
                "Use a negative number "
                "if the train departed early."
            )
        )


# ==========================================
# SCHEDULED ARRIVAL
# ==========================================

scheduled_arrival = st.time_input(
    "Scheduled Ann Arbor arrival",
    value=datetime.strptime(
        "7:14 AM",
        "%I:%M %p"
    ).time()
)


# ==========================================
# DETERMINE WHETHER PREDICTION CAN RUN
# ==========================================

prediction_ready = (
    det_delay is not None
    and
    der_delay is not None
)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    "Predict Ann Arbor arrival",
    type="primary",
    use_container_width=True,
    disabled=not prediction_ready
):

    # --------------------------------------
    # RUN MODELS
    # --------------------------------------

    prediction, lower, upper = (
        predict_delay(
            det_delay,
            der_delay
        )
    )


    # Ensure bounds display logically
    if lower > upper:

        lower, upper = upper, lower


    # --------------------------------------
    # CONVERT DELAYS TO CLOCK TIMES
    # --------------------------------------

    predicted_time = (
        add_delay_to_time(
            scheduled_arrival,
            prediction
        )
    )

    lower_time = (
        add_delay_to_time(
            scheduled_arrival,
            lower
        )
    )

    upper_time = (
        add_delay_to_time(
            scheduled_arrival,
            upper
        )
    )


    # ======================================
    # RESULT
    # ======================================

    st.divider()

    st.subheader(
        "Prediction"
    )


    with st.container(
        border=True
    ):

        st.caption(
            "PREDICTED ANN ARBOR ARRIVAL"
        )

        st.markdown(
            f"# {predicted_time}"
        )


        if prediction >= 0:

            st.markdown(
                f"**{prediction:.1f} "
                "minutes late**"
            )

        else:

            st.markdown(
                f"**{abs(prediction):.1f} "
                "minutes early**"
            )


        st.markdown(
            f"Likely arrival: "
            f"**{lower_time} – "
            f"{upper_time}**"
        )

        st.caption(
            "Approximately 80% historical "
            "prediction interval."
        )


    # ======================================
    # OBSERVED STATUS
    # ======================================

    st.subheader(
        "Observed train status"
    )

    status1, status2 = (
        st.columns(2)
    )


    if det_delay >= 0:

        det_text = (
            f"{det_delay:.0f} min late"
        )

    else:

        det_text = (
            f"{abs(det_delay):.0f} min early"
        )


    if der_delay >= 0:

        der_text = (
            f"{der_delay:.0f} min late"
        )

    else:

        der_text = (
            f"{abs(der_delay):.0f} min early"
        )


    status1.metric(
        "Detroit",
        det_text
    )

    status2.metric(
        "Dearborn",
        der_text
    )


# ==========================================
# HISTORICAL PERFORMANCE
# ==========================================

st.divider()

st.subheader(
    "Historical model performance"
)

metric1, metric2, metric3 = (
    st.columns(3)
)


metric1.metric(
    "Mean absolute error",
    (
        f"{model_info[
            'historical_test_mae_minutes'
        ]:.1f} min"
    )
)


metric2.metric(
    "Within 5 min",
    (
        f"{model_info[
            'within_5_minutes_percent'
        ]:.1f}%"
    )
)


metric3.metric(
    "Interval coverage",
    (
        f"{model_info[
            'interval_coverage_percent'
        ]:.1f}%"
    )
)


# ==========================================
# MODEL DETAILS
# ==========================================

with st.expander(
    "How does this model work?"
):

    st.write(
        """
        The model uses historical Amtrak Wolverine
        performance to learn how delays propagate
        from Detroit and Dearborn to Ann Arbor.

        The primary prediction uses Gradient
        Boosting. Separate quantile-regression
        models estimate lower and upper bounds
        for the arrival-time prediction interval.

        Testing showed that observed upstream
        train delays were substantially more
        useful than calendar variables such as
        weekday or month.
        """
    )


# ==========================================
# LIMITATIONS
# ==========================================

with st.expander(
    "Important limitations"
):

    st.write(
        """
        Rare operational events can cause large
        amounts of delay after the train leaves
        Dearborn.

        Historical testing found that these rare
        events are difficult to predict using
        Detroit and Dearborn delay information
        alone.

        Weather produced only a small improvement
        during testing and did not explain the
        largest unexpected delay events.

        This application should therefore be
        treated as an experimental estimate,
        not an official Amtrak arrival time.
        """
    )


# ==========================================
# LIVE FEED DEBUG
# ==========================================

if DEBUG_AVAILABLE:

    with st.expander(
        "Live feed debug"
    ):

        try:

            debug_data = (
                debug_train_ids()
            )

            st.write(
                "**Train 351 TripUpdate IDs**"
            )

            st.write(
                debug_data.get(
                    "trip_updates",
                    []
                )
            )

            st.write(
                "**Train 351 Vehicle IDs**"
            )

            st.write(
                debug_data.get(
                    "vehicles",
                    []
                )
            )

        except Exception as exc:

            st.write(
                "Debug data could not "
                "be loaded."
            )

            st.code(
                str(exc)
            )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Built as an independent data science "
    "project using historical Amtrak performance "
    "data and machine-learning models."
)
