import io
import zipfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from google.transit import gtfs_realtime_pb2


# ==========================================
# DATA SOURCES
# ==========================================

REALTIME_URL = (
    "https://asm-backend.transitdocs.com/gtfs/amtrak"
)

STATIC_GTFS_URL = (
    "https://content.amtrak.com/content/gtfs/GTFS.zip"
)


# ==========================================
# GET AMTRAK STATION IDS
# ==========================================

@st.cache_data(ttl=86400)
def get_station_ids():
    """
    Download Amtrak static GTFS and determine
    the stop IDs for Detroit, Dearborn and
    Ann Arbor.

    Cached for 24 hours.
    """

    response = requests.get(
        STATIC_GTFS_URL,
        timeout=30
    )

    response.raise_for_status()

    gtfs_zip = zipfile.ZipFile(
        io.BytesIO(response.content)
    )

    stops = pd.read_csv(
        gtfs_zip.open("stops.txt"),
        dtype=str
    )

    # Clean up possible missing values
    stops = stops.fillna("")

    station_codes = {
        "DET": "Detroit",
        "DER": "Dearborn",
        "ARB": "Ann Arbor"
    }

    station_ids = {}

    for code, station_name in station_codes.items():

        ids = set()

        # Best option: GTFS stop_code
        if "stop_code" in stops.columns:

            matches = stops[
                stops["stop_code"]
                .str.upper()
                .eq(code)
            ]

            ids.update(
                matches["stop_id"].tolist()
            )

        # Fallback: stop_id itself
        if not ids:

            matches = stops[
                stops["stop_id"]
                .str.upper()
                .eq(code)
            ]

            ids.update(
                matches["stop_id"].tolist()
            )

        # Final fallback: station name
        if not ids:

            matches = stops[
                stops["stop_name"]
                .str.contains(
                    station_name,
                    case=False,
                    na=False
                )
            ]

            ids.update(
                matches["stop_id"].tolist()
            )

        station_ids[code] = ids

    return station_ids


# ==========================================
# READ DEPARTURE DELAY
# ==========================================

def get_departure_delay(stop_update):
    """
    Return departure delay in minutes.

    Only use the value after the scheduled/
    predicted departure time has passed so
    we don't treat a future prediction as an
    observed station departure.
    """

    departure = stop_update.departure

    if not departure.HasField("delay"):
        return None

    if not departure.HasField("time"):
        return None

    current_timestamp = datetime.now(
        timezone.utc
    ).timestamp()

    # Station has not been departed yet
    if departure.time > current_timestamp:
        return None

    return round(
        departure.delay / 60,
        1
    )


# ==========================================
# LIVE TRAIN 351 STATUS
# ==========================================

@st.cache_data(ttl=60, show_spinner=False)
def get_train_351_status():
    """
    Fetch live Amtrak GTFS-Realtime data
    and return Train 351's observed Detroit
    and Dearborn departure delays.
    """

    result = {
        "train_found": False,
        "feed_age_minutes": None,
        "det_delay": None,
        "der_delay": None,
        "arb_delay": None,
        "error": None
    }

    try:

        # ----------------------------------
        # Download realtime feed
        # ----------------------------------

        response = requests.get(
            REALTIME_URL,
            timeout=20
        )

        response.raise_for_status()

        feed = gtfs_realtime_pb2.FeedMessage()

        feed.ParseFromString(
            response.content
        )


        # ----------------------------------
        # Calculate feed age
        # ----------------------------------

        if feed.header.timestamp:

            feed_time = datetime.fromtimestamp(
                feed.header.timestamp,
                tz=timezone.utc
            )

            now = datetime.now(timezone.utc)

            age = now - feed_time

            result["feed_age_minutes"] = round(
                age.total_seconds() / 60,
                1
            )


        # ----------------------------------
        # Today's date in Michigan
        # ----------------------------------

        michigan_now = datetime.now(
            ZoneInfo("America/Detroit")
        )

        today_trip_prefix = (
            michigan_now.strftime("%Y-%m-%d")
        )

        today_gtfs_date = (
            michigan_now.strftime("%Y%m%d")
        )


        # ----------------------------------
        # Find today's Train 351
        # ----------------------------------

        train_update = None

        for entity in feed.entity:

            if not entity.HasField(
                "trip_update"
            ):
                continue

            update = entity.trip_update

            trip_id = update.trip.trip_id

            start_date = update.trip.start_date

            is_train_351 = (
                trip_id.endswith("_AMTK_351")
                or "_351" in trip_id
            )

            is_today = (
                trip_id.startswith(
                    today_trip_prefix
                )
                or start_date == today_gtfs_date
            )

            if is_train_351 and is_today:

                train_update = update
                break


        if train_update is None:

            return result


        result["train_found"] = True


        # ----------------------------------
        # Determine station GTFS IDs
        # ----------------------------------

        station_ids = get_station_ids()

        det_ids = station_ids["DET"]
        der_ids = station_ids["DER"]
        arb_ids = station_ids["ARB"]


        # ----------------------------------
        # Read station updates
        # ----------------------------------

        for stop in train_update.stop_time_update:

            stop_id = stop.stop_id

            delay = get_departure_delay(
                stop
            )

            if stop_id in det_ids:

                result["det_delay"] = delay

            elif stop_id in der_ids:

                result["der_delay"] = delay

            elif stop_id in arb_ids:

                result["arb_delay"] = delay


        return result


    except Exception as exc:

        result["error"] = str(exc)

        return result
