import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st


APP_VERSION = "1.0.0"

WAITROSE_CLIENT = "Waitrose Supermarkets"
RAPID_DELIVERY = "Rapid Delivery"

BASE_SOURCE_MAP = {
    "ORDER": "order_internal_id",
    "CLIENT": "client_name",
    "VISIT": "internal_id",
    "SITE": "site_internal_id",
    "PREMISES": "site_name",
    "SITE CODE": "site_code",
    "DATE OF VISIT": "date_of_visit_local",
    "TIME OF VISIT": "time_of_visit_local",
    "RESULT": "primary_result",
}


@dataclass(frozen=True)
class QuestionSpec:
    output_header: str
    aliases: tuple[str, ...]


# These are the non-upload questions used by the Waitrose Deliveroo audit.
# Aliases allow old and new questionnaire wording to be combined into one
# stable report column. If more than one alias exists in an export, the first
# non-blank answer in alias order is used for each audit.
QUESTION_SPECS = (
    QuestionSpec(
        "Please describe the doorstep transaction:",
        (
            "Please describe the doorstep transaction:",
            "Please describe the delivery transaction:",
        ),
    ),
    QuestionSpec(
        "Please enter the order number:",
        (
            "Please enter the order number:",
            "Please enter your order number:",
            "Please enter the 11-digit order number:",
        ),
    ),
    QuestionSpec(
        "Please use this space to explain anything unusual about your visit or to clarify any detail of your report:",
        (
            "Please use this space to explain anything unusual about your visit or to clarify any detail of your report:",
            "Is there anything unusual you wish to explain?",
        ),
    ),
    QuestionSpec(
        "Please confirm below whether or not you were asked for ID:",
        (
            "Please confirm below whether or not you were asked for ID:",
            "Please confirm below if the rider asked for ID?",
            "Please confirm below if the courier asked for ID?",
        ),
    ),
    QuestionSpec(
        "Please confirm the name of the Waitrose store you tested:",
        (
            "Please confirm the name of the Waitrose store you tested:",
            "What is the name of the Waitrose branch that delivered to you?",
        ),
    ),
    QuestionSpec(
        "Please confirm the postcode of the Waitrose store you tested:",
        (
            "Please confirm the postcode of the Waitrose store you tested:",
            "Please confirm the postcode of the Waitrose branch you tested:",
        ),
    ),
    QuestionSpec(
        "Please give details of the alcohol that you purchased:",
        (
            "Please give details of the alcohol that you purchased:",
            "Please give details of the age-restricted product(s) that you purchased:",
            "Please give details of the age restricted product(s) purchased:",
        ),
    ),
    QuestionSpec(
        "Did the rider ask for your date of birth?",
        (
            "Did the rider ask for your date of birth?",
            "Did the driver ask for your date of birth?",
            "Did the courier ask for your date of birth?",
        ),
    ),
    QuestionSpec(
        "Did the rider ask for your ID?",
        (
            "Did the rider ask for your ID?",
            "Did the driver ask for ID?",
            "Did the courier ask for ID?",
        ),
    ),
    QuestionSpec(
        "Did the rider check your ID?",
        (
            "Did the rider check your ID?",
            "Did the driver check your ID?",
            "Did the courier check your ID?",
        ),
    ),
    QuestionSpec(
        "Anything else important to note from your interaction with the rider?",
        (
            "Anything else important to note from your interaction with the rider?",
            "Anything else important to note from your interaction with the driver?",
            "Anything else important to note from your interaction with the courier?",
        ),
    ),
    QuestionSpec(
        "Deliveroo operates a contactless delivery process: Did the rider leave the delivery on the doorstep?",
        (
            "Deliveroo operates a contactless delivery process: Did the rider leave the delivery on the doorstep?",
            "Deliveroo operates a contactless delivery process: Did the driver leave the delivery on the doorstep?",
            "Did the rider leave the delivery on the doorstep?",
        ),
    ),
    QuestionSpec(
        "If no, then did the rider hand you the delivery?",
        (
            "If no, then did the rider hand you the delivery?",
            "If no, then did the driver hand you the delivery?",
            "If no, did the rider hand you the delivery?",
        ),
    ),
    QuestionSpec(
        "I confirm that at the time this order was delivered, it was me who accepted the delivery and at no point during the transaction was there another adult present (i.e. who answered the door, could be seen in the background, etc.).",
        (
            "I confirm that at the time this order was delivered, it was me who accepted the delivery and at no point during the transaction was there another adult present (i.e. who answered the door, could be seen in the background, etc.).",
            "I confirm that I accepted the delivery and that no other adult was present during the transaction.",
        ),
    ),
)


@dataclass(frozen=True)
class ReportResult:
    data: bytes
    filename: str
    row_count: int
    reporting_month: str


def clean_text(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def normalise_text(value) -> str:
    """Normalise labels while ignoring spacing, punctuation and smart quotes."""
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    if not csv_bytes:
        raise ValueError("The audits export is empty.")

    last_error = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            frame = pd.read_csv(
                io.BytesIO(csv_bytes),
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The audits export is empty.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"The audits export could not be read: {exc}") from exc
    else:
        raise ValueError(
            "The audits export is not encoded as UTF-8 or Windows-1252."
        ) from last_error

    frame.columns = [clean_text(column) for column in frame.columns]
    return frame


def require_columns(frame: pd.DataFrame) -> None:
    required = {
        *BASE_SOURCE_MAP.values(),
        "item_to_order",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            "The audits export is missing required column(s): "
            + "; ".join(missing)
        )


def waitrose_deliveroo_mask(frame: pd.DataFrame) -> pd.Series:
    client = frame["client_name"].map(normalise_text)
    audit_type = frame["item_to_order"].map(normalise_text)
    premises = frame["site_name"].map(normalise_text)

    return (
        client.eq(normalise_text(WAITROSE_CLIENT))
        & audit_type.eq(normalise_text(RAPID_DELIVERY))
        & premises.str.contains(r"\bwaitrose\b", regex=True, na=False)
        & premises.str.contains(r"\bdeliveroo\b", regex=True, na=False)
    )


def load_waitrose_deliveroo_rows(csv_bytes: bytes) -> pd.DataFrame:
    frame = read_csv_bytes(csv_bytes)
    require_columns(frame)

    selected = frame.loc[waitrose_deliveroo_mask(frame)].copy()
    if selected.empty:
        raise ValueError(
            "No Waitrose audits fulfilled by Deliveroo were found. The app only "
            "includes rows where client_name is 'Waitrose Supermarkets', "
            "item_to_order is 'Rapid Delivery', and site_name contains both "
            "'Waitrose' and 'Deliveroo'."
        )

    blank_visits = selected["internal_id"].map(clean_text).eq("")
    if blank_visits.any():
        raise ValueError(
            f"{int(blank_visits.sum())} selected audit(s) have no internal_id."
        )

    selected["_visit_date"] = pd.to_datetime(
        selected["date_of_visit_local"], dayfirst=True, errors="coerce"
    )
    invalid_dates = selected["_visit_date"].isna()
    if invalid_dates.any():
        visits = selected.loc[invalid_dates, "internal_id"].map(clean_text)
        preview = ", ".join(visits.head(10))
        extra = "" if invalid_dates.sum() <= 10 else f" and {invalid_dates.sum() - 10} more"
        raise ValueError(
            "The local visit date could not be read for: " + preview + extra
        )

    selected["_reporting_month"] = selected["_visit_date"].dt.strftime("%Y-%m")
    return selected


def available_reporting_months(csv_bytes: bytes) -> list[str]:
    selected = load_waitrose_deliveroo_rows(csv_bytes)
    return sorted(selected["_reporting_month"].unique(), reverse=True)


def format_reporting_month(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m").strftime("%B %Y")
    except ValueError as exc:
        raise ValueError("The reporting month must use YYYY-MM format.") from exc


def resolve_question_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    normalised_columns: dict[str, list[str]] = {}
    for column in frame.columns:
        normalised_columns.setdefault(normalise_text(column), []).append(column)

    resolved: dict[str, list[str]] = {}
    missing = []
    for spec in QUESTION_SPECS:
        matches = []
        for alias in spec.aliases:
            for column in normalised_columns.get(normalise_text(alias), []):
                if column not in matches:
                    matches.append(column)
        if not matches:
            missing.append(spec.output_header)
        else:
            resolved[spec.output_header] = matches

    if missing:
        raise KeyError(
            "The audits export is missing Waitrose Deliveroo question column(s): "
            + "; ".join(missing)
        )
    return resolved


def coalesce_answers(frame: pd.DataFrame, source_columns: list[str]) -> pd.Series:
    answers = frame[source_columns].copy()
    for column in answers.columns:
        answers[column] = answers[column].map(clean_text).replace("", pd.NA)
    return answers.bfill(axis=1).iloc[:, 0].fillna("")


def parse_time(value, visit_id: str) -> tuple[str, int]:
    text = clean_text(value)
    for time_format in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"):
        try:
            parsed = datetime.strptime(text, time_format)
            return parsed.strftime("%H:%M"), parsed.hour * 60 + parsed.minute
        except ValueError:
            pass
    raise ValueError(
        f"The local visit time could not be read for {visit_id or 'a selected audit'}."
    )


def generate_report(csv_bytes: bytes, reporting_month: str) -> ReportResult:
    month_label = format_reporting_month(reporting_month)
    selected = load_waitrose_deliveroo_rows(csv_bytes)
    selected = selected.loc[selected["_reporting_month"].eq(reporting_month)].copy()
    if selected.empty:
        raise ValueError(
            f"No Waitrose Deliveroo audits were found for {month_label}."
        )

    question_sources = resolve_question_columns(selected)

    times = [
        parse_time(value, clean_text(visit_id))
        for value, visit_id in zip(
            selected["time_of_visit_local"], selected["internal_id"]
        )
    ]
    selected["_output_time"] = [value[0] for value in times]
    selected["_sort_time"] = [value[1] for value in times]
    selected.sort_values(
        ["_visit_date", "_sort_time", "internal_id"],
        kind="stable",
        inplace=True,
    )

    output = pd.DataFrame(index=selected.index)
    for output_header, source_column in BASE_SOURCE_MAP.items():
        if output_header == "DATE OF VISIT":
            output[output_header] = selected["_visit_date"].dt.strftime("%d/%m/%Y")
        elif output_header == "TIME OF VISIT":
            output[output_header] = selected["_output_time"]
        else:
            output[output_header] = selected[source_column].map(clean_text)

    for spec in QUESTION_SPECS:
        output[spec.output_header] = coalesce_answers(
            selected, question_sources[spec.output_header]
        )

    csv_text = output.to_csv(index=False, lineterminator="\r\n")
    filename = f"Deliveroo Waitrose Raw Data {month_label}.csv"
    return ReportResult(
        data=csv_text.encode("utf-8-sig"),
        filename=filename,
        row_count=len(output),
        reporting_month=reporting_month,
    )


def main() -> None:
    st.set_page_config(
        page_title="Deliveroo Waitrose Monthly Raw Data Generator",
        layout="centered",
    )
    st.title("Deliveroo Waitrose Monthly Raw Data Generator")
    st.caption(f"Version {APP_VERSION}")
    st.write(
        "Upload an audits export to create a monthly raw-data CSV containing "
        "only Waitrose Rapid Delivery audits fulfilled by Deliveroo."
    )
    st.markdown(
        """
The app will:

- exclude every retailer and delivery platform other than **Waitrose / Deliveroo**
- use local visit dates and times
- combine recognised old and new wording for equivalent Waitrose questions
- sort the report chronologically
- name the output using the month covered by the report
"""
    )

    uploaded_file = st.file_uploader(
        "Upload audits_basic_data_export.csv", type=["csv"]
    )
    if uploaded_file is None:
        return

    csv_bytes = uploaded_file.getvalue()
    try:
        months = available_reporting_months(csv_bytes)
    except (KeyError, ValueError) as exc:
        st.error(str(exc).strip("'"))
        return
    except Exception as exc:
        st.error(f"The audits export could not be checked: {exc}")
        return

    if len(months) == 1:
        reporting_month = months[0]
        st.info(f"Reporting month detected: {format_reporting_month(reporting_month)}")
    else:
        reporting_month = st.selectbox(
            "Reporting month",
            options=months,
            format_func=format_reporting_month,
            help=(
                "More than one month of Waitrose Deliveroo audits was found. "
                "Only the selected month will be included."
            ),
        )

    signature = (
        APP_VERSION,
        uploaded_file.name,
        hashlib.sha256(csv_bytes).hexdigest(),
        reporting_month,
    )
    if st.session_state.get("waitrose_input_signature") != signature:
        st.session_state.pop("waitrose_report_result", None)
        st.session_state["waitrose_input_signature"] = signature

    if st.button("Generate report", type="primary", use_container_width=True):
        try:
            with st.spinner("Generating the Waitrose Deliveroo report..."):
                st.session_state["waitrose_report_result"] = generate_report(
                    csv_bytes, reporting_month
                )
        except (KeyError, ValueError) as exc:
            st.session_state.pop("waitrose_report_result", None)
            st.error(str(exc).strip("'"))
        except Exception as exc:
            st.session_state.pop("waitrose_report_result", None)
            st.error(f"The report could not be generated: {exc}")

    result = st.session_state.get("waitrose_report_result")
    if result is None:
        return

    st.success(
        f"{format_reporting_month(result.reporting_month)} report generated "
        f"with {result.row_count} Waitrose Deliveroo audit(s)."
    )
    st.download_button(
        "Download Deliveroo Waitrose Raw Data",
        data=result.data,
        file_name=result.filename,
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
