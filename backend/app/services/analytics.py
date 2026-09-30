from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _num(value: Any, digits: int = 4) -> float | None:
    """Round a numeric value, mapping NaN/inf to None so responses stay JSON-safe."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    return pd.read_csv(path, low_memory=False) if suffix == ".csv" else pd.read_excel(path)


def preprocess_frame(frame: pd.DataFrame) -> pd.DataFrame:
    processed = frame.copy()
    for column in processed.select_dtypes(include="object").columns:
        processed[column] = processed[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    for column in date_columns(processed):
        processed[column] = pd.to_datetime(processed[column], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in numeric_columns(processed):
        if processed[column].isna().any():
            processed[column] = processed[column].fillna(processed[column].median())
    return processed.drop_duplicates().reset_index(drop=True)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return list(frame.select_dtypes(include="number").columns)


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return list(frame.select_dtypes(exclude="number").columns)


def date_columns(frame: pd.DataFrame) -> list[str]:
    dates: list[str] = []
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            dates.append(column)
            continue
        if "date" in column.lower() or "time" in column.lower():
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                dates.append(column)
    return dates


def choose_numeric(frame: pd.DataFrame, question: str = "") -> str | None:
    columns = numeric_columns(frame)
    lowered = question.lower()
    normalized = lowered.replace("_", " ")
    for column in columns:
        column_name = column.lower().replace("_", " ")
        if re.search(rf"\b{re.escape(column_name)}\b", normalized):
            return column
    for term in ("revenue", "sales", "amount", "salary", "profit", "price", "score", "quantity"):
        for column in columns:
            if term in column.lower():
                return column
    measures = [column for column in columns if not _is_identifier(column)]
    return (measures or columns or [None])[0]


def choose_category(frame: pd.DataFrame, question: str = "") -> str | None:
    columns = categorical_columns(frame)
    lowered = question.lower()
    normalized = lowered.replace("_", " ")
    for column in columns:
        column_name = column.lower().replace("_", " ")
        if re.search(rf"\b{re.escape(column_name)}\b", normalized):
            return column
    for term in ("region", "department", "category", "product", "segment", "gender", "country", "type"):
        for column in columns:
            if term in column.lower():
                return column
    dates = set(date_columns(frame))
    groupable = [column for column in columns if column not in dates and not _is_identifier(column) and 1 < frame[column].nunique(dropna=True) <= 50]
    return (groupable or columns or [None])[0]


def _is_identifier(column: str) -> bool:
    name = column.lower().replace("_", "").replace(" ", "")
    return name == "id" or name.endswith("id") or name.endswith("no") or name.endswith("number")


def profile_frame(frame: pd.DataFrame, file_size: int = 0) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("The dataset is empty.")
    missing = int(frame.isna().sum().sum())
    cells = max(1, frame.shape[0] * frame.shape[1])
    duplicates = int(frame.duplicated().sum())
    completeness = 100 * (1 - missing / cells)
    uniqueness = 100 * (1 - duplicates / max(1, len(frame)))
    validity = 100 - min(100, sum(str(dtype) == "object" for dtype in frame.dtypes) / max(1, len(frame.columns)) * 8)
    quality = round(max(0, min(100, completeness * 0.55 + uniqueness * 0.25 + validity * 0.2)))
    dates = date_columns(frame)
    date_range = None
    if dates:
        parsed = pd.to_datetime(frame[dates[0]], errors="coerce").dropna()
        if not parsed.empty:
            date_range = {"column": dates[0], "start": parsed.min().date().isoformat(), "end": parsed.max().date().isoformat()}
    numerical = {}
    for column in numeric_columns(frame):
        series = frame[column].dropna()
        numerical[column] = {"count": int(series.count()), "mean": _num(series.mean()), "median": _num(series.median()), "min": _num(series.min()), "max": _num(series.max()), "std": _num(series.std(ddof=0))}
    categorical = {column: {"unique": int(frame[column].nunique(dropna=True)), "top": str(frame[column].mode(dropna=True).iloc[0]) if not frame[column].mode(dropna=True).empty else None} for column in categorical_columns(frame)}
    columns = [{"name": column, "dtype": str(frame[column].dtype), "missing": int(frame[column].isna().sum()), "unique": int(frame[column].nunique(dropna=True))} for column in frame.columns]
    return {"rows": int(len(frame)), "columns": int(len(frame.columns)), "file_size": file_size, "quality_score": quality, "missing_values": missing, "duplicates": duplicates, "columns_detail": columns, "column_names": list(frame.columns), "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()}, "numeric_summary": numerical, "categorical_summary": categorical, "date_range": date_range, "date_columns": dates, "numeric_columns": numeric_columns(frame), "categorical_columns": categorical_columns(frame)}


def _chart(frame: pd.DataFrame, question: str, value: str | None, category: str | None) -> dict[str, Any] | None:
    lowered = question.lower()
    if value is None:
        return None
    if category and any(term in lowered for term in ("top", "highest", "best", "department", "region", "category", "product", "group")):
        aggregate = "mean" if "average" in lowered or "mean" in lowered else "sum"
        grouped = getattr(frame.groupby(category)[value], aggregate)().sort_values(ascending=False).head(10)
        return {"type": "bar", "title": f"{aggregate.title()} {value} by {category}", "x": category, "y": value, "data": [{"label": str(index), "value": round(float(item), 4)} for index, item in grouped.items()]}
    dates = date_columns(frame)
    if dates:
        date_column = dates[0]
        working = frame.copy()
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        trend = working.dropna(subset=[date_column]).groupby(working[date_column].dt.to_period("M"))[value].sum().tail(24)
        return {"type": "line", "title": f"{value} over time", "x": date_column, "y": value, "data": [{"label": str(index), "value": round(float(item), 4)} for index, item in trend.items()]}
    return {"type": "histogram", "title": f"Distribution of {value}", "x": value, "data": [{"label": str(index), "value": int(item)} for index, item in frame[value].dropna().round(2).value_counts().head(12).sort_index().items()]}


def analyze_question(frame: pd.DataFrame, question: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("The dataset is empty.")
    lowered = question.lower()
    value = choose_numeric(frame, question)
    category = choose_category(frame, question)
    agents = ["Dataset Profiling Agent"]
    if any(term in lowered for term in ("anomal", "unusual", "outlier")) and value:
        agents += ["EDA Agent", "Anomaly Detection Agent"]
        series = frame[value].dropna()
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        threshold = mean + 2 * std
        count = int((series > threshold).sum())
        return {"answer": f"I found {count} potential anomalous value(s) in {value}. Values above {threshold:,.2f} were flagged using a two-standard-deviation rule. These are signals for review, not confirmed errors.", "agents": agents, "intent": "anomaly", "chart": _chart(frame, question, value, None), "supporting_data": [{"metric": value, "mean": round(mean, 4), "threshold": round(threshold, 4), "potential_anomalies": count}]}
    if ("what if" in lowered or "scenario" in lowered) and value:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", lowered)
        percentage = float(match.group(1)) if match else 10.0
        current = float(frame[value].sum())
        simulated = current * (1 + percentage / 100)
        agents += ["What-If Simulation Agent", "Recommendation Agent"]
        return {"answer": f"A simulated {percentage:g}% change in {value} moves the total from {current:,.2f} to {simulated:,.2f}. This is a scenario, not a guaranteed forecast.", "agents": agents, "intent": "simulation", "simulation": {"column": value, "current": current, "simulated": simulated, "difference": simulated - current, "change_percent": percentage, "method": "Applied the selected percentage to the observed aggregate."}}
    if "insight" in lowered:
        insights = generate_insights(frame)
        agents += ["EDA Agent", "Anomaly Detection Agent", "Explainability Agent"]
        if not insights:
            return {"answer": f"This dataset contains {len(frame):,} records and {len(frame.columns)} columns, but no measurable pattern was strong enough to report.", "agents": agents, "intent": "insights", "supporting_data": []}
        answer = " ".join(f"{item['title']}." for item in insights[:4])
        return {"answer": answer, "agents": agents, "intent": "insights", "chart": _chart(frame, question, value, category), "supporting_data": insights}
    dates = date_columns(frame)
    if value and dates and any(term in lowered for term in ("trend", "monthly", "over time", "per month", "by month", "timeline")):
        trend, partial = _complete_months(frame, dates[0], value)
        agents += ["EDA Agent", "Visualization Agent"]
        if trend.empty:
            return {"answer": f"{dates[0]} could not be parsed into dates, so no trend is available.", "agents": agents, "intent": "trend"}
        best = trend.idxmax()
        answer = f"{value} spans {len(trend)} month(s) from {trend.index[0]} to {trend.index[-1]}. The strongest month is {best} at {trend.max():,.2f}."
        if len(trend) >= 2 and trend.iloc[-2]:
            change = (trend.iloc[-1] - trend.iloc[-2]) / abs(trend.iloc[-2]) * 100
            answer += f" The latest complete month ({trend.index[-1]}) is {change:+.1f}% versus the previous month."
        if partial:
            answer += f" {partial} was excluded because the data stops partway through that month."
        chart = {"type": "line", "title": f"Monthly {value}", "x": dates[0], "y": value, "data": [{"label": str(index), "value": round(float(item), 4)} for index, item in trend.tail(24).items()]}
        return {"answer": answer, "agents": agents, "intent": "trend", "chart": chart, "supporting_data": [{"period": str(index), "value": round(float(item), 4)} for index, item in trend.items()]}
    if "correlation" in lowered or "relationship" in lowered:
        numbers = numeric_columns(frame)
        if len(numbers) >= 2:
            correlation = frame[numbers].corr().round(4).to_dict()
            agents += ["EDA Agent", "Visualization Agent"]
            return {"answer": f"I calculated correlations across {len(numbers)} numerical columns. The strongest relationship should be reviewed in the correlation matrix below.", "agents": agents, "intent": "correlation", "correlation": correlation, "chart": {"type": "heatmap", "title": "Numerical correlation matrix", "columns": numbers, "data": correlation}}
    if any(term in lowered for term in ("recommend", "focus on", "should we", "what should")) and value and category:
        grouped = frame.groupby(category)[value].sum().sort_values(ascending=False)
        leader = str(grouped.index[0])
        agents += ["EDA Agent", "Recommendation Agent"]
        return {"answer": f"Consider focusing on {leader}: it leads the observed {value} total at {grouped.iloc[0]:,.2f}. Review the drivers behind this result before applying the pattern elsewhere.", "agents": agents, "intent": "recommendation", "supporting_data": [{"group": str(index), "value": round(float(item), 4)} for index, item in grouped.head(5).items()], "chart": _chart(frame, question, value, category)}
    if value and category and any(term in lowered for term in ("highest", "lowest", "most", "least", "top", "best", "average", "by", "department", "region", "category", "product")):
        aggregate = "mean" if "average" in lowered or "mean" in lowered else "sum"
        grouped = getattr(frame.groupby(category)[value], aggregate)().sort_values(ascending=False)
        limit_match = re.search(r"top\s+(\d+)", lowered)
        limit = int(limit_match.group(1)) if limit_match else 5
        top = grouped.head(limit)
        agents += ["EDA Agent", "Visualization Agent"]
        return {"answer": f"{top.index[0]} has the highest {'average' if aggregate == 'mean' else 'total'} {value} at {top.iloc[0]:,.2f}.", "agents": agents, "intent": "grouped_comparison", "chart": _chart(frame, question, value, category), "supporting_data": [{"group": str(index), "value": round(float(item), 4)} for index, item in top.items()]}
    if value:
        total = float(frame[value].sum())
        average = float(frame[value].mean())
        agents += ["EDA Agent", "Explainability Agent"]
        return {"answer": f"This dataset contains {len(frame):,} records. For {value}, the total is {total:,.2f} and the average is {average:,.2f}.", "agents": agents, "intent": "summary", "chart": _chart(frame, question, value, category), "supporting_data": [{"metric": value, "total": total, "average": average}]}
    return {"answer": f"This dataset contains {len(frame):,} records and {len(frame.columns)} columns. It has no numerical measure available for aggregate analysis.", "agents": agents, "intent": "summary", "supporting_data": [{"rows": len(frame), "columns": len(frame.columns)}]}


def dashboard_summary(frame: pd.DataFrame, file_size: int = 0, original_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = original_profile or profile_frame(frame, file_size)
    value = choose_numeric(frame)
    category = choose_category(frame)
    derived = frame.attrs.get("derived", {})
    measure_note = f"Derived as {derived[value]} per row" if value in derived else "Calculated from current dataset"
    rows = int(profile["rows"])
    removed = rows - len(frame)
    records_note = f"{len(frame):,} analyzed after cleaning ({removed:,} duplicate rows removed)" if removed > 0 else "Rows in your upload"
    records = {"label": "Records", "value": rows, "kind": "count", "detail": records_note}
    quality = {"label": "Data quality", "value": profile["quality_score"], "kind": "quality", "detail": "Score of the original upload"}
    summary: dict[str, Any] = {"profile": profile, "kpis": [], "derived": derived, "chart": _chart(frame, "trend", value, category)}
    if value:
        series = frame[value].dropna()
        summary["kpis"] = [{"label": f"Total {value}", "value": round(float(series.sum()), 2), "kind": "sum", "detail": measure_note}, {"label": f"Average {value}", "value": round(float(series.mean()), 2), "kind": "average", "detail": measure_note}, records, quality]
    else:
        summary["kpis"] = [records, {"label": "Columns", "value": len(frame.columns), "kind": "count", "detail": "Fields detected"}, quality]
    return summary


def forecast_summary(frame: pd.DataFrame) -> dict[str, Any]:
    dates = date_columns(frame)
    value = choose_numeric(frame)
    if not dates or value is None:
        return {"available": False, "reason": "A reliable date column and numerical measure are required for a forecast."}
    trend, partial = _complete_months(frame, dates[0], value)
    if len(trend) < 3:
        return {"available": False, "reason": "At least three dated observations are required for a directional forecast."}
    x = list(range(len(trend)))
    slope, intercept = __import__("numpy").polyfit(x, trend.to_numpy(dtype=float), 1)
    fitted = [slope * index + intercept for index in x]
    mae = float(sum(abs(actual - predicted) for actual, predicted in zip(trend.to_numpy(dtype=float), fitted)) / len(trend))
    next_value = float(slope * len(trend) + intercept)
    next_period = (trend.index[-1].to_timestamp() + pd.offsets.MonthBegin(1)).date().isoformat()
    return {"available": True, "column": value, "date_column": dates[0], "next_period": next_period, "prediction": round(next_value, 2), "mae": round(mae, 2), "historical": [{"period": str(index), "value": round(float(item), 2)} for index, item in trend.items()], "method": "Directional linear trend over monthly aggregates." + (f" The final month ({partial}) has only partial data and was excluded." if partial else "")}


def _monthly(frame: pd.DataFrame, date_column: str, value: str) -> pd.Series:
    working = frame[[date_column, value]].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column])
    return working.groupby(working[date_column].dt.to_period("M"))[value].sum().sort_index()


def _complete_months(frame: pd.DataFrame, date_column: str, value: str) -> tuple[pd.Series, str | None]:
    """Monthly totals without a trailing month the data stops partway through (e.g. data ending on the 9th)."""
    trend = _monthly(frame, date_column, value)
    dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if len(trend) < 4 or dates.empty:
        return trend, None
    # How far into a month the data normally reaches (1 for month-start data, ~30 for daily data).
    reach = dates.groupby(dates.dt.to_period("M")).max().dt.day
    typical = float(reach.iloc[:-1].median())
    if reach.iloc[-1] < typical * 0.6:
        return trend.iloc[:-1], str(trend.index[-1])
    return trend, None


def derive_measures(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Revenue = Quantity x Price when a transactional dataset has no monetary total column."""
    lowered = [column.lower() for column in frame.columns]
    if any(term in name for name in lowered for term in ("revenue", "sales", "amount", "total")):
        return frame
    numbers = numeric_columns(frame)
    quantity = next((column for column in numbers if "quantity" in column.lower() or column.lower() in {"qty", "units"}), None)
    price = next((column for column in numbers if "price" in column.lower()), None)
    if not quantity or not price:
        return frame
    enriched = frame.copy()
    enriched["Revenue"] = enriched[quantity] * enriched[price]
    enriched.attrs["derived"] = {"Revenue": f"{quantity} × {price}"}
    return enriched


def generate_insights(frame: pd.DataFrame, original_profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Derive reviewable findings from the dataset itself. Returns [] when nothing measurable exists."""
    if frame.empty:
        return []
    insights: list[dict[str, Any]] = []
    value = choose_numeric(frame)
    category = choose_category(frame)
    derived = frame.attrs.get("derived", {})
    if value in derived:
        insights.append({"category": "Derived measure", "title": f"{value} was derived as {derived[value]}", "detail": f"The upload has no revenue or sales column, so {value} is calculated per row as {derived[value]}. Negative quantities (returns) reduce it.", "metric": derived[value], "tone": "blue", "link": "/datasets"})
    if value and category and category not in numeric_columns(frame):
        grouped = frame.groupby(category)[value].sum().sort_values(ascending=False)
        total = float(grouped.sum())
        if len(grouped) >= 2 and total:
            leader, leader_value = str(grouped.index[0]), float(grouped.iloc[0])
            insights.append({"category": "Performance", "title": f"{leader} leads {value} by {category}", "detail": f"{leader} accounts for {leader_value / total:.1%} of total {value} ({leader_value:,.2f} of {total:,.2f}) across {len(grouped)} {category} values.", "metric": f"{leader_value:,.2f}", "tone": "mint", "link": "/ask-data"})
            laggard, laggard_value = str(grouped.index[-1]), float(grouped.iloc[-1])
            insights.append({"category": "Opportunity", "title": f"{laggard} has the lowest {value} by {category}", "detail": f"{laggard} contributes {laggard_value:,.2f} ({laggard_value / total:.1%} of the total), the smallest share among {category} values.", "metric": f"{laggard_value / total:.1%} share", "tone": "lavender", "link": "/recommendations"})
    dates = date_columns(frame)
    if value and dates:
        trend, partial = _complete_months(frame, dates[0], value)
        if len(trend) >= 2:
            best = trend.idxmax()
            previous, latest = float(trend.iloc[-2]), float(trend.iloc[-1])
            change = (latest - previous) / abs(previous) * 100 if previous else None
            detail = f"{best} is the strongest of {len(trend)} months at {float(trend.max()):,.2f}."
            if change is not None:
                detail += f" The latest complete month ({trend.index[-1]}) moved {change:+.1f}% versus {trend.index[-2]}."
            if partial:
                detail += f" {partial} is excluded because the data stops partway through it."
            insights.append({"category": "Trend", "title": f"{best} is the strongest month for {value}", "detail": detail, "metric": f"{change:+.1f}% latest" if change is not None else f"{float(trend.max()):,.2f}", "tone": "blue", "link": "/predictions"})
    if value:
        series = frame[value].dropna()
        if len(series) >= 4:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = int(((series < low) | (series > high)).sum())
            if outliers:
                insights.append({"category": "Anomaly", "title": f"{outliers:,} potential outlier(s) in {value}", "detail": f"Values outside {low:,.2f} to {high:,.2f} (1.5 x IQR) were flagged. They are signals for review, not confirmed errors.", "metric": f"Review {outliers:,}", "tone": "peach", "link": "/anomalies"})
    numbers = [column for column in numeric_columns(frame) if not _is_identifier(column)]
    if len(numbers) >= 2:
        matrix = frame[numbers].corr()
        pairs = [(abs(float(matrix.iloc[i, j])), numbers[i], numbers[j]) for i in range(len(numbers)) for j in range(i + 1, len(numbers)) if pd.notna(matrix.iloc[i, j])]
        if pairs:
            strength, left, right = max(pairs)
            if strength >= 0.5:
                signed = float(matrix.loc[left, right])
                insights.append({"category": "Relationship", "title": f"{left} and {right} move together", "detail": f"The correlation between {left} and {right} is {signed:+.2f}. Correlation does not imply causation.", "metric": f"r = {signed:+.2f}", "tone": "lavender", "link": "/ask-data"})
    profile = original_profile or {}
    missing = int(profile.get("missing_values", frame.isna().sum().sum()))
    duplicates = int(profile.get("duplicates", frame.duplicated().sum()))
    if missing or duplicates:
        detail = f"{missing:,} missing cell(s) and {duplicates:,} duplicate row(s) were found in the original upload."
        gaps = [column for column in profile.get("columns_detail", []) if column.get("missing")]
        if gaps:
            worst = max(gaps, key=lambda column: column["missing"])
            detail += f" {worst['name']} has the most gaps ({worst['missing']:,})."
        insights.append({"category": "Data quality", "title": "Data quality issues need review", "detail": detail + " The processed copy fills numeric gaps and removes duplicates.", "metric": f"{missing + duplicates:,} issues", "tone": "peach", "link": "/datasets"})
    return insights
