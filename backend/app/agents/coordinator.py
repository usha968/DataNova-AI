from __future__ import annotations

import pandas as pd

from app.services.analytics import analyze_question as run_dynamic_analysis


class MultiAgentCoordinator:
    """Routes controlled analytical intents to the dynamic analytics service."""

    def run(self, frame: pd.DataFrame, question: str) -> dict:
        return run_dynamic_analysis(frame, question)


def analyze_question(frame: pd.DataFrame, question: str) -> dict:
    return MultiAgentCoordinator().run(frame, question)
