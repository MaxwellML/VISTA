# post_opf_redundancy.py

from dataclasses import dataclass
import numpy as np


@dataclass
class ViewpointOPF:
    """
    OPF information associated with one candidate viewpoint.
    """
    id: str
    opf: np.ndarray


@dataclass
class SelectionRecord:
    """
    Records why a viewpoint was selected.
    """
    viewpoint_id: str
    marginal_gain: float
    marginal_gain_fraction: float
    retained_information_fraction: float


@dataclass
class RedundancyResult:
    selected: list[str]
    discarded: list[str]
    history: list[SelectionRecord]
    retained_information_fraction: float


class PostOPFRedundancyCorrector:

    def __init__(
        self,
        min_gain_fraction: float = 0.01,
        target_retention: float = 1.0
    ):
        """
        Parameters
        ----------
        min_gain_fraction:
            Minimum marginal information contribution required for another
            viewpoint to be selected.

            Example:
                0.01 means the viewpoint must contribute at least 1% of the
                total OPF information obtainable from all viewpoints.

        target_retention:
            Stop once this fraction of the total obtainable information
            has been retained.

            Example:
                0.95 means stop once the selected viewpoints retain at
                least 95% of all available OPF information.
        """

        if not 0 <= min_gain_fraction <= 1:
            raise ValueError("min_gain_fraction must be between 0 and 1.")

        if not 0 < target_retention <= 1:
            raise ValueError("target_retention must be between 0 and 1.")

        self.min_gain_fraction = min_gain_fraction
        self.target_retention = target_retention

    def select(
        self,
        viewpoints: list[ViewpointOPF]
    ) -> RedundancyResult:

        if not viewpoints:
            return RedundancyResult(
                selected=[],
                discarded=[],
                history=[],
                retained_information_fraction=0.0
            )

        # ---------------------------------------------------------
        # Validate / clean input
        # ---------------------------------------------------------

        shape = viewpoints[0].opf.shape

        cleaned_viewpoints = []

        for viewpoint in viewpoints:

            if viewpoint.opf.shape != shape:
                raise ValueError(
                    "All OPF rasters must have the same shape."
                )

            # Treat NaN / infinities as zero useful information
            cleaned = np.nan_to_num(
                viewpoint.opf.astype(float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # OPF should not be negative
            cleaned = np.maximum(cleaned, 0)

            cleaned_viewpoints.append(
                ViewpointOPF(
                    id=viewpoint.id,
                    opf=cleaned
                )
            )

        # ---------------------------------------------------------
        # Calculate the best information obtainable using ALL
        # viewpoints.
        #
        # For every raster cell, take the best OPF value available
        # from any viewpoint.
        # ---------------------------------------------------------

        full_information = np.maximum.reduce(
            [v.opf for v in cleaned_viewpoints]
        )

        total_possible_information = np.sum(full_information)

        if total_possible_information == 0:

            return RedundancyResult(
                selected=[],
                discarded=[v.id for v in cleaned_viewpoints],
                history=[],
                retained_information_fraction=0.0
            )

        # ---------------------------------------------------------
        # Begin with no viewpoints selected
        # ---------------------------------------------------------

        accumulated_information = np.zeros(
            shape,
            dtype=float
        )

        remaining = cleaned_viewpoints.copy()

        selected_ids = []
        history = []

        # ---------------------------------------------------------
        # Greedy selection
        # ---------------------------------------------------------

        while remaining:

            best_viewpoint = None
            best_gain = -1
            best_information = None

            # Test every remaining viewpoint
            for viewpoint in remaining:

                # What would our information look like if we added it?
                combined_information = np.maximum(
                    accumulated_information,
                    viewpoint.opf
                )

                # What information does it add?
                marginal_information = (
                    combined_information
                    - accumulated_information
                )

                marginal_gain = np.sum(
                    marginal_information
                )

                if marginal_gain > best_gain:

                    best_gain = marginal_gain
                    best_viewpoint = viewpoint
                    best_information = combined_information

            # -----------------------------------------------------
            # Convert marginal contribution to fraction of ALL
            # obtainable information
            # -----------------------------------------------------

            gain_fraction = (
                best_gain
                / total_possible_information
            )

            # -----------------------------------------------------
            # If even the BEST remaining viewpoint adds too little,
            # all remaining viewpoints are redundant.
            # -----------------------------------------------------

            if gain_fraction < self.min_gain_fraction:
                break

            # -----------------------------------------------------
            # Accept viewpoint
            # -----------------------------------------------------

            accumulated_information = best_information

            selected_ids.append(
                best_viewpoint.id
            )

            remaining.remove(
                best_viewpoint
            )

            retained_fraction = (
                np.sum(accumulated_information)
                / total_possible_information
            )

            history.append(
                SelectionRecord(
                    viewpoint_id=best_viewpoint.id,
                    marginal_gain=float(best_gain),
                    marginal_gain_fraction=float(gain_fraction),
                    retained_information_fraction=float(
                        retained_fraction
                    )
                )
            )

            # -----------------------------------------------------
            # Optional early stop once enough information retained
            # -----------------------------------------------------

            if retained_fraction >= self.target_retention:
                break

        discarded_ids = [
            viewpoint.id
            for viewpoint in remaining
        ]

        retained_fraction = (
            np.sum(accumulated_information)
            / total_possible_information
        )

        return RedundancyResult(
            selected=selected_ids,
            discarded=discarded_ids,
            history=history,
            retained_information_fraction=float(
                retained_fraction
            )
        )