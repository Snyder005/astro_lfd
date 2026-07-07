"""Adapter exposing streak line geometry on an afw `SourceRecord`.

`StreakAdapter` wraps a single `lsst.afw.table.SourceRecord` and presents its
``line_*`` fields as `~astro_lfd.geom.line.Line2D` / `LineSegment2D` geometry,
bridging the LSST measurement catalog to the LFD line primitives.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import lsst.afw.detection as afwDetect
import lsst.afw.table as afwTable
import lsst.geom as geom

from ..geom.line import Line2D, LineSegment2D

__all__ = ["StreakAdapter"]


class StreakAdapter:
    """A view of a `SourceRecord` as a streak line segment.

    Parameters
    ----------
    record : `lsst.afw.table.SourceRecord`
        The source record whose ``line_*`` fields describe the streak.
    """

    def __init__(self, record: afwTable.SourceRecord) -> None:
        self._record = record

    def __repr__(self) -> str:
        seg = self.getLineSegment()

        return (
            f"StreakAdapter("
            f"rho={seg.rho:.2f}, "
            f"theta={seg.theta.asDegrees():.2f} deg, "
            f"length={seg.length:.2f})"
        )

    def __getitem__(self, key: Any) -> Any:
        return self._record[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._record[key] = value

    @property
    def record(self) -> afwTable.SourceRecord:
        """The wrapped source record (`lsst.afw.table.SourceRecord`)."""
        return self._record

    def getLine(self) -> Line2D:
        """Return the infinite line for this streak.

        Returns
        -------
        line : `astro_lfd.geom.line.Line2D`
            The line defined by the ``line_rho`` and ``line_theta`` fields.
        """
        return Line2D(rho=self["line_rho"], theta=self["line_theta"])

    def getLineSegment(self) -> LineSegment2D:
        """Return the finite streak segment.

        Returns
        -------
        segment : `astro_lfd.geom.line.LineSegment2D`
            The segment defined by the line plus ``line_u_center`` and
            ``line_length``.
        """
        return LineSegment2D.from_center_length(
            line=self.getLine(),
            u_center=self["line_u_center"],
            length=self["line_length"],
        )

    def setLineSegment(self, segment: LineSegment2D) -> None:
        """Store a line segment into the record's ``line_*`` fields.

        Parameters
        ----------
        segment : `astro_lfd.geom.line.LineSegment2D`
            The segment to persist.
        """
        self["line_rho"] = segment.rho
        self["line_theta"] = segment.theta
        self["line_u_center"] = segment.interval.center
        self["line_length"] = segment.length

    def getFootprint(self) -> afwDetect.Footprint:
        """Return the record's footprint (`lsst.afw.detection.Footprint`)."""
        return self.record.getFootprint()

    def setFootprint(self, footprint: afwDetect.Footprint) -> None:
        """Set the record's footprint.

        Parameters
        ----------
        footprint : `lsst.afw.detection.Footprint`
            The footprint to store.
        """
        self.record.setFootprint(footprint)

    def getCentroid(self) -> geom.Point2D:
        """Return the record's centroid (`lsst.geom.Point2D`)."""
        return self.record.getCentroid()

    def getCoord(self) -> geom.SpherePoint:
        """Return the record's sky coordinate (`lsst.geom.SpherePoint`)."""
        return self.record.getCoord()

    def setCoord(self, coord: geom.SpherePoint) -> None:
        """Set the record's sky coordinate.

        Parameters
        ----------
        coord : `lsst.geom.SpherePoint`
            The sky coordinate to store.
        """
        self.record.setCoord(coord)

    @staticmethod
    def makeMinimalSchema() -> afwTable.Schema:
        """Build a minimal source schema with the streak ``line_*`` fields.

        Returns
        -------
        schema : `lsst.afw.table.Schema`
            A minimal source schema extended with ``line_rho``,
            ``line_theta``, ``line_u_center``, ``line_length``, the profile-fit
            quality fields ``line_sigma``, ``line_reduced_chi2``, and
            ``line_model_maximum``, and a ``line_center`` centroid key.
        """
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField(
            "line_rho",
            type=np.float64,
            units="pixel",
        )
        schema.addField(
            "line_theta",
            type=geom.Angle,
        )
        schema.addField(
            "line_u_center",
            type=np.float64,
            units="pixel",
        )
        schema.addField(
            "line_length",
            type=np.float64,
            units="pixel",
        )
        schema.addField(
            "line_sigma",
            type=np.float64,
            doc="Moffat sigma (width) parameter of the fit line profile.",
            units="pixel",
        )
        schema.addField(
            "line_reduced_chi2",
            type=np.float64,
            doc="Reduced chi-squared of the line profile fit.",
        )
        schema.addField(
            "line_model_maximum",
            type=np.float64,
            doc="Peak absolute value of the fit line profile model.",
            units="nJy",
        )
        afwTable.Point2DKey.addFields(
            schema,
            "line_center",
            "Line segment center",
            "pixel",
        )

        return schema
