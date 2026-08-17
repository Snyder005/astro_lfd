__all__ = ["WriteStreakCatalogConfig", "WriteStreakCatalogTask"]

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pipe.base as pipeBase

from ..geom.line import Line2D
from ..table.streakAdapter import StreakAdapter


class WriteStreakCatalogConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
    defaultTemplates={"catalogType": "", "coaddName": "deep", "fakesType": ""},
):
    """Connections for `WriteStreakCatalogTask`."""

    streaks = pipeBase.connectionTypes.Input(
        doc="Structured dictionary of streak fit parameters for the difference image.",
        name="{catalogType}streaks",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit", "detector"),
    )
    difference = pipeBase.connectionTypes.Input(
        doc="Input difference image with detection mask plane filled in.",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )
    output_catalog = pipeBase.connectionTypes.Output(
        doc="Catalog of detected streaks on the difference image.",
        name="{catalogType}strk",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )


class WriteStreakCatalogConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=WriteStreakCatalogConnections,
):
    """Configurable parameters for `WriteStreakCatalogTask`."""

    pass


class WriteStreakCatalogTask(pipeBase.PipelineTask):
    """Write streaks dictionary to Astropy table format."""

    _DefaultName = "writeStreakCatalog"
    ConfigClass = WriteStreakCatalogConfig

    def run(self, streaks: dict, difference: afwImage.ExposureF) -> pipeBase.Struct:
        """Convert a `streaks` structured dictionary to a catalog.

        Parameters
        ----------
        streaks: `dict`
            The structured dictionary of detected streaks to be converted.
        difference: `lsst.afw.image.ExposureF`
            The difference image exposure with a detected pixel mask plane
            filled in.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            The result as a struct with attributes:

            ``output_catalog``
                Catalog of detected streaks (`lsst.afw.table.SourceCatalog`).
        """
        schema = StreakAdapter.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)

        wcs = difference.getWcs()
        box = difference.getBBox()
        # The streak parameters are recorded in image-array coordinates
        # centered on the image, so map each line into absolute pixel
        # coordinates by translating from the image center (matching
        # `astro_lfd.meas.detectStreaks.KHTDetectTask`).
        shift = geom.Extent2D(box.getCenter())
        num_streaks = len(streaks["rho"])
        for n in range(num_streaks):
            kht_line = Line2D(streaks["rho"][n], streaks["theta"][n] * geom.degrees)
            det_line = kht_line.translated(shift)
            line_segment = det_line.intersection(box)
            if line_segment is None:
                continue
            center = line_segment.center

            streak = StreakAdapter(catalog.addNew())
            streak.setLineSegment(line_segment)
            # No way to separate individual mask contribution, so the
            # per-streak footprint is left unset.
            if wcs is not None:
                streak.setCoord(wcs.pixelToSky(center))
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()
            streak["line_sigma"] = streaks["sigma"][n]
            streak["line_reduced_chi2"] = streaks["reducedChi2"][n]
            streak["line_model_maximum"] = float(streaks["modelMaximum"][n])

        result = pipeBase.Struct(output_catalog=catalog)
        return result
