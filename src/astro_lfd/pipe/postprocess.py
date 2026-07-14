import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pipe.base as pipeBase

from astro_lfd.geom import Line2D
from astro_lfd.table import StreakAdapter

__all__ = ["WriteStreakTaskConfig", "WriteStreakCatalogTask"]


class WriteStreakCatalogConnections(
    pipeBase.PipelineTaskConnections, 
    dimensions=("instrument", "visit", "detector"),
    defaultTemplates={"catalogType": "", "coaddName": "deep", "fakesType": ""},
):
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
        doc="Catalog of streaks in Astropy format.",
        name="{catalogType}streakTable",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )


class WriteStreakCatalogConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=WriteStreakCatalogConnections,
):
    pass


class WriteStreakCatalogTask(pipeBase.PipelineTask):
    """Write streaks dictionary to Astropy table format.
    """
    _DefaultName = "writeStreakCatalog"
    ConfigClass = WriteStreakCatalogConfig

    def run(self, streaks, difference):

        schema = StreakAdapter.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)

        wcs = differnece.getWcs()
        box = difference.getBBox()
        shift = geom.Extent2D(box.getCenter())
        for n in range(len(num_streaks)):
            kht_line = Line2D(streaks["rho"][n], streaks["theta"][n] * geom.degrees)
            det_line = kht_line.translated(shift)
            line_segment = det_line.intersection(box)
            if line_segment is None:
                continue
            center = line_segment.center

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(line_segment)
            #streak.setFootprint(footprint)  # No way to separate individual mask contribution
            if wcs is not None:
                streak.setCoord(wcs.pixelToSky(center))
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()
            streak["line_sigma"] = fit.sigma
            streak["line_reduced_chi2"] = fit.reducedChi2
            streak["line_model_maximum"] = float(model_maximum)

        result = pipeBase.Struct(output_catalog=catalog)
        return result
