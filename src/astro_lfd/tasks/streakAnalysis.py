__all__ = ["StreakAnalysisConfig", "StreakAnalysisTask"]

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from ..algorithms.khtDetect import KHTDetectTask
from ..table.streakAdapter import StreakAdapter


class StreakAnalysisConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
    defaultTemplates={"coaddName": "deep", "fakesType": ""},
):
    """Connections for `WriteStreakCatalogTask`."""

    difference = pipeBase.connectionTypes.Input(
        doc="Input difference image with detection mask plane filled in.",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )
    dia_streaks = pipeBase.connectionTypes.Output(
        doc="Detected dia_streaks on the difference image.",
        name="{fakesType}{coaddName}Diff_diaStrk",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )


class StreakAnalysisConfig(pipeBase.PipelineTaskConfig, pipelineConnections=StreakAnalysisConnections):
    """Configurable parameters for `StreakAnalysisTask`."""

    detection_algorithm = pexConfig.ChoiceField(
        dtype=str,
        default="kht",
        doc="Line detection algorithm to use.",
        allowed={
            "kht": "Kernel Hough Transform.",
            # "adrt": "Approximate Discrete Radon Transform.",
        },
    )

    # Detector tasks
    kht_detect = pexConfig.ConfigurableField(
        target=KHTDetectTask,
        doc="Detect streaks using kernel Hough transform.",
    )


class StreakAnalysisTask(pipeBase.PipelineTask):
    """Detect and measure linear features on a difference image."""

    ConfigClass = StreakAnalysisConfig
    _DefaultName = "streakAnalysis"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config.detection_algorithm == "kht":
            self.makeSubtask("kht_detect")

    def run(self, difference: afwImage.ExposureF) -> pipeBase.Struct:
        """Detect and measure linear features on a difference image.

        The difference image will be processed using the configured linear
        feature detector.

        Parameters
        ----------
        difference : `lsst.afw.image.ExposureF`
            Difference image with detection mask filled in.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result as a struct with attributes:

            ``dia_streaks``
                Catalog of detected streaks (`lsst.afw.table.SourceCatalog`).
        """
        schema = StreakAdapter.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)

        if self.config.detection_algorithm == "kht":
            detect_result = self.kht_detect.run(table, difference)

        return pipeBase.Struct(dia_streaks=detect_result.streaks)
