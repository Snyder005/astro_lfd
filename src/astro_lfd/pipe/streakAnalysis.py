# To Do: Add runQuantum function to incorporate partial failures and id generator
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from astro_lfd.meas import KHTDetectTask
from astro_lfd.table import StreakAdapter

__all__ = ["StreakAnalysisConfig", "StreakAnalysisTask"]


class StreakAnalysisConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
    defaultTemplates={"coaddName": "deep", "fakesType": ""},
):
    difference = pipeBase.connectionTypes.Input(
        doc="Input difference image with detection mask plane filled in.",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )
    dia_streaks = pipeBase.connectionTypes.Output(
        doc="Detected dia_streaks on the difference image.",
        name="{fakesType}{coaddName}Diff_diaStreaks",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )


class StreakAnalysisConfig(pipeBase.PipelineTaskConfig, pipelineConnections=StreakAnalysisConnections):
    detection_algorithm = pexConfig.ChoiceField(
        dtype=str,
        default="kht",
        doc="Line detection algorithm to use.",
        allowed={
            "kht": "Kernel Hough Transform.",
            #"adrt": "Approximate Discrete Radon Transform.",
        },
    )

    # Detector tasks
    kht_detect = pexConfig.ConfigurableField(
        target=KHTDetectTask,
        doc="Detect streaks using kernel Hough transform.",
    )


class StreakAnalysisTask(pipeBase.PipelineTask):
    """Detect and measure linear features on a difference image.
    """
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
        results : `lsst.pipe.base.Struct`
        
            ``dia_streaks`` : `lsst.afw.table.SourceCatalog`
                The catalog of detected streaks.
        """ 
        schema = StreakAdapter.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        results = pipeBase.Struct()

        if self.config.detection_algorithm == "kht":
            detect_results = self.kht_detect.run(table, exposure)

        results.streaks = detect_results.streaks
        return results
