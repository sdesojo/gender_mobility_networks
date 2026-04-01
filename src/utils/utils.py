from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from math import acos, cos, sin, radians
import numpy as np

############################ USEFUL VARS ############################

GENDER = ["MALE", "FEMALE"]
CTRY = ["NLD", "AUS", "DEU", "USA", "FRA", "GBR", "TWN", "ESP", "SWE", "JPN"]

dMLAB = {
    "nStops_M": "Activity, N",
    "nuStops_M": "Repertoire size, k",
}
dTEST = {"ie30": "inactive", "mid3080": "moderate", "ae80": "active", "all": "all"}


############################ SET-UP VISUALIZATIONS ############################
def set_mpl_style():
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "HelvetMatplotlib"
    mpl.rcParams["lines.linewidth"] = 1
    mpl.rcParams["figure.figsize"] = (3.5, 2.5)
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["pdf.fonttype"] = 42  # To edit text in Illustrator

    fs = 7  # 9
    mpl.rcParams["font.size"] = fs
    mpl.rcParams["axes.titlesize"] = fs
    mpl.rcParams["axes.labelsize"] = fs
    mpl.rcParams["xtick.labelsize"] = fs
    mpl.rcParams["ytick.labelsize"] = fs
    mpl.rcParams["legend.fontsize"] = fs

    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"


cm_to_inch = 1 / 2.54

dTESTmarker = {
    "ie30": "v",
    "mid3080": "d",
    "ae80": "^",
    "all": "o",
}
dTESTsize = {
    "ie30": 2,
    "mid3080": 2,
    "ae80": 2,
    "all": 10,
}

COLOR = {
    "MALE": "#EB8347",
    "FEMALE": "#5D3A93",
    "MALE light": "#F4C1A4",  # --> male lighter
    "FEMALE light": "#C0B1D2",  # --> female lighter
    "MALE background": "#F9DFD0",  # --> male lighter
    "FEMALE background": "#D6CDE3",  # --> female lighter
    "all": "k",
    "inactive": "#64D2DC",
    "moderate": "#03BC8C",
    "active": "#006349",
}

percent_labels = [i for i in range(1, 11, 1)]
quantile_labels = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, 11)]
perc2quant = {p: q for p, q in zip(percent_labels, quantile_labels)}


############################ SPARK UTILS ############################
from pyspark.sql import SparkSession


def start_spark(n_workers: int, temp_folder: str, mem: str = "80g") -> SparkSession:
    return (
        SparkSession.builder.config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.driver.memory", mem)
        .config("spark.driver.maxResultSize", "40g")
        .config("spark.executer.memory", "40g")
        .config("spark.local.dir", temp_folder)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.kryoserializer.buffer.max", "128m")
        .config("spark.storage.memoryFraction", "0.5")
        .config("spark.sql.broadcastTimeout", "7200")
        .master(f"local[{n_workers}]")
        .getOrCreate()
    )


############################ FUNCTIONS ############################


def distance(lat_p1=None, lon_p1=None, lat_p2=None, lon_p2=None):
    """
    Calculates the great-circle distance (in km) between two
    GPS points p1 and p2 - Harvesine formula
    https://en.wikipedia.org/wiki/Great-circle_distance#Formulae
    Code from @Simone Centellegher
    -------------------------------------
    :param lat_p1: latitude of origin point
    :param lon_p1: longitude of origin point
    :param lat_p2: latitude of destination point
    :param lon_p2: longitude of destination point
    :returns: distance in km
    """
    return F.acos(
        F.sin(F.toRadians(lat_p1)) * F.sin(F.toRadians(lat_p2))
        + F.cos(F.toRadians(lat_p1))
        * F.cos(F.toRadians(lat_p2))
        * F.cos(F.toRadians(lon_p1) - F.toRadians(lon_p2))
    ) * F.lit(6371.0)


def distance_py(lat_p1, lon_p1, lat_p2, lon_p2):  # [PYTHON version]
    """
    Calculates great-circle distance (in km) between two GPS points using the same
    spherical law of cosines formula as your Spark version.
    """
    if (lat_p1 == lat_p2) and (lon_p1 == lon_p2):
        return 0
    return (
        acos(
            sin(radians(lat_p1)) * sin(radians(lat_p2))
            + cos(radians(lat_p1))
            * cos(radians(lat_p2))
            * cos(radians(lon_p1) - radians(lon_p2))
        )
        * 6371.0
    )  # Earth radius in kilometers


def get_fract_pop(ctry):
    """
    Individuals per country, normalized across countries available in the analysis
    """
    # https://www.populationpyramid.net/population-size-per-country/2017/
    data = {
        "USA": 0.416723038548721,
        "JPN": 0.1600502537374388,
        "DEU": 0.10440386812912543,
        "GBR": 0.08347925503431211,
        "FRA": 0.08105224253089227,
        "ESP": 0.058863593224739726,
        "AUS": 0.03107225947862733,
        "TWN": 0.029903041019936665,
        "NLD": 0.02174295481940657,
        "SWE": 0.012709493476800057,
    }
    return data[ctry]


def get_resmapled_df(df, ctry, pop_weight=False):
    """
    Resample the dataframe by country and optionally by population weights.
    """

    if pop_weight:
        frac_pop = get_fract_pop(ctry)
        df_resampled = df.sample(frac=frac_pop, replace=True)

    else:
        df_resampled = df.sample(frac=1, replace=True)

    return df_resampled


def log_to_km(x, pos):
    km = 10**x
    # format nicely: 1, 10, 100, 1000 km
    if km < 1:
        return f"{km:.2f}"
    elif km < 10:
        return f"{km:.1f}"
    else:
        return f"{int(km):,}"


def pooled_se(x):
    n = len(x)
    return np.sqrt(np.sum(x**2)) / n
