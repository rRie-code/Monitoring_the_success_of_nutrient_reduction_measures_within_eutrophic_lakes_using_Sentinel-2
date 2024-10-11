# -*- coding: utf-8 -*-
"""
>>Monitoring the success of nutrient reduction measures within eutrophic lakes using Sentinel-2<<
[Sentinel-2-basierte Erfolgskontrolle von Sanierungsmaßnahmen zur Nährstoffminderung in 
eutrophen Gewässern]
                    - Data analysis script for a thesis -
================================================================================================
@author: R. Riethmüller, 2024 
@version: 1.0 (DE)
Note: labels are in German - a pure English version may follow

This script provides time series analysis for Sentinel-2 based chlorophyll-a concentrations 
in water. First, satellite scenes were processed with the SeNtinel Applications Platform 
(SNAP v10.0.0) and the inbuilt Case 2 Regional CoastColor processor (C2RCC v1.2), using the 
C2X-Complex neural net. The results were exported with the PixEx operator (v1.3) 
without spatial aggregation (key word in brackets explains if step is mandatory or optional). 
Here, the following data analysis steps are done:
    1) General data preparation (MANDATORY)
    -opening an in situ data set for validation
    -opening the data set exported from SNAP and 
    -customising: defining your file prefixes and the statistical method for spatial aggregation
    -analysis of every scene: write statistical properties (e.g. mean, median, max) to a log file
    2) Analysis of whole time series (OPTIONAL)
    -plotting the whole time series (chlorophyll-a vs. time). Possible to choose between Sentinel-2 based
    chl-a values only and both, Sentinel-2 based and in situ data.
    3) Sliced time series (OPTIONAL)
    -same as in 2), but divided into three periods (2016-2018, 2019-2021, 2022-2024)
    4) Validation for whole timeseries (MANDATORY)
    -matching of in situ data and Sentinel-2 data (possibility to specify time interval)
    -checking the error distribution (normal or log-normal)
    -calculating error metrics, depending on the distribution (see Seegers et al., 2018)
    5) Violin plot (OPTIONAL)
    -comparing the chlorophyll-a distribution between years
    6) Yearwise analysis of satellite data (MANDATORY)
    -trying to calculate error metrics for each year
    -calculating statistical means (e.g. mean, chlorophyll-a trophic state indices after 
     Carlson (1977) and Riedmüller et al. (2014)) for each year
    -saving to a csv file
    7) Yearwise analysis of in situ data with focus on indices (MANDATORY)
    -calculating chlorophyll-a trophic state indices after Riedmüller et al. (2014) and 
    Carlson (1977) as well as the seasonal mean (March - October)
    -saving to a csv file
    8) Indices time series (MANDATORY)
    -plotting a time series of the trophic state indices for both data sets, in situ and 
    Sentinel-2 based.
    
    
References:
    C2RCC Community. (2024). Neural Nets. C2RCC Community Project. https://c2rcc.org/neural-nets/ 
        [2024-09-27].
    Carlson, R. E. (1977). A trophic state index for lakes. Limnology and Oceanography, 22(2),
        361–369. https://doi.org/10.4319/lo.1977.22.2.0361
    ESA. (2024). Snap and the Sentinel toolboxes (Version 10.0.0) [Computer software].
        European Space Agency. https://step.esa.int/main/
    Llodrà-Llabrés, J., Martínez-López, J., Postma, T., Pérez-Martínez, C. & Alcaraz-Segura, D. 
        (2023). Retrieving water chlorophyll-a concentration in inland waters from Sentinel-2 
        imagery: review of operability, performance and ways forward. International Journal of 
        Applied Earth Observation and Geoinformation, 125, 103605. 
        https://doi.org/10.1016/j.jag.2023.103605
    Riedmüller, U., Hoehn, E. & Mischke, U. (2014). Trophieklassifikation von Seen:
        Richtlinie zur Ermittlung des Trophie-Index nach LAWA für natürliche Seen,
        Baggerseen, Talsperren und Speicherseen (Stand: 2014). 
        Empfehlungen oberirdische Gewässer. Kulturbuch-Verlag.
    Seegers, B.N., Stumpf, R.P., Schaeffer, B.A., Loftin, K. & Werdell, P.J. (2018). 
        Performance metrics for the assessment of satellite data products: an ocean 
        color case study. Optics Express, 26(6), 7404–7422. https://doi.org/10.1364/OE.26.007404
"""

from datetime import datetime
from math import log10, log
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
pd.options.mode.copy_on_write = True #for compatibility with next pandas version
plt.style.use('seaborn-v0_8-muted')


def insi_reader():
    """This function reads the in situ chlorophyll-a values from a csv file.
    As there may be different csv formatting and different column names, your are
    asked to specify the arguments for the pandas.read_csv() function."""
    filepath = input("Path to your in situ data csv file. ").strip('\"')
    all_insi = pd.read_csv(filepath_or_buffer=filepath, delimiter=";",
        header=0, index_col=False, usecols=["Date", "Chl-a", "Remark"], encoding="utf-8")
    filtering = input("Do you need to filter the data by a certain column? Press y. Else, press another key.")
    if filtering.upper() == "Y":
        filter_col = input("Please specify the name of the column you want to filter")
        filter_val = input("Please specify the value you want to use for the filter")
        insi = all_insi[all_insi[filter_col]==filter_val] #only values matching the filter value
    else:
        insi = all_insi
    insi.dropna(axis=0, how="all", subset=["Chl-a"] , inplace=True)
    insi.set_index("Date", verify_integrity=True).sort_index()
    insi["Date"] = pd.to_datetime(insi["Date"], format="%d.%m.%Y")
    insi = insi[["Date", "Chl-a"]].set_index("Date")
    return insi


def c2rcc_reader():
    """Function reading the csv file generated by the PixEx operator in SNAP.
    Important: please remove the supplementary information header of the file
    so that the column names are in the first line. Otherwise, an error will occur."""
    filepath = input("Specify your file path to the data exported from SNAP. ").strip('\"')
    usecols = ["Date(yyyy-MM-dd)", "conc_chl", "conc_tsm", "IDEPIX_CLEAR_WATER", "IDEPIX_CLOUD",
              "IDEPIX_CLOUD_BUFFER", "IDEPIX_CLOUD_SHADOW", "IDEPIX_CIRRUS_SURE",
              "IDEPIX_CIRRUS_AMBIGUOUS", "Rtosa_OOR", "Rhow_OOR", "Iop_OOR"] #used bands
    all_vals = pd.read_csv(filepath, delimiter="\t", header=0,
                          usecols=usecols, parse_dates=["Date(yyyy-MM-dd)"])
    all_vals.sort_values(by="Date(yyyy-MM-dd)", inplace=True)
    all_vals.rename(columns={"Date(yyyy-MM-dd)":"Date"}, inplace=True)
    return all_vals


def chl_stats(scene):
    """This is the core element of each satellite scene analysis regarding chlorophyll-a.
    From a given pandas DataFrame, it calculates arithmetic mean, median, standard deviation, 
    25th quantile, 75th quantile, minimum and maximum as well as the number of pixels available
    for Chl-a (=chl_number)."""
    s_mean = float(scene.mean())
    s_med = float(scene.median())
    s_std = float(scene.std())
    s_25 = float(scene.quantile(0.25))
    s_75 = float(scene.quantile(0.75))
    s_min = float(scene.min())
    s_max = float(scene.max())
    chl_number = int(scene.count())
    stat_chl = {"mean":s_mean, "median":s_med, "std":s_std, "25th perc.":s_25, "75th percentile":
                s_75, "minimum":s_min, "maximum":s_max, "value_number":chl_number}
    return stat_chl


def scene_quality_analysis(scene):
    """This is the core element of each satellite scene analysis regarding quality parameters.
    From a given pandas DataFrame, it uses the IdePix masks to calculate the number of 
    cloud pixels, of cloud buffer pixels, of cloud shadow pixels, of cirrus pixels 
    (cirrus sure + cirrus ambigous). Moreover, it uses the OutOfRange bands
    generated by the C2RCC processor to calculate the number of extrapolated pixels."""
    cloud_pix = int((scene.IDEPIX_CLOUD.values == 1).sum())
    cloud_buffer_pix = int((scene.IDEPIX_CLOUD_BUFFER.values == 1).sum())
    cloud_shadow_pix = int((scene.IDEPIX_CLOUD_SHADOW.values == 1).sum())
    cirrus_pix = int((scene.IDEPIX_CIRRUS_SURE.values == 1).sum() +
                     (scene.IDEPIX_CIRRUS_AMBIGUOUS.values == 1).sum())
    oor_pix = int((scene.Rtosa_OOR.values == 1).sum() + (scene.Rhow_OOR.values == 1).sum()
                  + (scene.Iop_OOR == 1).sum())
    all_pix = int(len(scene))
    stats_quali = {"cloud_pixel":cloud_pix, "cloud_buffer_pixel":cloud_buffer_pix,
                   "cloud_shadow_pixel":cloud_shadow_pix, "cirrus_pixels":cirrus_pix,
                   "oor_pixel":oor_pix, "all_pixel":all_pix}
    return stats_quali


def scene_statistics(product, filename, agg_method:str="mean"):
    """Takes the Sentinel-2 data set (<product>) that has been exported from SNAP and read with
    (read with c2rcc_reader()). From the overall data set, each scene / date is extracted
    and the related parameters (chlorophyll-a and quality) are calculated with an
    exported data file and calculates statistics for each scene with the respective functions,
    chl_stats(scene) and scene_quality_analysis(scene).
    The results are written to "scene_statistics.txt" as log file, taking into account the
    file prefix (<filename>) defined externally [in the main programme]. 
    If more than 10 % of the scene are valid chlorophyll-a values, a single
    chlorophyll-a value is determined for this scene.
    By default, it is the arithmetic mean of the pixel values. This can be changed by
    passing another <agg_method> argument, e.g. "median" """
    filename = filename  + "_scene_statistics"
    #extracting all dates available
    dates = []
    date_x = 0
    for i in range(len(product)):
        date_i = product["Date"].iloc[i]
        if date_i != date_x:
            date_x = date_i
            dates.append(date_x)
    #build an own data frame for each date and calculate its statistics
    scene_means = []
    scene_stds = []
    scene_chl_pix = []
    scene_oor_pix = []
    scene_indexes = []
    product.set_index("Date", inplace=True)
    with open(filename+".txt","w", encoding="utf-8") as logfile:
        now = datetime.now()
        print(f"Timestamp: {now}", file=logfile)
        print(f"Number of scenes available: {len(dates)} \n", file=logfile)
        for date_i in dates:
            remo_x = product.loc[date_i]
            scene_chl_stats = chl_stats(remo_x["conc_chl"]) #chl-a related statistics
            scene_quali_stats = scene_quality_analysis(remo_x) #quality parameters
            scene_stats = scene_chl_stats | scene_quali_stats #combining the dictionaries
            scene_stats["port_chl-a_pix"] = scene_stats["value_number"]/scene_stats["all_pixel"]
            #use scene only if there are at least 10% of valid chl-a pixels
            if scene_stats["port_chl-a_pix"] > 0.1:
                scene_means.append(scene_stats[agg_method])#mean or median
                scene_stds.append(scene_stats["std"])
                scene_chl_pix.append(scene_stats["value_number"])
                scene_oor_pix.append(scene_stats["oor_pixel"])
                scene_indexes.append(date_i)
                print(f"Statistics for {date_i}, scene used: \n {scene_stats} \n",file=logfile)
            else:
                print(f"Statistics for {date_i}, too few chl-a pixels:\n{scene_stats} \n",
                      file=logfile)
        agg_values = pd.DataFrame(data={"Date":scene_indexes, "Chl-a_MEAN":scene_means,
                        "Chl-a_STD":scene_stds, "Number_of_pixels":
                           scene_chl_pix, "Out of range-pixels": scene_oor_pix}).set_index("Date",
                                                        verify_integrity=True).sort_index()
        print(f"Number of scenes accepted: {len(agg_values)}", file=logfile)
    print(f"Number of scenes available: {len(dates)}; number of scenes accepted: {len(agg_values)}")
    return agg_values


def matching(insitu, sat, interval:int=3):
    """This function checks for matches between in situ and remote sensing observations.
    The time interval can be specified (recommendation: 3 days, but may depend on the system, see
    e.g. Llodrà-Llabrés et al. (2023). For each match, the difference between the
    satellite-derived and the in situ chlorophyll-a value is calculated.
    Based on an approach by:
    manjeet_04 / GeeksforGeeks (11 Apr, 2023). Python – Find the closest date from a List. 
        https://www.geeksforgeeks.org/python-find-the-closest-date-from-a-list/ [2024-10-01]."""
    dates = sat.index
    sat_match = sat.copy(deep=True)
    match_date_list = [] #all satellite scenes in specified time interval around field measurement
    for i in range(len(dates)):
        match_date = min(insitu.index, key=lambda sub: abs(sub - dates[i]))
        timeshift = abs(match_date - dates[i])
        if timeshift.days <= interval:
            match_date_list.append(match_date)
        else:
            sat_match.drop(dates[i], inplace=True)
    insi_match = insitu.loc[match_date_list] #extract the matching in situ dates
    return insi_match, sat_match


def total_timeseries_plot(insi_data, sat_data, filename:str, agg_method_str:str):
    """Plotting a time series of chlorophyll-a concentration means with standard deviations
    of each Sentinel-2 scene as error bars. First date: 2016-01-01; last date: 2024-08-31.
    Pass satellite-based data (<sat_data>) and in situ data (<insi_data>) as pandas DataFrames.
    The <filename> argument is a prefix for the produced image file. Pass an <agg_method_str> 
    to match the labeling with your way of pixel aggregation, e.g. "Median"."""
    filename = filename + "_whole_time_series"
    height = 15/2.54 #conversion cm -> inches
    width = 30/2.54 #conversion cm -> inches
    marker = "."
    first = datetime.strptime("2016-01-01", "%Y-%m-%d") #first date
    last = datetime.strptime("2024-08-31", "%Y-%m-%d") #last date
    fig, ax = plt.subplots(figsize=(width, height), layout="tight")
    ax.errorbar(sat_data.index, sat_data["Chl-a_MEAN"], yerr=sat_data["Chl-a_STD"],
                fmt=marker, linestyle="-", linewidth=0.5, color="#00ff00", ecolor="k",
                label=f"Satellitendaten ({agg_method_str} und Standardabweichung)")
    plot_insi = input("Do you want to plot in situ data time series too? Press Y. Else, press another key. ")
    if plot_insi.upper() == "Y":
        ax.plot(insi_data, marker=marker, linestyle="-", linewidth=0.5, color="#008000",
                label="Insitu-Daten")
    ax.legend(framealpha=0.7)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,13,3)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=50))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=10))
    ax.set_xlim(xmin=first, xmax=last)
    ax.set_ylim(ymin=0)
    ax.grid(which="both", axis="both")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Chlorophyll-a-Konzentration in µg/l")
    plt.xticks(rotation=90)
    plt.savefig(filename+".png", dpi=300, bbox_inches="tight")
    plt.show()


def sliced_timeseries_plot(insi_data, sat_data, filename:str, agg_method_str:str,
                           leg=None):
    """Plotting a sliced version of the total time series: 2016-01-01 - 2018-12-31; 
    2019-01-01 - 2021-12-31; 2022-01-01 - 2024-08-31.
    Pass satellite-based data (<sat_data>) and in situ data (<insi_data>) as pandas DataFrames.
    The <filename> argument is a prefix for the produced image file. Pass an <agg_method_str> 
    to match the labeling with your way of pixel aggregation, e.g. "Median".
    <leg> determines which plot gets a legend ("all", 0, 1, 2 or None; default: None = no legend).
    Saving the result as svg file allows a post-processing, e.g. of the legend position."""
    filename = filename + "_sliced_time_series"
    height = 8/2.54 #conversion cm -> inches
    width = 16/2.54 #conversion cm -> inches
    marker = "."
    y_major = int(input("Specify major y axis ticks (integer) "))
    y_minor = int(input("Specify minor y axis ticks (integer) "))
    ymax = int(input("""Specify your y axis limit (integer). Tip: Have a look on the 
whole_time_series plot created before. """))
    slices = ["2016-01-01", "2019-01-01", "2022-01-01", "2024-08-31"]
    plot_insi = input("Do you want to plot in situ data time series too? Press Y. Else, press another key. ")
    for day in range(len(slices)):
        slices[day] = datetime.strptime(slices[day], "%Y-%m-%d") #conversion to datetime object
    #iterating through the periods
    for row in range(3):
        fig, ax = plt.subplots(figsize=(width, height))
        ax.errorbar(sat_data.index, sat_data["Chl-a_MEAN"], yerr=sat_data["Chl-a_STD"],
                    fmt=marker, linestyle="-", linewidth=0.5, color="#00ff00", ecolor="k",
                    label=f"Satellitendaten ({agg_method_str} und Standardabweichung)")
        if plot_insi.upper() == "Y":
            ax.plot(insi_data, marker=marker, linestyle="-", linewidth=0.5, color="#008000",
                    label="Insitu-Daten")
        if str(row) == leg:
            ax.legend(framealpha=0.7) #legend only for third plot
        if leg == "all":
            ax.legend()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,13,3)))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=y_minor))
        ax.set_xlim(xmin=slices[row], xmax=slices[row+1])
        ax.set_ylim(ymin=0, ymax=ymax)
        ax.grid(which="both", axis="both")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Chlorophyll-a-Konzentration in µg/l")
        plt.xticks(rotation=90)
        plt.savefig(filename+f"-{row}.svg", dpi=300, bbox_inches="tight")
        plt.show()


def sat_vs_field(insi_match, sat_match, filename, year:str=""):
    """Plotting satellite chlorophyll-a values (on the y axis, <sat_match>) vs. matching
    in situ chlorophyll-a values (on the x axis, <insi_match>). 
    Besides, a linear regression is performed and plotted.
    The <filename> argument is a prefix for the image files produced.
    The <year> argument is necessary for the file name if a specific year is analysed."""
    filename = filename + "_satellite_vs_insitu"
    if len(insi_match) > 2:
        height = 12/2.54 #conversion cm -> inches
        width = 12/2.54 #conversion cm -> inches
        insi_max = insi_match.max()
        sat_max = sat_match.max()
        xmax = ymax = max(insi_max, sat_max) + 5 #determine axis limits and unify them
        fig, ax = plt.subplots(figsize=(width, height))
        ax.scatter(x=insi_match, y=sat_match, marker=".")
        ax.set_xlabel("Insitu-Daten \n (Chlorophyll-a-Konzentration in µg/l)")
        ax.set_ylabel("Satellitendaten \n (Chlorophyll-a-Konzentration in µg/l) ")
        ax.set_xlim(xmin=0, xmax=xmax)
        ax.set_ylim(bottom=0, top=ymax)
        manual_tick = input("Give a number for the interval of axis ticks, plot sat vs. in situ. Press ENTER to keep default of 5. ")
        if manual_tick == "":
            tickspace = 5
        else:
            tickspace = float(manual_tick)
        loc = ticker.MultipleLocator(base=tickspace)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.axline(xy1=[0,0], slope=1, color="gray", linestyle="--", label="1:1 line")
        res = stats.linregress(insi_match, sat_match)
        ax.plot(insi_match, res.intercept + res.slope*insi_match, 'r',
                 label=f"""Lineare Regression, R2={res.rvalue:.3f}, N={len(insi_match)}
Anstieg={res.slope:.3f}, Intercept={res.intercept:.3f}""")
        ax.legend(framealpha=0.7)
        ax.grid(which="both", axis="both")
        plt.xticks(rotation=90)
        plt.savefig(filename + str(year) + ".png", bbox_inches="tight", dpi=300)
        plt.show()
        res_stat = [res.rvalue, res.slope, res.intercept]
    else:
        res = "Too few matches for Satellitendaten vs. Insitu-Daten"
        print(res)
        res_stat = ["Too few matches", "Too few matches", "Too few matches"]
    with open(filename+".txt","a", encoding="utf-8") as statlog:
        now = datetime.now()
        print(f"Timestamp: {now}", file=statlog)
        print(f"{year} Linear regression: {res}, N={len(insi_match)}", file=statlog)
    return res_stat


def distrib_plot(frame, title:str, test):
    """Creating a q-q-plot and a histogram for a given sorted, one-column pandas DataFrame
    containing the differences between in situ and satellite-based chlorophyll-a value.
    The <title> argument is a prefix for the image files produced."""
    plt.figure()
    sm.qqplot(frame)
    plt.axline(xy1=[0,0], slope=1, color="gray", linestyle="--", label="1:1 line")
    plt.figtext(0, 1, test, fontsize="medium")
    plt.legend()
    plt.savefig(title+"_qqplot.png", dpi=300, bbox_inches="tight")
    plt.figure()
    plt.hist(frame)
    plt.ylabel("Chlorophyll-a-Konzentration in µg/l")
    plt.figtext(0, 1, test, fontsize="medium")
    plt.savefig(title+"_hist.png", dpi=300, bbox_inches="tight")
    plt.show()


def match_difference(insitu_m, remote_m, filename, year:str=""):
    """Calculating the difference ("error") between satellite-derived chlorophyll-a data and
    matching in situ data for each satellite scene available.
    The <filename> argument is a prefix for the image files produced in the sub-function
    distrib_plot(frame, title:str, test).
    The <year> argument is necessary for the file name if a specific year is analysed."""
    filename = filename + "_matching_differences" + year
    dif = []
    for match_i in range(len(insitu_m)):
        insi_i = insitu_m.iloc[match_i]
        remote_i = remote_m.iloc[match_i]
        dif_i = remote_i - insi_i
        dif.append(dif_i)
    if len(insitu_m) > 2: #at least three pairs of value necessary!
        testval = str(stats.shapiro(dif)) + f", N={len(dif)}"
    else:
        testval = f"Too few data for test ({len(dif)} < 3)"
    dif_frame = pd.DataFrame(data={"Chl-a": dif}, index=pd.RangeIndex(len(dif)))
    distrib_plot(dif_frame.sort_values(by="Chl-a"), filename, testval)
    return dif_frame


def match_log_difference(insitu_m, remote_m, filename, year:str=""):
    """Calculating the difference ("error") between log-transformed satellite-derived 
    chlorophyll-a data and matching log-transformed in situ data.
    The <filename> argument is a prefix for the image files produced in the sub-function
    distrib_plot(frame, title:str, test).
    The <year> argument is necessary for the file name if a specific year is analysed."""
    filename = filename + "_matching_log_differences" + year
    dif = []
    for match_i in range(len(insitu_m)):
        insi_i = insitu_m.iloc[match_i]
        remote_i = remote_m.iloc[match_i]
        dif_i = log10(remote_i) - log10(insi_i)
        dif.append(dif_i)
    if len(insitu_m) > 2: #at least three pairs of value necessary!
        testval = str(stats.shapiro(dif)) + f", N={len(dif)}"
    else:
        testval = f"Too few data for test ({len(dif)} < 3)"
    dif_frame = pd.DataFrame(data={"Chl-a": dif}, index=pd.RangeIndex(len(dif)))
    distrib_plot(dif_frame.sort_values(by="Chl-a"), filename, testval)
    return dif_frame


def norm_error_metrix(dif, filename:str, year:str=""):
    """This function calculates Root Mean Square Error (rmse), Mean Average Error (mae) and bias
    out of the differences between satellite derived data (remote) and matching
    in situ data under the assumption of a normally distributed difference / error.
    See Seegers et al. (2018).
    The <filename> argument is a prefix for log file produced.
    The <year> argument is necessary for the file name if a specific year is analysed."""
    filename = filename + "_satellite_vs_insitu"
    rmse_sum = 0
    mae_sum = 0
    bias = 0
    N = len(dif)
    if N > 0:
        for i in range(N):
            diff_i = float(dif.iloc[i])
            rmse_sum = rmse_sum + (diff_i)**2
            mae_sum = mae_sum + abs(diff_i)
            bias = bias + diff_i
        rmse = (rmse_sum/N)**0.5
        mae = mae_sum/N
        bias = bias/N
    else:
        rmse, mae, bias = float("nan"), float("nan"), float("nan")
    with open(filename+".txt","a", encoding="utf-8") as statlog:
        print(f"""{year} Number of matches: {N} \n Root Mean Square Error: {rmse:.3f}
         Mean Average Error: {mae:.3f} \n Bias: {bias:.3f}""", file=statlog)
    return [N, mae, bias, rmse]


def log_error_metrix(dif, filename:str, year:str=""):
    """This functions calculates Mean Average Error (mae) and bias out of the differences
    between satellite derived data (remote) and matching in situ data under the assumption of a
    log-normally distributed difference / error.
    See Seegers et al. (2018).
    The <filename> argument is a prefix for log file produced.
    The <year> argument is necessary for the file name if a specific year is analysed."""
    filename = filename + "_satellite_vs_insitu"
    mae_sum = 0
    bias = 0
    N = len(dif)
    if N > 0:
        for i in range(N):
            log_diff_i = float(dif.iloc[i])
            mae_sum = mae_sum + abs(log_diff_i)
            bias = bias + log_diff_i
        mae = pow(10, mae_sum/N)
        bias = pow(10, bias/N)
    else:
        mae, bias = float("nan"), float("nan")
    with open(filename+".txt","a", encoding="utf-8") as statlog:
        print(f"""{year} Number of matches: {N} \n Mean Average Error: {mae:.3f}
              Bias: {bias:.3f}""", file=statlog)
    return [N, mae, bias, "n.c."]


def violins_plot(sat_data, filename):
    """Creates a violin plot for the chlorophyll-a means, grouped by year.
    <sat_data> is a pandas DataFrame consisting of two columns: chlorophyll-a value and 
    year. <filename> is a prefix for the image file created."""
    y_major = float(input("Specify major y axis ticks (float number) "))
    y_minor = float(input("Specify minor y axis ticks (float number) "))
    fig, ax = plt.subplots(figsize=(8.53, 6.4))
    sns.violinplot(sat_data, x="Year", y="Chl-a_MEAN", inner="point", color="#ffffff",
                   linewidth=0.5, linecolor="k", ax=ax) #color="#4878cf" same blue as in other
                                                        #plots, color="#ffffff" is black/white
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Chlorophyll-a-Konzentration in µg/l")
    ax.set_ylim(bottom=-5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_major))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=y_minor))
    axr = ax.twinx() #additional right hand y axis for improved readability
    axr.set_ylabel(ax.get_ylabel())
    axr.set_ylim(ax.get_ylim())
    axr.yaxis.set_major_locator(ax.yaxis.get_major_locator())
    axr.yaxis.set_minor_locator(ax.yaxis.get_minor_locator())
    ax.grid(axis="y", which="both")
    plt.savefig(filename+"_violins.png", dpi=300, bbox_inches="tight")
    plt.show()


def year_chl_stats(whole_agg):
    """Perform mostly the same operations as in chl_stats(), but for data grouped by year:
    mean, median, standard deviation, 25th quantile, 75th quantile, minimum, maximum.
    A dataframe with those values for each year is returned.
    <whole_agg> is a pandas DataFrame with all scene-specific chlorophyll-a values."""
    grouped = whole_agg.groupby(whole_agg.index.year)
    s_mean = grouped.mean()
    s_med = grouped.median()
    s_std = grouped.std()
    s_25 = grouped.quantile(0.25)
    s_75 = grouped.quantile(0.75)
    s_min = grouped.min()
    s_max = grouped.max()
    stat_chl = pd.concat([s_mean, s_med, s_std, s_25, s_75, s_min, s_max],
                         axis="columns")
    stat_chl.columns = ["Mittel", "Median", "STD", "25th", "75th", "Minimum", "Maximum"]
    return stat_chl


def calc_carlson(conc_chl):
    """Calculates the trophic state index (TSI) after Carlson (1977) from a given
    chlorophyll-a value."""
    tsi_val = 10 * (6 - (2.04 - 0.68 * log(conc_chl))/log(2)) #official formula
    return round(tsi_val, 1)


def calc_lawa(year_conc, year_x):
    """Calculates the trophic state index after Riedmüller et al. (2014)
    from a given pandas DataFrame containing chlorophyll-a concentrations of one year.
    Therefore, the mean chlorophyll-a concentrations from March to October is calculated.
    Afterwards, it is transformed to the "LAWA trophic state index". Input is a data frame with
    date indexes and chlorophyll-a concentration values.
    The <year_> argument is passed to for extracting the seasonal value."""
    season_conc  = year_conc[f"{year_x}-03-01":f"{year_x}-10-31"]
    season_conc = float(season_conc.mean())
    tsi_val = 0.856 * log(season_conc) + 0.560 #official formula
    if tsi_val < 0.1:
        tsi_val = 0.1
    if tsi_val > 5.5:
        tsi_val = 5.5
    return round(tsi_val, 2), round(season_conc, 3) #three decimal digits


def year_indice_stats(year_frame, year):
    """Calculates trophic state indices for a pandas DataFrame (<year_frame>) subset of one year 
    (wrapper for calc_carlson and calc_lawa). The <year> argument is passed to calc_lawa for
    extracting the seasonal values of the respective year."""
    tsi = calc_carlson(year_frame.mean())
    lawa, seas_conc = calc_lawa(year_frame, year)
    scenes = year_frame.count()
    return [tsi, lawa, seas_conc, scenes]


def lawa_timeseries_plot(sat_lawa, insi_lawa, filename, legendloc:str="best"):
    """Plotting a time series for the LAWA index (see Riedmüller et al., 2014) with trophic
    levels differentiated by colour. <sat_lawa> is a pandas DataFrame with two columns: 
    satellite-based trophic state index values (after Riedmüller et al., 2014) and year.
    <insi_lawa> is a pandas DataFrame with two columns: in situ data based trophic state index 
    values (after Riedmüller et al., 2014) and year. <filename> is the prefix for the image file
    created. <legendloc> allows to specify the position of the legend for a uniform design."""
    plt.figure()
    plt.plot(sat_lawa, linestyle="-", linewidth=0, marker="o", color="#00ff00",
              label="LAWA-Indexwert aus Satellitendaten")
    plot_insi = input("Do you want to plot the LAWA index in situ time series too? Press Y. Else, press another key. ")
    if plot_insi.upper() == "Y":
        plt.plot(insi_lawa, linestyle="-", linewidth=0, marker="o", color="#008000",
                label="LAWA-Indexwert aus Insitu-Daten")
    #trophic levels according to Riedmüller et al. (2014):
    plt.fill_between(x=[2015,2025], y1=[0, 0], y2=[1.5, 1.5], color="b", alpha=0.25)
    plt.fill_between(x=[2015,2025], y1=[1.5, 1.5], y2=[2.0, 2.0], color="c", alpha=0.2)
    plt.fill_between(x=[2015,2025], y1=[2.0, 2.0], y2=[2.5, 2.5], color="g", alpha=0.2)
    plt.fill_between(x=[2015,2025], y1=[2.5, 2.5], y2=[3.0, 3.0], color="#69fe66", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[3.0, 3.0], y2=[3.5, 3.5], color="yellow", alpha=0.2)
    plt.fill_between(x=[2015,2025], y1=[3.5, 3.5], y2=[4.0, 4.0], color="#f97306", alpha=0.2)
    plt.fill_between(x=[2015,2025], y1=[4.0, 4.0], y2=[4.5, 4.5], color="red", alpha=0.2)
    plt.fill_between(x=[2015,2025], y1=[4.5,4.5], y2=[5.6, 5.6], color="#970d01", alpha=0.3)
    plt.xlim(2015.5, 2024.5)
    plt.ylim(0, 5.6)
    plt.ylabel("LAWA-Indexwert")
    plt.xlabel("Jahr")
    plt.legend(framealpha=0.7, loc=legendloc)
    plt.grid(which="both", axis="both")
    plt.savefig(filename+"_lawa_timeseries.png", dpi=300, bbox_inches="tight")
    plt.show()


def carlson_timeseries_plot(sat_carlson, insi_carlson, filename, legendloc:str="best"):
    """Plotting a time series for the Trophic State Index after Carlson (1977) with trophic
    levels differentiated by colour. <sat_carlson> is a pandas DataFrame with two columns: 
    satellite-based trophic state index values (after Carlson, 1977) and year.
    <insi_carlson> is a pandas DataFrame with two columns: in situ data based trophic state index 
    values (after Carlson, 1977) and year. <filename> is the prefix for the image file
    created. <legendloc> allows to specify the position of the legend for a uniform design."""
    plt.figure()
    plt.plot(sat_carlson, linestyle="-", linewidth=0, marker="o", color="#00ff00",
              label="TSI nach Carlson aus Satellitendaten")
    plot_insi = input("Do you want to plot the Carlson's TSI in situ time series too? Press Y. Else, press another key. ")
    if plot_insi.upper() == "Y":
        plt.plot(insi_carlson, linestyle="-", linewidth=0, marker="o", color="#008000",
                label="TSI nach Carlson aus Insitu-Daten")
    plt.legend(framealpha=0.7)
    plt.fill_between(x=[2015,2025], y1=[0, 0], y2=[10, 10], color="#0000ff", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[10, 10], y2=[20, 20], color="#4771e9", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[20, 20], y2=[30, 30], color="#29bbec", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[30, 30], y2=[40, 40], color="#21ebac", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[40, 40], y2=[50, 50], color="#69fe66", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[50, 50], y2=[60, 60], color="#b3f836", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[60, 60], y2=[70, 70], color="#f2c93a", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[70,70], y2=[80, 80], color="#f28223", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[80,80], y2=[90, 90], color="#e04009", alpha=0.3)
    plt.fill_between(x=[2015,2025], y1=[90,90], y2=[100, 100], color="#970d01", alpha=0.3)
    plt.xlim(2015.5, 2024.5)
    plt.ylim(0, 100)
    plt.ylabel("Trophic State Index (TSI) nach Carlson")
    plt.xlabel("Jahr")
    plt.legend(framealpha=0.7, loc=legendloc)
    plt.grid(which="both", axis="both")
    plt.savefig(filename+"_carlson_timeseries.png", dpi=300, bbox_inches="tight")
    plt.show()


#%% 1 General data preparation: Loading the data sets (MANDATORY)
if __name__ == "__main__":
    insitu_data = insi_reader()
    sat_data_raw = c2rcc_reader()
    AGGREGATION = input("""Provide your pixel value aggregation method.
    Possible aggregations: Mean, Median, 25th quantile or 75th quantile. """)
    FILEPREFIX = input("Give a filename prefix for the files storing analysis data. ")
    sat_data_agg = scene_statistics(sat_data_raw, FILEPREFIX, AGGREGATION.lower()) #Mean -> mean
    sat_data_agg["Year"] = sat_data_agg.index.year

#%% 2 Analysis of whole time series (OPTIONAL)
    total_timeseries_plot(insitu_data, sat_data_agg[["Chl-a_MEAN", "Chl-a_STD"]], FILEPREFIX,
                          AGGREGATION)

#%% 3 Sliced time series: three periods (2016-2018, 2019-2021, 2022-2024) (OPTIONAL)
    legplot=input("Which plot shall carry a legend: all, 0, 1, 2 or None, default: None ")
    sliced_timeseries_plot(insitu_data, sat_data_agg[["Chl-a_MEAN", "Chl-a_STD"]], FILEPREFIX,
                          AGGREGATION, leg=legplot)

#%% 4 Validation for whole timeseries: comparing Sentinel-2 based an in situ data (MANDATORY)
    all_insitu_match, all_sat_match = matching(insitu_data["Chl-a"], sat_data_agg["Chl-a_MEAN"])
    sat_vs_field(all_insitu_match, all_sat_match, FILEPREFIX) #default plotting
    all_norm_error = match_difference(all_insitu_match, all_sat_match, FILEPREFIX)
    all_log_error = match_log_difference(all_insitu_match, all_sat_match, FILEPREFIX)
    choose_dist_t = input("Which distribution assumed for error metrics? 1=normal; 2=lognormal. ")
    if choose_dist_t == "1":
        norm_error_metrix(all_norm_error, FILEPREFIX)
    elif choose_dist_t == "2":
        log_error_metrix(all_log_error, FILEPREFIX)

#%% 5 Violin plot: comparing the chlorophyll-a value distribution between years (OPTIONAL)
    violins_plot(sat_data_agg[["Chl-a_MEAN", "Year"]], FILEPREFIX)

#%% 6 Yearwise analysis of satellite data: chlorophyll-a statistics and validation (MANDATORY)
    yearwise_stats = year_chl_stats(sat_data_agg["Chl-a_MEAN"]) #make use of efficient groupby func.
    yearwise_indices = []
    yearwise_errors = []
    yearwise_regres = []
    years = list(sat_data_agg.index.year.unique())
    for year_i in years: #iteration over all years
        print(f"Next year: {year_i}")
        year_i_data = sat_data_agg.loc[str(year_i)]
        yearwise_indices.append(year_indice_stats(year_i_data["Chl-a_MEAN"], year_i))
        yea_insitu_match, yea_sat_match = matching(insitu_data["Chl-a"],
                                                   year_i_data["Chl-a_MEAN"], interval=3)
        yearwise_regres.append(sat_vs_field(yea_insitu_match, yea_sat_match, FILEPREFIX,
                                                 year_i))
        yea_norm_error = match_difference(yea_insitu_match, yea_sat_match, FILEPREFIX,str(year_i))
        yea_log_error = match_log_difference(yea_insitu_match, yea_sat_match, FILEPREFIX,
                                             str(year_i))
        choose_dist_y = input("Which distribution can be used? 1=normal; 2=lognormal; If none fits, press another key. ")
        if choose_dist_y == "1":
            yearwise_errors.append(norm_error_metrix(yea_norm_error, FILEPREFIX, str(year_i)))
        elif choose_dist_y == "2":
            yearwise_errors.append(log_error_metrix(yea_log_error, FILEPREFIX, str(year_i)))
        else:
            yearwise_errors.append(["n.c.", "n.c.", "n.c.", "n.c."])
    yearwise_indices = pd.DataFrame(data=yearwise_indices, index=years, columns=["Carlson's TSI",
                                    "LAWA-Index", "Saisonmittel", "Anzahl Szenen"])
    yearwise_errors = pd.DataFrame(data=yearwise_errors, index=years,
                                   columns=["Matches", "MAE", "Bias", "RMSE"])
    yearwise_regres = pd.DataFrame(data=yearwise_regres, index=years,
                                   columns=["R2", "Steigung", "y-Achsenabschnitt"])
    yearwise_stats = pd.concat([yearwise_stats,yearwise_indices,yearwise_errors], axis="columns")
    yearwise_stats.to_csv(FILEPREFIX+"_yearwise_statistics.csv", sep=";", encoding="utf-8")

#%% 7 Yearwise analysis of in situ data: chlorophyll-a based trophic state indices (MANDATORY)
    years = list(insitu_data.index.year.unique())
    yearly_insitu_lawa = []
    yearly_insitu_carlson = []
    for year_i in years: #iteration over years
        year_i_data = insitu_data.loc[str(year_i)]
        yearly_insitu_carlson.append(calc_carlson(year_i_data.mean())) #TSI after Carlson (1977)
        yearly_insitu_lawa.append(calc_lawa(year_i_data["Chl-a"], year_i)) #LAWA index
    yearly_insitu_carlson = pd.DataFrame(data=yearly_insitu_carlson, index=years,
                                           columns=["Carlson's TSI"])
    yearly_insitu_lawa = pd.DataFrame(data=yearly_insitu_lawa, index=years,
                                        columns=["LAWA-Index", "Saisonmittel"])
    yearly_insitu_stats = pd.concat([yearly_insitu_lawa, yearly_insitu_carlson], axis="columns")
    yearly_insitu_stats.to_csv(FILEPREFIX+"_yearly_insitu_indices.csv", sep=";", encoding="utf-8")

#%% 8 Trophic state time series: Indices after Carlson (1977) and Riedmüller et al. (2014) (MANDATORY)
    LEGENDPOS = input("""Where to place the legend for the trophic state time series? Default: 'best'.
\n 'upper left', 'upper right', 'lower left', 'lower right' place the legend at the corresponding 
corner of the plot. 'upper center', 'lower center', 'center left', 'center right' place the 
legend at the center of the corresponding edge of the plot. """).strip("\'")
    if LEGENDPOS == "":
        LEGENDPOS = 'best'
    lawa_timeseries_plot(yearwise_stats["LAWA-Index"], yearly_insitu_stats["LAWA-Index"],
                         FILEPREFIX, LEGENDPOS)
    carlson_timeseries_plot(yearwise_stats["Carlson's TSI"], yearly_insitu_stats["Carlson's TSI"],
                            FILEPREFIX, LEGENDPOS)
