import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
sns.set_theme()

def subj_file(subj_str: str, trial_type: str, data_type: str):
    folder = f"bloodflow_data/{subj_str}"
    filename = None
    files = None
    
    if trial_type in ['baseline', 'base', 'basline']:
        files = [file_name for file_name in os.listdir(folder) if 'baseline' in file_name or 'basline' in file_name or 'base' in file_name]

    elif trial_type in ['post-oc', 'post-occ', 'post-occlusion', 'post']:
        files = [file_name for file_name in os.listdir(folder) if 'post' in file_name]

    else:
        raise ValueError(f"{trial_type} is not an acceptable parameter for trial_type")
    
    if data_type in ['diameter', 'diam']:
        files = [file_name for file_name in files if 'diam' in file_name]
        filename = files[0][:-4]
    elif data_type in ['time', 'time_series', 'time series', 'cyclic'] :
        files = [file_name for file_name in files if 'flow' in file_name or 'doppler' in file_name]
        filename = files[0][:str.index(files[0], '_')]
        
    else:
        raise ValueError(f"{data_type} is not an acceptable parameter for data_type")
    
    return f"{folder}/{filename}"


def get_matlab_data(filename: str):
    matlab_data = sp.io.loadmat(f"{filename}.fig")[
        "hgS_070000"]["children"][0][0][0]["children"][0][0][0]["properties"]["YData"][0][0][0]
    
    matlab_data[matlab_data == -1] = np.NaN
    return matlab_data


def get_raw_data(filename: str):
    raw_data = np.load(f"{filename}.npy")
    return raw_data["diameters"]

def spline_interpolate_nan(array):
    nan_indices = np.isnan(array)
    valid_indices = np.arange(len(array))[~nan_indices]
    valid_values = array[~nan_indices]
    
    spline = sp.interpolate.UnivariateSpline(valid_indices, valid_values, s=0)

    # Interpolate NaN values using the fitted spline
    array[nan_indices] = spline(np.flatnonzero(nan_indices))
    return array

def filter_diameter(diameter_series: np.ndarray, cutoff_frame=False, apply_median_filter=False):

    # print(f"Percent NaN before filtering: {diameter_series.isna().sum() / len(diameter_series) * 100}%")
    
    diameter_array = diameter_series.copy()
    
    # Interpolate NaN values
    interpolated_diameter = spline_interpolate_nan(diameter_array.copy())
    
    # Low-pass and high-pass filter parameters
    low_pass_cutoff_frequency = 0.05
    filter_order = 2
    
    # Apply high-pass filter
    highpass_filter = sp.signal.butter(filter_order, low_pass_cutoff_frequency, 'highpass', output='sos')
    highpass_filtered = sp.signal.sosfilt(highpass_filter, interpolated_diameter)
    
    # Plot high-pass filtered data before thresholding
    
    # Calculate deviation and identify outliers
    deviation_threshold = np.std(highpass_filtered) * 0.9
    outlier_mask = (highpass_filtered > deviation_threshold) | (highpass_filtered < -deviation_threshold)
    
    # Exclude the first and last 30 samples to prevent ringing from high pass filtering
    outlier_mask[:30] = False
    outlier_mask[-30:] = False
    
    # Mark indices to be removed, including 2 indices before and after
    removal_mask = np.zeros_like(highpass_filtered, dtype=bool)
    for index in range(len(outlier_mask)):
        if outlier_mask[index]:
            start = max(0, index - 1)
            end = min(len(outlier_mask), index + 2)
            removal_mask[start:end] = True
    
    # Set outliers to NaN
    diameter_array[removal_mask] = np.NaN
    highpass_filtered[removal_mask] = np.NaN
    
    
    # Calculate median and threshold for further filtering
    median_diameter = np.nanmedian(diameter_array)
    filtering_threshold = np.nanstd(diameter_array) * 2
    extreme_outlier_mask = (diameter_array > median_diameter + filtering_threshold) | (diameter_array < median_diameter - filtering_threshold)
    
    
    # Set extreme outliers to NaN
    diameter_array[extreme_outlier_mask] = np.NaN
    
    # Convert to DataFrame
    diameter_dataframe = pd.DataFrame({"diameter": diameter_array})
    
    # print(f"Percent NaN after filtering: {diameter_dataframe.isna().sum() / len(diameter_dataframe) * 100}%")
    
    
    # Apply cutoff frame if specified
    if cutoff_frame:
        diameter_dataframe = diameter_dataframe.iloc[:cutoff_frame]
    
    # Apply median filter if specified
    if apply_median_filter:
        diameter_dataframe = diameter_dataframe.rolling(window=len(diameter_dataframe) // 50, min_periods=1).median()
    
    return diameter_dataframe

def main():
    files = []
    subj_nums = list(range(5,25))
    subj_name = lambda num: f"ASCI{num:03}"
    
    exclude = []
    for subj_num in subj_nums:
        if subj_num not in exclude:
            files.append(subj_file(subj_name(subj_num), 'baseline', 'diam'))
            files.append(subj_file(subj_name(subj_num), 'post', 'diam'))
            
    subj_nums = list(range(1,15))
    subj_name = lambda num: f"ASCI1{num:02}"
    
    exclude = []
    for subj_num in subj_nums:
        if subj_num not in exclude:
            files.append(subj_file(subj_name(subj_num), 'baseline', 'diam'))
            files.append(subj_file(subj_name(subj_num), 'post', 'diam'))
            
            
    files = np.array(files)
    files = np.unique(files)
    


    bishoy_means = []
    flowave_means = []

    bishoy_stds = []
    flowave_stds = []

    baseline = True
    
    total_frames = 0
    bishoy_frames = []
    flowave_frames = []
    
    for filename in files:
        tag = 'baseline' if baseline else 'post'

        
        diameter_data = get_raw_data(filename)
        diameter = pd.DataFrame()
        diameter['bishoy'] = filter_diameter(diameter_data)
        diameter["flowave"] = get_matlab_data(filename)
        

        bishoy_frames.append( len(diameter['bishoy'].dropna())  / len(diameter)) 
        flowave_frames.append(len(diameter['flowave'].dropna()) / len(diameter)) 
    
        
        diameter.plot()

        bishoy_means.append(diameter["bishoy"].mean())
        flowave_means.append(diameter["flowave"].mean())

        bishoy_stds.append(diameter["bishoy"].std())
        flowave_stds.append(diameter["flowave"].std())
        
        lower_confidence, upper_confidence = sp.stats.norm.interval(0.95, loc = diameter['bishoy'].mean(), scale = diameter['bishoy'].std())
        
        plt.axhline(lower_confidence, color = '#4c72b0', linestyle = '--', alpha = 0.4)
        plt.axhline(upper_confidence, color = '#4c72b0', linestyle = '--', alpha = 0.4)
        
        lower_confidence, upper_confidence = sp.stats.norm.interval(0.95, loc = diameter['flowave'].mean(), scale = diameter['flowave'].std())
 
        plt.axhline(lower_confidence, color = '#dd8452', linestyle = '--', alpha = 0.4)
        plt.axhline(upper_confidence, color = '#dd8452', linestyle = '--', alpha = 0.4)
        plt.title(f"{filename[15:22]} {tag}")
        plt.xlabel("Frame #")
        plt.ylabel("Diameter (px)")


        # plt.savefig(f"bland_altman_plots/{filename[15:22]}_{tag}_diameter_comparison.png")
        # plt.show()
        
        baseline = not baseline


    bishoy_means = np.array(bishoy_means)
    flowave_means = np.array(flowave_means)
    np.nan_to_num(flowave_means, copy=False)
    means = (bishoy_means + flowave_means) / 2
    diffs = bishoy_means - flowave_means

    bias = np.mean(diffs)
    s = np.std(diffs, ddof=1)

    upper_loa = bias + 1.96 * s
    lower_loa = bias - 1.96 * s

    plt.title('Bland-Altman Plot of Bishoy vs. FloWave')
    plt.xlabel('Mean Diameters (px)')
    plt.ylabel('Difference in Diameters (px)')
    plt.axhline(upper_loa, color='red', linestyle='--')
    plt.axhline(lower_loa, color='red', linestyle='--')
    plt.axhline(bias, color='red', linestyle='--')
    plt.axhline(0)
    sns.scatterplot(x=means, y=diffs)
    
    # for i, filename in enumerate(files):
    #     tag = 'b' if i%2==0 else 'p'
        
    #     plt.annotate(f"{filename[15:22]}{tag}", (means[i], diffs[i]))
    
    # plt.savefig("bland_altman_plots/bishoy_flowave_bland_altman.png")
    # plt.show()

    print(f"Total Frames: {total_frames}")
    print(f"Bishoy Frames: {np.mean(bishoy_frames)}")
    print(f"FloWave Frames: {np.mean(flowave_frames)}")
if __name__ == "__main__":
    main()
