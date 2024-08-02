import numpy as np
import cv2 as cv
from skimage import filters, morphology, measure, segmentation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import savemat
import pandas as pd
import scipy as sp
import os

def preprocess_opencv(image):
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    contrasted = cv.equalizeHist(src)
    blur = cv.bilateralFilter(contrasted, 9, 75, 75)
    return blur


def preprocess_skimage(image):
    edges = filters.sobel(image)

    thresh = filters.threshold_multiotsu(image)

    markers = np.zeros_like(image)
    markers[image > thresh[1]] = 1
    markers[image < thresh[0]] = 2

    ws = segmentation.watershed(edges, markers)

    ws = morphology.closing(ws)
    ws = morphology.opening(ws)

    labels = measure.label(ws == 2)

    if len(np.unique(labels)) < 4:

        thresh = filters.threshold_otsu(image)
        markers[image > thresh] = 1
        markers[image < thresh] = 2
        ws = segmentation.watershed(edges, markers)

    ws = morphology.closing(ws)
    ws = morphology.opening(ws)
    boundaries = segmentation.find_boundaries(ws, mode='inner')
    return boundaries


def get_pixel_diameter(image, plot=False):
    # Labels image into connected pixels using 2-Connectivity (grouped into any series of pixels touching edges or corners)
    labeled_image = measure.label(image, connectivity=2)
    # Get the height of the image
    height = image.shape[0]

    # Compute properties
    regions = measure.regionprops(labeled_image)
    filtered_labels = []
    filtered_image = labeled_image.copy()

    for i in range(len(regions)):
        if regions[i].axis_major_length > height//1.1:
            filtered_labels.append(regions[i].label)
        else:
            filtered_image[filtered_image == regions[i].label] = 0

    if len(filtered_labels) < 2:
        return np.NaN
    elif len(filtered_labels) > 2:
        filtered_labels = np.array(filtered_labels)
        residuals = np.empty_like(filtered_labels)
        for i in range(len(filtered_labels)):
            label_mask = labeled_image == filtered_labels[i]
            y, x = np.where(label_mask)
            residuals[i] = np.polyfit(x, y, 1, full=True)[1].item()
        filtered_labels = filtered_labels[np.argsort(residuals)[:2]]

    y_mean = np.empty(2)
    x_mean = np.empty(2)
    xy_mean = np.empty(2)
    x_squared_mean = np.empty(2)
    for i in range(2):
        label_mask = labeled_image == filtered_labels[i]
        y, x = np.where(label_mask)
        y_mean[i] = y.mean()
        x_mean[i] = x.mean()
        xy_mean[i] = (x*y).mean()
        x_squared_mean[i] = (x**2).mean()

    slope = (xy_mean[0] + xy_mean[1] - x_mean[0]*y_mean[0] - x_mean[1]*y_mean[1]
             ) / (x_squared_mean[0]+x_squared_mean[1] - x_mean[0]**2 - x_mean[1]**2)
    intercept = y_mean - slope * x_mean
    if plot:
        for inter in intercept:
            plt.axline((0, inter), slope=slope, color="red")

        plt.imshow(filtered_image, cmap = 'viridis')
        plt.show()

    diameter = np.abs(intercept[0] - intercept[1]) / np.sqrt(1+slope**2)
    return diameter


def get_pixel_diameter_from_frame(frame, r, plot=False):
    global test_num
    cropped_frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    image = preprocess_opencv(cropped_frame)
    image = image.T
    
    if plot:
        plt.imshow(image, cmap = 'viridis')
        plt.show()

    image = preprocess_skimage(image)

    diameter = get_pixel_diameter(image, plot)

    return diameter


def get_px_to_cm(frame):
    print("Select ROI that surrounds the reference scale")
    r = cv.selectROI(frame)
    px_to_cm = float(input("How many centimeters is this? "))
    px_to_cm /= r[3]

    return px_to_cm


def get_diameter_roi(frame):
    print("Select ROI for Diameter Calculations")
    r = cv.selectROI(frame)

    diameter_init = get_pixel_diameter_from_frame(frame, r, plot=True)

    print(f"Diameter calculated to be {diameter_init} pixels")
    print("Press ENTER to continue calculations. Press R to repeat ROI selection. Press any other key to exit.")
    key = cv.waitKey(0)

    return r, key


def calculate_all_diameters(capture: cv.VideoCapture, r):
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) - 1
    print(f"Frame Count: {frame_count}")

    capture.set(cv.CAP_PROP_POS_FRAMES, 0)

    diameters_pixel = np.empty(frame_count+1)
    index = 0

    ret, frame = capture.read()

    while (ret):
        diameters_pixel[index] = get_pixel_diameter_from_frame(frame, r)
        print(f"Frame {index}/{frame_count}: {diameters_pixel[index]}")
        ret, frame = capture.read()
        index += 1

    capture.release()

    return diameters_pixel

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
    
    print(f"Percent NaN after filtering: {diameter_dataframe.isna().sum() / len(diameter_dataframe) * 100}%")
    
    
    # Apply cutoff frame if specified
    if cutoff_frame:
        diameter_dataframe = diameter_dataframe.iloc[:cutoff_frame]
    
    # Apply median filter if specified
    if apply_median_filter:
        diameter_dataframe = diameter_dataframe.rolling(window=len(diameter_dataframe) // 50, min_periods=1).median()
    
    return diameter_dataframe


def main():
    filename = "ASCI004 - post-occ  long diam 20200911125234597"
    folder = "raw_video"
    capture = cv.VideoCapture(f"{folder}/{filename}.avi")

    ret, frame = capture.read()
    save_data = False

    if not ret:
        exit(f"File: \"{folder}/{filename}.avi\" not found.")

    px_to_cm = get_px_to_cm(frame)
    print(f"Pixel to cm: {px_to_cm}\n")

    # Select ROI
    r, key = get_diameter_roi(frame)
    cv.destroyWindow("ROI selector")

    while (key == ord("\r") or key == ord("\n") or key == ord("r") or key == ord("R")):

        if key == ord("\r") or key == ord("\n"):

            diameters_pixel = calculate_all_diameters(capture, r)

            sns.set_theme()

            # Filtering and Plotting Data
            diameter = filter_diameter(diameters_pixel)
            sns.lineplot(x = np.arange(len(diameters_pixel)), y = diameters_pixel, label = "raw", alpha = 0.4)
            sns.lineplot(x= diameter.index.values, y = diameter["diameter"], label = "after filtering")
            

            print(f"\nMean Diameter: {diameter["diameter"].mean()} px")
            print(f"Median Diameter: {diameter["diameter"].median()} px")

            print(f"\nMean Diameter: {diameter["diameter"].mean() * px_to_cm} cm")
            print(f"Median Diameter: {diameter["diameter"].median() * px_to_cm} cm")

            print(f"Px to Cm: {px_to_cm}")
            plt.show()

            redo = True if input(
                "\nRedo ROI and diameter calculations? y/n: ") == 'y' else False

            if redo:
                key = ord('r')
                capture = cv.VideoCapture(f"raw_video/{filename}.avi")
                continue

            save_data = True if input("\nSave data? y/n: ") == 'y' else False

            if save_data:
                os.makedirs(f"bloodflow_data/{filename[:7]}/", exist_ok=True)
                
                print(f"Writing raw data to bloodflow_data/{filename[:7]}/{filename}.npy...")
                with open(f"bloodflow_data/{filename[:7]}/{filename}.npy", "wb") as f:
                    np.savez(f, diameters=diameters_pixel, roi=r, px_to_cm=px_to_cm)

                print("Done writing.")

                print(
                    f"Writing raw data to bloodflow_data/{filename[:7]}/{filename}.mat...")

                savemat(f"bloodflow_data/{filename[:7]}/{filename}.mat",
                        {"roi": r,
                        "diameters": diameters_pixel,
                         "px_to_cm": px_to_cm})

                print("Done writing.")

            break

        if key == ord("r") or key == ord("R"):
            # Select ROI
            r, key = get_diameter_roi(frame)
            cv.destroyWindow("ROI selector")

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()