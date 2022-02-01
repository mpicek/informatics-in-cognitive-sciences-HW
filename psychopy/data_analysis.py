import numpy as np
from scipy.optimize import curve_fit
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import glob

# global variables describing the experiment and the subjects
subjects = ["m", "e", "t"]
experiments = ["rgb", "contrast"]
num_of_repeats = 30
alpha_significance = 0.2

def find_threshold(f, params):
    """
    Finds the threshold where the subject is 50% likely to say "the right square
    is brighter"
    """
    func = lambda x : f(x, params[0], params[1], params[2]) - 0.5
    threshold = sp.optimize.bisect(func, 0, 1)
    return threshold

def find_confidence_interval(f, params, alpha, threshold):
    """
    Finds the confidence interval of a probability 1 - alpha 
    for the given threshold
    """
    func = lambda x : f(x, params[0], params[1], params[2])
    to_optimize = lambda d: func(threshold + d) - func(threshold - d) - (1 - alpha)
    d = sp.optimize.bisect(to_optimize, 0, 1)
    return d

def psychometric_function(x, alpha, beta):
    """
    Our psychometric function - the logistic function
    """
    return 1. / (1 + np.exp(-(x-alpha)/beta))

def better_pf(x, alpha, beta, l):
    """
    Improved psychometric function - a new parameter is introduced that is
    supposed to account for the subjects mistakes during the experiment
    """
    return (1 - l) * psychometric_function(x, alpha, beta)


def main():
    # for each experiment
    for e in experiments:

        # for each subject
        for sub in subjects:

            experiment_name = "red" if e == "rgb" else "gray"
            main_info_str = "Subject id: " + sub + ", experiment: " + experiment_name

            # open the correct .csv file
            sub_exp = sub + "_" + e
            filename = "data/" + sub_exp + "*" + ".csv"
            assert len(glob.glob(filename)) == 1
            filename = glob.glob(filename)[0]
            df = pd.read_csv(filename)

            # get luminances
            df.sort_values("rgb_luminance")
            luminances = df["rgb_luminance"].unique()
            luminances = np.sort(luminances)
            
            # get only responses that the right square is brighter
            df = df[df["key_response.keys"] == "right"]

            # calculate the percentages of the response "right square is brighter"
            right = []
            for l in luminances:
                s = (df["rgb_luminance"] == l).sum()
                right.append(s / num_of_repeats)
            right = np.array(right)

            # fit the psychometric function
            params, _ = curve_fit(better_pf, luminances, right)

            # find the threshold nad the confidence interval
            threshold = find_threshold(better_pf, params)
            d = find_confidence_interval(better_pf, params, alpha_significance, threshold)

            # print information about the experiment
            threshold_str = "threshold = " + str(round(threshold, 3))
            interval_str = str((1 - alpha_significance)*100) + \
                        "% confidence interval: " + \
                        "[" + str(round(threshold - d, 2)) + ", " + \
                        str(round(threshold + d, 2)) + "]"

            print(main_info_str)
            print("    " + threshold_str)
            print("    " + interval_str)

            # plot the data
            plt.figure(figsize=[7.5, 6.0])
            plt.plot(luminances, right, 'go')
            plt.plot(luminances, better_pf(luminances, params[0], params[1], params[2]))
            
            plt.title(main_info_str + "\n" + threshold_str + "\n" + interval_str)
            plt.xlabel("Luminance")
            plt.ylabel("Percentage of the response \"the right square is brighter\"")
            plt.axvline(x=threshold, color='black', linestyle='--')
            plt.axhline(y=0.5, color='black', linestyle='--')
            plt.axvline(x=threshold + d, color='red', linestyle='--')
            plt.axvline(x=threshold - d, color='red', linestyle='--')
            plt.axvspan(threshold - d, threshold + d, 0, 1, color='red', alpha=0.3)
            
            plt.savefig(sub_exp + ".png")
            plt.show()

if __name__ == "__main__":
    main()