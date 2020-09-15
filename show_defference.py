import pandas as pd

def show_diff(series1, series2):
    if len(series1) != len(series2):
        print(series1.name + ',' + series2.name, "different length")
    else:
        for i in range(0, len(series2)):
            if series1[i] != series2[i]:
                print(series1[i], series2[i])


if __name__ == '__main__':
    series1 = pd.read_csv('../strcol/e-commerce-simple-eda/Division Name/pair/2.csv')

    show_diff(series1['before'],series1['after'])