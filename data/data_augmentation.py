# Simple script to double the amount of Tasks in a given .csv file
# Initial test of how the setup is working under higher load
import sys
import os
import pandas as pd

if __name__ == "__main__":
    input_file = os.path.join(os.getcwd(), "data", "Task_1.csv")
    output_file = os.path.join(os.getcwd(), "data", "Task_1_doubled.csv")

    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

    df = pd.read_csv(input_file, header=None)
    print("\n", df.head())

    df_copy = df.copy(deep=True)

    df = pd.concat((df, df_copy))
    df.sort_index(inplace=True)

    print("\n", df.head())

    df.to_csv(output_file, header=False, index=False)

    