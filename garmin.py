import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

df = pd.read_csv("Activities.csv")
df.head()

# Keep only running, hiking, treadmill running, mountaineering, and the box step
# test. Keep only vars related to cardio performance and time.
df = df[
    [
        "Date",
        "Activity Type",
        "Distance",
        "Time",
        "Avg HR",
        "Max HR",
        "Avg Speed",
    ]
][
    (df["Activity Type"] == "Running")
    | (df["Activity Type"] == "Hiking")
    | (df["Activity Type"] == "Treadmill Running")
    | (df["Activity Type"] == "Mountaineering")
    | (df["Activity Type"] == "Cardio")
]

df.reset_index(drop=True, inplace=True)
df.head()

df.info()

# Trim the time off of the date
df["Date"] = df["Date"].str[:10]
df.head()

# Convert string to datetime.
df["Date"] = pd.to_datetime(df["Date"])
df.head()
df.info()

# Per manual inspection, a few times have decimals after the seconds. One
# example is below.
df["Time"].loc[65]

# To remove these, split on the decimal delimiter and keep the left-most part of
# the string.
df["Time"] = df["Time"].str.split(".").str[0]

df["Time"].head()

# Below splits the Time var at the colon, converts the left-most 0th index of
# the split object into an integer, multiplies by 60 to convert hours to minutes
# then adds the minutes after also converting those to an integer. Note that the
# seconds will need to be summed up and convered to minutes later.
df["Cum_Min"] = df["Time"].apply(
    lambda x: (int(x.split(":")[0]) * 60) + int(x.split(":")[1])
)

# Create a variable with the seconds
df["Sec"] = df["Time"].apply(lambda x: int(x.split(":")[2]))

# Calculate cumluative seconds for each month.
df["Cum_Sec_Monthly"] = df.groupby([df["Date"].dt.month, df["Date"].dt.year])[
    "Sec"
].transform(lambda x: x.sum())

# Calculate the cumulative minutes for each month.
df["Cum_Min_Monthly"] = df.groupby([df["Date"].dt.month, df["Date"].dt.year])[
    "Cum_Min"
].transform(lambda x: x.sum())

df.head()

# Add the minute portion of the cumulative monthly seconds to the cumulative
# minutes in the month
df["Cum_Min_Monthly"] = (
    df["Cum_Min_Monthly"] + divmod(df["Cum_Sec_Monthly"], 60)[0]
)

# Convert the cumulative minutes and minute portions of total seconds into
# hours. Note that the second remainders from the total montly seconds after
# converting to minutes are not included in the calculations, but these are
# immaterial.
df["Cum_Hr_Monthly"] = df["Cum_Min_Monthly"] / 60

# Calculate cumulative miles run for each month.
df["Cum_Miles_Monthly"] = df.groupby(
    [df["Date"].dt.month, df["Date"].dt.year]
)["Distance"].transform(lambda x: x.sum())

# Create base run paces and heart rates for runs with heart rates under 146. But
# first, deal with weirdo values for Avg HR set to missing by Garmin.
df["Activity Type"].value_counts()

df[df["Avg HR"] == "--"]

df.loc[df["Avg HR"] == "--", "Avg HR"] = np.nan
df["Avg HR"] = df["Avg HR"].astype(float)

# Use 146 average heart rate as the cutoff for base training. Per Steve House's
# "Training for the New Alpinism", Zone 1 is approximately 75% of maximum HR.
df["Base_Run_Pace"] = df["Avg Speed"][
    (df["Activity Type"] == "Running") & (df["Avg HR"] < 146)
]
df["Base_Run_HR"] = df["Avg HR"][
    (df["Activity Type"] == "Running") & (df["Avg HR"] < 146)
]

# Create a variable with the total seconds of the Base_Run_Pace. This is to
# facilitate calulating average seconds for the run later. Trying to average a
# bunch of paces converted to time stamps seemed weird.
df["Base_Run_Pace_Sec"] = df["Base_Run_Pace"][
    df["Base_Run_Pace"].notnull()
].apply(lambda x: (int(x.split(":")[0]) * 60) + int(x.split(":")[1]))


df.head()

# Create fast run paces and heart rates for runs with heart rates over 146.
df["Fast_Run_Pace"] = df["Avg Speed"][
    (df["Activity Type"] == "Running") & (df["Avg HR"] >= 146)
]
df["Fast_Run_HR"] = df["Avg HR"][
    (df["Activity Type"] == "Running") & (df["Avg HR"] >= 146)
]

# Create a variable with the total seconds of the Fast_Run_Pace. Same reasoning
# as above.
df["Fast_Run_Pace_Sec"] = df["Fast_Run_Pace"][
    df["Fast_Run_Pace"].notnull()
].apply(lambda x: (int(x.split(":")[0]) * 60) + int(x.split(":")[1]))


df.head()

# Create monthly averages for base run heart rates, base run speeds, fast run
# heart rates, and fast runs speeds.

df["Base_Run_Pace_Avg_Sec"] = df.groupby(
    [df["Date"].dt.month, df["Date"].dt.year]
)["Base_Run_Pace_Sec"].transform(lambda x: x.mean())

df["Base_Run_HR_Avg"] = df.groupby([df["Date"].dt.month, df["Date"].dt.year])[
    "Base_Run_HR"
].transform(lambda x: x.mean())

df["Fast_Run_Pace_Avg_Sec"] = df.groupby(
    [df["Date"].dt.month, df["Date"].dt.year]
)["Fast_Run_Pace_Sec"].transform(lambda x: x.mean())

df["Fast_Run_HR_Avg"] = df.groupby([df["Date"].dt.month, df["Date"].dt.year])[
    "Fast_Run_HR"
].transform(lambda x: x.mean())

df.head()

# Convert the monthly averages back into minutes per mile. An 8:15 pace would be
# converted as follows without divmod: 8 x 60 = 480 + 15= 495 to get to total
# seconds. To get back to a pace, you would take 495/ 60 = 8.25 but note the .25
# is in fractions of a minute. A minute is 60 seconds, so .25 x 60 = 15. You
# ened up adding 15 seconds to the whole number portion, which was 8, to get
# back to 8:15. See the divmod examples below.

# I did one run in Feb. that was a fast run. It was the half marathon at a 9:01
# pace.
divmod(541, 60)

# Per above, you can see the 9 is the minutes and the 1 is the seconds. Next,
# try again for the 8:15 pace discussed above.
divmod(495, 60)

# Again, divmod returns the tuple with the minutes and seconds but we don't have
# to do the .25 times 60 seconds conversion when using divmod.
df.head(10)


# Convert back to monthly average min:sec pace. To do this with apply, I am
# retaining only the integer portion of the minutes and seconds, otherwise a
# 10:53 time with fractional seconds will look like 10.0:53.5
df["Base_Run_Pace_Avg"] = df["Base_Run_Pace_Avg_Sec"][
    df["Base_Run_Pace_Avg_Sec"].notnull()
].apply(
    lambda x: str(int(divmod(x, 60)[0])) + ":" + str(int(divmod(x, 60)[1]))
)

df["Fast_Run_Pace_Avg"] = df["Fast_Run_Pace_Avg_Sec"][
    df["Fast_Run_Pace_Avg_Sec"].notnull()
].apply(
    lambda x: str(int(divmod(x, 60)[0])) + ":" + str(int(divmod(x, 60)[1]))
)


df.head(10)
df.tail(20)

# The 9:01 pace for 2/2022 was input as 9:1 after divmod. Any 0-9 seconds will
# show up like this, and longer paces (e.g., 9:12) will show up correctly. To
# fix this, I will insert a 0 is there is only one spot to the right of the
# colon.


def insert_zero(col):
    # Skip the NaN's
    if type(col) != float:
        # If the length of the right-most sting is one
        if len(col.split(":")[1]) == 1:
            # return the left-most string, append a :0, then append the
            # right-most string
            return col.split(":")[0] + ":0" + col.split(":")[1]
        else:
            # if the above conditions aren't met, then just return the value in
            # the column as is.
            return col


# Overwrite the base and fast runs with the above modification.
df["Fast_Run_Pace_Avg"] = df["Fast_Run_Pace_Avg"].apply(insert_zero)
df["Base_Run_Pace_Avg"] = df["Base_Run_Pace_Avg"].apply(insert_zero)

df.head()

# Round to two digits

df = df.round(2)

df_agg = (
    df.groupby([df["Date"].dt.month, df["Date"].dt.year])
    .first()
    .reset_index(level=[0, 1], drop=True)
)

df_agg.shape

df_agg

vars_list = [
    "Date",
    "Cum_Miles_Monthly",
    "Cum_Hr_Monthly",
    "Base_Run_HR_Avg",
    "Base_Run_Pace_Avg",
    "Base_Run_Pace_Avg_Sec",
    "Fast_Run_HR_Avg",
    "Fast_Run_Pace_Avg",
    "Fast_Run_Pace_Avg_Sec",
]

df_agg[vars_list]

df_agg = df_agg[vars_list].sort_values(by="Date", ascending=True)


df_agg.reset_index(drop=True, inplace=True)
df_agg

df_agg.to_excel("python_processed.xlsx", index=False)

# Limit data to June 2020 and later.
df_agg = df_agg[df_agg["Date"] >= "2020-09-01"]

plt.bar(df_agg["Date"], df_agg["Cum_Miles_Monthly"], width=26.0)
plt.xticks(rotation="vertical")
plt.xlabel("Calendar Month")
plt.ylabel("Total Miles")
plt.title("Monthly Mileage")
plt.tight_layout()
plt.savefig("Miles.pdf", bbox_inches="tight")
plt.show()

df.head()

# Find minimum pace, divided by 10 to overlay on graph well.
df_agg["Base_Run_Pace_Avg_Sec"].min()
# The 9.241667 is 9 minutes and .241667 of a minute, rounds to 15 seconds.

min_pace = df_agg["Base_Run_Pace_Avg_Sec"].min() / 60
# Find date of minimum pace
min_pace_date = df_agg["Date"][
    df_agg["Base_Run_Pace_Avg_Sec"] == df_agg["Base_Run_Pace_Avg_Sec"].min()
]
min_pace
min_pace_date


# Repeat for maximum pace
max_pace = df_agg["Base_Run_Pace_Avg_Sec"].max() / 60

# Find date of minimum pace
max_pace_date = df_agg["Date"][
    df_agg["Base_Run_Pace_Avg_Sec"] == df_agg["Base_Run_Pace_Avg_Sec"].max()
]
max_pace
max_pace_date

# Repeat for another data point near the middle
other_pace = (
    df_agg["Base_Run_Pace_Avg_Sec"][df_agg["Date"] == "2021-09-16"] / 60
)
other_date = "2021-09-16"


# Create standalone graph of the base paces since this will be easier to see, as
# opposed to plotting the points over the bars with the axis length of 70ish
# makes the trends harder to see.
plt.scatter(df_agg["Date"], df_agg["Base_Run_Pace_Avg_Sec"] / 60)
plt.text(
    min_pace_date, min_pace + 0.10, "9:57 Pace", horizontalalignment="center"
)
plt.text(
    max_pace_date, max_pace - 0.25, "12:34 Pace", horizontalalignment="center"
)
plt.text(
    other_date, other_pace - 0.20, "10:40 Pace", horizontalalignment="center"
)
plt.xticks(rotation="vertical")
plt.xlabel("Calendar Month")
plt.ylabel("Base Pace- Minutes per Mile")
plt.title("Monthly Pace")
plt.tight_layout()
plt.savefig("Pace.pdf")
plt.show()

# Combined the distance bars and the paces.
plt.bar(df_agg["Date"], df_agg["Cum_Miles_Monthly"], width=26.0, zorder=1)
plt.scatter(df_agg["Date"], df_agg["Base_Run_Pace_Avg_Sec"] / 60, zorder=2)
plt.text(
    min_pace_date, min_pace + 10, "9:57 Pace", horizontalalignment="center"
)
plt.vlines(x=min_pace_date, ymin=min_pace, ymax=min_pace + 10, color="black")
plt.text(
    max_pace_date, max_pace + 15, "12:34 Pace", horizontalalignment="right"
)
plt.vlines(x=max_pace_date, ymin=max_pace, ymax=max_pace + 15, color="black")
plt.text(
    other_date, other_pace + 30, "10:40 Pace", horizontalalignment="center"
)
plt.vlines(x=other_date, ymin=other_pace, ymax=other_pace + 30, color="black")
plt.xticks(rotation="vertical")
plt.xlabel("Time Period")
plt.ylabel("Total Miles")
plt.title("Monthly Mileage and Base Run Pace")
plt.tight_layout()
plt.savefig("Pace_and_Miles.pdf")
plt.show()
