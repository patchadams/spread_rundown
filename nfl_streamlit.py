import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, explained_variance_score
from run_test_investment import run_test_investment  # Import the function from the separate file

# Load your data
df = pd.read_csv('model_target.csv')
predictions = pd.read_csv('this_weeks_predictions.csv')

# Streamlit Title
st.title("PATCH ADAMS NFL RUNDOWN")

# Display the DataFrame
st.write("WEEK 5:")
st.dataframe(predictions)



st.write("TEST SUMMARY:")

# User input to select the number of bins
bins = st.slider("Select number of bins for histograms and heatmap", key='slider_bufff2', min_value=2, max_value=9, value=5)

# Define tighter preset bin ranges relevant to NFL scores
if bins == 2:
    bin_ranges = [0]  # For 2 bins, divide between away and home wins
    x_labels = ['away_win', 'home_win']
    y_labels = x_labels
else:
    bin_ranges = [-14, -10, -7, -3, 0, 3, 7, 10, 14][ 5- int(bins/2)  -1 : 5 +int(bins/2)]  # Include 0 and extend beyond -14 and 14
    x_labels = [-np.inf] + [f'{bin_ranges[i]} to {bin_ranges[i+1]}' for i in range(len(bin_ranges)-1) ] + [np.inf]

    x_labels[0] = f'<{bin_ranges[0]}'  # Label for outliers on the left side
    x_labels[-1] = f'>{bin_ranges[-1]}'  # Label for outliers on the right side
    y_labels = x_labels

# Create the dashboard layout with Streamlit
fig, axs = plt.subplots(3, 2, figsize=(14, 10))

# 1. Line plot of predicted vs spread
axs[0, 0].plot(df['spread'], label="spread")
axs[0, 0].plot(df['predicted'], label="predicted")
axs[0, 0].legend()
axs[0, 0].set_title('Line plot: predicted vs spread')

# 2. Histogram of the difference between predicted and spread
diff = df['spread'] - df['predicted']
axs[0, 1].hist(diff,bins=25,weights=np.ones(len(diff)) / len(diff))
axs[0, 1].set_title('Histogram of Errors (spread - predicted)')
axs[0, 1].set_ylabel('Percentage (%)')

# 3. Scatter plot comparing predicted and spread
axs[1, 0].scatter( df['predicted'], df['spread'],alpha=0.2, color='blue')
axs[1, 0].set_title('Scatter plot: predicted vs spread')
axs[1, 0].set_xlabel('predicted')
axs[1, 0].set_ylabel('spread')

binned_actual = np.digitize(df['spread'].values, bin_ranges)
binned_predicted = np.digitize(df['predicted'].values, bin_ranges)
# 4. Heatmap: 2D histogram of actual (spread) on y-axis vs predicted on x-axis
conf_matrix = confusion_matrix(binned_actual, binned_predicted, labels=range(len(x_labels)))
sns.heatmap(conf_matrix, ax=axs[1, 1], cmap="Blues", cbar=True, annot=True, fmt=".0f",
            xticklabels=x_labels, yticklabels=y_labels)
axs[1, 1].set_title('2D Histogram Heatmap: actual (spread) vs predicted')
axs[1, 1].set_xlabel('predicted bins')
axs[1, 1].set_ylabel('actual (spread) bins')


cm_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
# Calculate correct and incorrect percentages for each class
correct_percentages = np.diag(cm_percentage) * 100  # True Positives
incorrect_percentages = 100 - correct_percentages  # False Positives + False Negatives
classes = x_labels  # Adjust this if you have specific class names
bars_correct = axs[2, 0].bar(classes, correct_percentages, color='green', label='Correct')
axs[2, 0].set_title('Accuracy per Bin')
bars_incorrect = axs[2, 0].bar(classes, incorrect_percentages, color='red', bottom=correct_percentages, label='Incorrect')
# Annotating the correct percentages
for bar in bars_correct:
    height = bar.get_height()
    axs[2, 0].annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
precision = np.nan_to_num(TP / (TP + FP)) * 100    
axs[2, 1].bar(classes, precision)
axs[2, 1].set_title('Precision per Bin')
for i, accuracy in enumerate(precision):
    axs[2, 1].text(i, accuracy , f'{accuracy:.2f}%', ha = 'center')
# Adjust layout
plt.tight_layout()

# Display the full dashboard
st.pyplot(fig)


import nfl_streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from run_test_investment import run_test_investment  # Import the function from the separate file



# Streamlit Title
st.title("Investment Simulator with Spread Buffer")

# Buffer slider input
buffer = st.slider("Select Spread Buffer", key='spread_bufff',min_value=0.0, max_value=10.0, step=0.1, value=5.0)

# Bet percentage input slider
bet_percentage = st.slider("Select Bet Percentage", key='bet_pct', min_value=0.01, max_value=0.20, step=0.01, value=0.07)

# Run the simulation based on buffer and bet percentage input
history, winners, losers, pushes, pred_winner, wrong, bets = run_test_investment(df, buffer, bet_percentage)

# Plot the balance history
st.write("Balance History")
fig, ax = plt.subplots()
ax.plot(history)
ax.set_xlabel("Iteration")
ax.set_ylabel("Balance")
st.pyplot(fig)

# Display simulation results
st.write(f"Winners: {winners}")
st.write(f"Losers: {losers}")
st.write(f"Pushes: {pushes}")
st.write(f"Money Line Winners: {pred_winner}")
st.write(f"Money Line Losers: {wrong}")




st.write("Bets Taken")
st.dataframe(bets)
