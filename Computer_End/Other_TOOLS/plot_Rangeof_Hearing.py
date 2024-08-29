import matplotlib.pyplot as plt

# Data
ranges = [(2, 200000), (48,85000), (20, 20000), (300,8000)] #, (125,2000)
labels = ['Dolphin', 'Cat', 'Human', 'Red-tailed Hawk'] #, 'Chicken'

# Create a figure and axis
fig, ax = plt.subplots()


# Plot the horizontal bar chart
for idx, (start, end) in enumerate(ranges):
    ax.barh(idx, end - start, left=start, height=0.5, label=labels[idx], zorder=2)

# Set the y-ticks to show the range labels
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)

# Add the legend
ax.legend()

# Set labels for x and y axes
ax.set_xlabel('Hearing Frequency (Hz)')
ax.set_ylabel('Living Beings')

ax.set_xscale('log')
ax.set_xlim(1, 220000)

ax.grid(which='both', linestyle='--', linewidth=0.5, zorder=1)
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth=0.5, zorder=1)

plt.title('Plot of Hearing Frequency Range for Different Animals')

# Show the plot
plt.show()
