#####################
# CS 181, Spring 2020
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
from CS_181_hw1 import calculate_loss

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# Create other basis functions

# initialize with ones
def basis_1(vector):
    X1 = np.vstack((np.ones(vector.shape))).T
    for j in range(1, 6):
        X1 = np.vstack((X1, vector**j))
    return X1
X1 = basis_1(years).T

def basis_2(vector):
    X2 = np.vstack((np.ones(vector.shape))).T
    for j in range(1960, 2015, 5):
        X2 = np.vstack((X2, np.exp(-(vector-j)**2/25)))
    return X2
X2 = basis_2(years).T

def basis_3(vector):
    X3 = np.vstack((np.ones(vector.shape))).T
    for j in range (1, 6):
        X3 = np.vstack((X3, np.cos(vector/j)))
    return X3
X3 = basis_3(years).T

def basis_4(vector):
    X4 = np.vstack((np.ones(vector.shape))).T
    for j in range (1, 26):
        X4 = np.vstack((X4, np.cos(vector/j)))
    return X4
X4 = basis_4(years).T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
w1 = np.linalg.solve(np.dot(X1.T, X1) , np.dot(X1.T, Y))
w2 = np.linalg.solve(np.dot(X2.T, X2) , np.dot(X2.T, Y))
w3 = np.linalg.solve(np.dot(X3.T, X3) , np.dot(X3.T, Y))
w4 = np.linalg.solve(np.dot(X4.T, X4) , np.dot(X4.T, Y))

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))

grid_X1 = basis_1(grid_years)
grid_X2 = basis_2(grid_years)
grid_X3 = basis_3(grid_years)
grid_X4 = basis_4(grid_years)

grid_Yhat = np.dot(grid_X.T, w)
grid_Y1hat = np.dot(grid_X1.T, w1)
grid_Y2hat = np.dot(grid_X2.T, w2)
grid_Y3hat = np.dot(grid_X3.T, w3)
grid_Y4hat = np.dot(grid_X4.T, w4)

# calculate residual sum of squares error
# go through and add predictions to list (every 5 years since 1960)
# use function from other thingy
preds1 = np.dot(X1, w1)
preds2 = np.dot(X2, w2)
preds3 = np.dot(X3, w3)
preds4 = np.dot(X4, w4)

print(calculate_loss(preds1, Y))
print(calculate_loss(preds2, Y))
print(calculate_loss(preds3, Y))
print(calculate_loss(preds4, Y))



# Plot the data and the regression line.
plt.figure(4)
plt.plot(years, republican_counts, 'o', grid_years, grid_Y1hat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('4.png')
plt.figure(5)
plt.plot(years, republican_counts, 'o', grid_years, grid_Y2hat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('5.png')
plt.figure(6)
plt.plot(years, republican_counts, 'o', grid_years, grid_Y3hat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('6.png')
plt.figure(7)
plt.plot(years, republican_counts, 'o', grid_years, grid_Y4hat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('7.png')
plt.show()


stop_index = 0
for year in years:
    if (year > last_year):
        break
    stop_index += 1


sunspots_truncated = sunspot_counts[0:stop_index]
republicans_truncated = republican_counts[0:stop_index]
sunspots_a = basis_1(sunspots_truncated).T
sunspots_c = basis_3(sunspots_truncated).T
sunspots_d = basis_4(sunspots_truncated).T

w_a = np.linalg.solve(np.dot(sunspots_a.T, sunspots_a) , np.dot(sunspots_a.T, republicans_truncated))
w_c = np.linalg.solve(np.dot(sunspots_c.T, sunspots_c) , np.dot(sunspots_c.T, republicans_truncated))
w_d = np.linalg.solve(np.dot(sunspots_d.T, sunspots_d) , np.dot(sunspots_d.T, republicans_truncated))

grid_sunspots = np.linspace(0, 200, 200)

grid_sunspots_a = basis_1(grid_sunspots)
grid_sunspots_c = basis_3(grid_sunspots)
grid_sunspots_d = basis_4(grid_sunspots)

grid_Yahat = np.dot(grid_sunspots_a.T, w_a)
grid_Ychat = np.dot(grid_sunspots_c.T, w_c)
grid_Ydhat = np.dot(grid_sunspots_d.T, w_d)

plt.figure(8)
plt.plot(sunspots_truncated, republicans_truncated, 'o', grid_sunspots, grid_Yahat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('8.png')
plt.figure(9)
plt.plot(sunspots_truncated, republicans_truncated, 'o', grid_sunspots, grid_Ychat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('9.png')
plt.figure(10)
plt.plot(sunspots_truncated, republicans_truncated, 'o', grid_sunspots, grid_Ydhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.savefig('10.png')
plt.show()

preds_a = np.dot(sunspots_a, w_a)
preds_c = np.dot(sunspots_c, w_c)
preds_d = np.dot(sunspots_d, w_d)

print(calculate_loss(preds_a, republicans_truncated))
print(calculate_loss(preds_c, republicans_truncated))
print(calculate_loss(preds_d, republicans_truncated))
