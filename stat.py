import math
from re import X
import statistics
import numpy as np
import scipy.stats
import pandas as pd

####  1
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
x_with_nan


math.isnan(np.nan), np.isnan(math.nan)

y_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
math.isnan(y_with_nan[3]), np.isnan(y_with_nan[3])

y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y
y_with_nan
z
z_with_nan


##### 2
mean_ = sum(x) / len(x)
mean_

mean_ = y.mean()
mean_

np.mean(y_with_nan)
y_with_nan.mean()

np.nanmean(y_with_nan)

mean_ = z.mean()
mean_

z_with_nan.mean()

0.2 * 2 + 0.5 * 4 + 0.3 * 8

x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean

wmean = np.average(z, weights=w)
wmean

(w * y).sum() / w.sum()


w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)

np.average(z_with_nan, weights=w)


### Harmonic Mean

hmean = len(x) / sum(1 / item for item in x)
hmean

hmean = statistics.harmonic_mean(x)
hmean


##The example above shows one implementation of statistics.harmonic_mean(). 
# If you have a nan value in a dataset, then it’ll return nan. If there’s at least one 0, then it’ll return 0. 
# If you provide at least one negative number, then you’ll get statistics.StatisticsError:

statistics.harmonic_mean(x_with_nan)

statistics.harmonic_mean([1, 0, 2])

statistics.harmonic_mean([1, 2, -2])  # Raises StatisticsError

scipy.stats.hmean(y)
scipy.stats.hmean(z)

### Geometric Mean
gmean = 1
for item in x:
    gmean *= item
    
gmean **= 1 / len(x)
gmean

###  converts all values to floating-point numbers and returns their geometric mean:
gmean = statistics.geometric_mean(x)
gmean

#### f you pass data with nan values, then statistics.geometric_mean() will behave like most similar functions and return nan:
gmean = statistics.geometric_mean(x_with_nan)
gmean

scipy.stats.gmean(y)

scipy.stats.gmean(z)


### Median
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
median_ = 0.5 * (x_ord[index-1] + x_ord[index])

median_


### median_low() and median_high() are two more functions related to the median 
# in the Python statistics library. They always return an element from the dataset:

statistics.median_low(x[:-1])

statistics.median_high(x[:-1])

statistics.median(x_with_nan)

statistics.median_low(x_with_nan)

statistics.median_high(x_with_nan)

median_ = np.median(y)
median_

median_ = np.median(y[:-1])
median_

np.nanmedian(y_with_nan)

np.nanmedian(y_with_nan[:-1])

z.median()

z_with_nan.median()


#### Mode
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
mode_

### mode() returned a single value, while multimode() returned the list that contains the result.
v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v)

### statistics.mode() and statistics.multimode() handle nan values as regular values and can return nan as the modal value

statistics.mode([2, math.nan, 2])

statistics.multimode([2, math.nan, 2])

statistics.mode([2, math.nan, 0, math.nan, 5])

statistics.multimode([2, math.nan, 0, math.nan, 5])


u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_
ModeResult(mode=array([2]), count=array([2]))
mode_ = scipy.stats.mode(v)
mode_


###You can get the mode and its number of occurrences as NumPy arrays with dot notation:
mode_.mode

mode_.count


u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()


v.mode()


w.mode()



### Measures of Variability
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

var_ = statistics.variance(x)
var_

###If you have nan values among your data, then statistics.variance() will return nan:

statistics.variance(x_with_nan)

var_ = np.var(y, ddof=1)
var_
var_ = y.var(ddof=1)
var_

##If you have nan values in the dataset, then np.var() and .var() will return nan:

np.var(y_with_nan, ddof=1)
y_with_nan.var(ddof=1)

np.nanvar(y_with_nan, ddof=1)

###pd.Series objects have the method .var() that skips nan values by default:

z.var(ddof=1)
z_with_nan.var(ddof=1)


#### Standard Deviation
std_ = var_ ** 0.5
std_


std_ = statistics.stdev(x)
std_

np.std(y, ddof=1)
y.std(ddof=1)
np.std(y_with_nan, ddof=1)

y_with_nan.std(ddof=1)

np.nanstd(y_with_nan, ddof=1)



z.std(ddof=1)

z_with_nan.std(ddof=1)

######################### Skewness
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
        * n / ((n - 1) * (n - 2) * std_**3))
skew_


####You can also calculate the sample skewness with scipy.stats.skew():

y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
scipy.stats.skew(y_with_nan, bias=False)

z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()
z_with_nan.skew()


#### Percentiles
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)

statistics.quantiles(x, n=4, method='inclusive')

####You can also use np.percentile() to determine any sample percentile in your dataset. 
# For example, this is how you can find the 5th and 95th percentiles:

y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)

np.percentile(y, [25, 50, 75])
array([ 0.1,  8. , 21. ])
np.median(y)

####to ignore nan values, then use np.nanpercentile() instead:

y_with_nan = np.insert(y, 2, np.nan)
y_with_nan
np.nanpercentile(y_with_nan, [25, 50, 75])

###NumPy also offers you very similar functionality in quantile() and nanquantile(). 
# If you use them, then you’ll need to provide the quantile values as the numbers between 0 and 1 instead of percentiles:

np.quantile(y, 0.05)

np.quantile(y, 0.95)
np.quantile(y, [0.25, 0.5, 0.75])
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])



z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)

z.quantile(0.95)
z.quantile([0.25, 0.5, 0.75])
z_with_nan.quantile([0.25, 0.5, 0.75])

##################### Ranges

np.ptp(y)

np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)

###calculate the maxima and minima of sequences:
np.amax(y) - np.amin(y)
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
y.max() - y.min()
z.max() - z.min()
z_with_nan.max() - z_with_nan.min()

####The interquartile range is the difference between the first and third quartile. Once  calculate the quartiles, It can take their difference:

quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]
quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

####Summary of Descriptive Statistics
####SciPy and Pandas offer useful routines to quickly get descriptive statistics 
# with a single function or method call. You can use scipy.stats.describe() like this:

result = scipy.stats.describe(y, ddof=1, bias=False)
result

##describe() returns an object that holds the following descriptive statistics:

nobs; minmax; mean; variance; skewness; kurtosis

result.nobs
result.minmax[0]  # Min
result.minmax[1]  # Max
result.mean
result.variance
result.skewness
result.kurtosis

###Pandas has similar, if not better, functionality. Series objects have the method .describe():

result = z.describe()
result

#####It returns a new Series that holds the following:

count; mean; std; min and max; 25%, 50%, and 75%

result['mean']
result['std']
result['min']
result['max']
result['25%']
result['50%']
result['75%']

###Measures of Correlation Between Pairs of Data

x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

#### Covariance
###This is how you can calculate the covariance in pure Python:

n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
           / (n - 1))
cov_xy

###NumPy has the function cov() that returns the covariance matrix:

cov_matrix = np.cov(x_, y_)
cov_matrix


x_.var(ddof=1)
y_.var(ddof=1)


###### The other two elements of the covariance matrix are equal and represent the actual covariance between x and y:

cov_xy = cov_matrix[0, 1]
cov_xy

cov_xy = cov_matrix[1, 0]
cov_xy


####### Pandas Series have the method .cov() that you can use to calculate the covariance:

cov_xy = x__.cov(y__)
cov_xy
cov_xy = y__.cov(x__)
cov_xy


##### Correlation Coefficient
###The value 𝑟 > 0 indicates positive correlation.
###The value 𝑟 < 0 indicates negative correlation.
###The value r = 1 is the maximum possible value of 𝑟. It corresponds to a perfect positive linear relationship between variables.
###The value r = −1 is the minimum possible value of 𝑟. It corresponds to a perfect negative linear relationship between variables.
#####The value r ≈ 0, or when 𝑟 is around zero, means that the correlation between variables is weak

var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

#####scipy.stats has the routine pearsonr() that calculates the correlation coefficient and the 𝑝-value:

r, p = scipy.stats.pearsonr(x_, y_)
r
p

####correlation coefficient matrix:

corr_matrix = np.corrcoef(x_, y_)
corr_matrix

 r = corr_matrix[0, 1]
r
r = corr_matrix[1, 0]
r


##### get the correlation coefficient with scipy.stats.linregress():

scipy.stats.linregress(x_, y_)



####linregress() takes x_ and y_, performs linear regression, and returns the results. 
# slope and intercept define the equation of the regression line, while rvalue is the 
# correlation coefficient. To access particular values from the result of linregress(), 
# including the correlation coefficient, use dot notation:

result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

####Pandas Series have the method .corr() for calculating the correlation coefficient:

r = x__.corr(y__)
r
r = y__.corr(x__)
r

###Working With 2D Data
#### AXES #####
a = np.array([[1, 1, 1],
               [2, 3, 1],
               [4, 9, 2],
               [8, 27, 4],
               [16, 1, 1]])
a

####### Now you have a 2D dataset, which you’ll use in this section. 
# You can apply Python statistics functions and methods to it just as you would to 1D data:

np.mean(a)

a.mean()
np.median(a)
a.var(ddof=1)

#### Let’s see axis=0 in action with np.mean():

np.mean(a, axis=0)
a.mean(axis=0)

###### If you provide axis=1 to mean(), then you’ll get the results for each row:

np.mean(a, axis=1)
a.mean(axis=1)

#################The parameter axis works the same way with other NumPy functions and methods:

np.median(a, axis=0)
np.median(a, axis=1)
a.var(axis=0, ddof=1)
a.var(axis=1, ddof=1)

#####This is very similar when you work with SciPy statistics functions. But remember that in this case, the default value for axis is 0:

scipy.stats.gmean(a)  # Default: axis=0
scipy.stats.gmean(a, axis=0)

##### If you specify axis=1, then you’ll get the calculations across all columns, that is for each row:

scipy.stats.gmean(a, axis=1)

####If you want statistics for the entire dataset, then you have to provide axis=None:

scipy.stats.gmean(a, axis=None)

#####You can get a Python statistics summary with a single function call for 2D data 
# with scipy.stats.describe(). It works similar to 1D arrays, but you have to be careful with the parameter axis:

scipy.stats.describe(a, axis=None, ddof=1, bias=False)
scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0
scipy.stats.describe(a, axis=1, ddof=1, bias=False)

###### You can get a particular value from the summary with dot notation:

result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

#############DataFrames

row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df

df.mean()
df.var()

####### What you get is a new Series that holds the results. In this case, the Series holds the mean 
# and variance for each column. If you want the results for each row, then just specify the parameter axis=1:

df.mean(axis=1)
df.var(axis=1)


###You can isolate each column of a DataFrame like this:

df['A']

##### Now, you have the column 'A' in the form of a Series object and you can apply the appropriate methods:

df['A'].mean()

df['A'].var()

###### Sometimes, you might want to use a DataFrame as a NumPy array and apply some function to it. 
# It’s possible to get all data from a DataFrame with .values or .to_numpy():

df.values

df.to_numpy()

############ Like Series, DataFrame objects have the method .describe() that returns another 
# DataFrame with the statistics summary for all columns:

df.describe()

###########If you want the resulting DataFrame object to contain other percentiles, 
# then you should specify the value of the optional parameter percentiles.

####You can access each item of the summary like this:

df.describe().at['mean', 'A']
df.describe().at['50%', 'B']

####Visualizing Data
####Box plots;Histograms; Pie charts; Bar charts; X-Y plots; Heatmaps

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#######################Box Plots
##############he box plot is an excellent tool to visually represent descriptive statistics of a given dataset. 
# It can show the range, interquartile range, median, mode, outliers, and all quartiles. 
# First, create some data to represent with a box plot:

np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

###### Histograms

hist, bin_edges = np.histogram(x, bins=10)
hist
bin_edges

##### What histogram() calculates, .hist() can show graphically:

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()


############### It’s possible to get the histogram with the cumulative numbers of items if you provide the argument cumulative=True to .hist():

fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

########Pie Charts
x, y, z = 128, 256, 1024

############Now, create a pie chart with .pie():

fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

#######Bar Charts
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

fig, ax = plt.subplots())
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#################X-Y Plots

 x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'


##########linregress returns several values. You’ll need the slope and intercept of the regression line, 
# as well as the correlation coefficient r. Then you can apply .plot() to get the x-y plot:
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

#############Heatmaps

matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()


###########You can obtain the heatmap for the correlation coefficient matrix following the same logic:

matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()