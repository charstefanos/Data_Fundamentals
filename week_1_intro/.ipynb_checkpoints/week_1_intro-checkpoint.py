---
jupyter:
  celltoolbar: Create Assignment
  jupytext_format_version: '1.0'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.6.3
---

# Lab 1: unassessed
## Introduction to Juypter, Python and Numpy
**Marked out of 75**


This lab is intended to introduce you to the basic use of the Juypter+Python+Numpy environment, and show how the autograding in exercises works.

When you complete this exercise, you will submit it on Moodle. You will get a mark, but this **will not count towards your final grade**. 

It is strongly recommended that you complete this exercise fully. This will take around two hours if you already know some NumPy.

# # Outline

* Using Jupyter
* A quick NumPy tutorial
* NumPy questions

# 1: Jupyter
If you have not used Jupyter before [jump to the Jupyter Quickstart](guides/JupyterGuide.ipynb) before continuing. If you have done Level 1 Computing Science, you do not need to do this.

# 2: YACRS
DF(H) uses a class response system to administer quizzes during lectures. These will count for 5% of your final grade. Question responses will be entered on your phone during lectures. To set this up:

* **On your phone, not this computer** go to `learn.gla.ac.uk/yacrs`
* Log in using your GUID (MyCampus login), not your DCS login.
* **Bookmark this page so you can return quickly in lectures.**
* Join `Session 4`; this will be the session for all of DF(H)
* Answer the question about your Python experience.                                             




# # 3: Autograder tests

Lab exercises will (mainly) be autograded via automatic tests.


The following parts have some questions to answer, and some tests (which you cannot alter) which will be run against the code you have written. If the tests pass, you will see how many marks you got with a green tick. If they do not pass, you will see a red cross. Remember, this exercise doesn't count for anything, but do try to complete the exercises.

```{python}
## %
# Make sure you run this cell!

from utils.tick import reset_marks, summarise_marks, marks
from utils.checkarr import array_hash, check_hash
import numpy as np  # NumPy
from utils.matrices import print_matrix, show_boxed_tensor_latex

# Set up Matplotlib
import matplotlib as mpl   
import matplotlib.pyplot as plt
# %matplotlib inline

import utils.image_audio as ia

reset_marks()
print("Everything imported OK")

```

Here's a free 4 marks:

```{python nbgrader={'grade': True, 'grade_id': 'cell-b12d0a22c191c138', 'locked': True, 'points': 4, 'schema_version': 1, 'solution': False}}
with marks(4):
    print("Hello world")
```

And here's what happens when you have an error. Try setting `a` to 1, and making sure you can get this to pass. 

```{python nbgrader={'grade': False, 'grade_id': 'cell-3040ccf7669e107d', 'locked': False, 'schema_version': 1, 'solution': True}}
a = 2
### BEGIN SOLUTION
a = 1
### END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-221df43d91c7b63b', 'locked': True, 'points': 4, 'schema_version': 1, 'solution': False}}
with marks(4):
    assert(a==1)
```

----------------

----------------


# 4. Introduction to NumPy

We will be using [numpy](numpy.org) as the basis for our numerical operations. This provides a datatype called `ndarray`, that can be used to store and manipulate arrays of numbers.


## NumPy worked example
If you have not used NumPy before or if you are rusty [then work through the example before starting](guides/numpy_example.ipynb)

---


# # References and cheat sheets

If you are stuck, the following resources are very helpful:

### Cheatsheets
* [NumPy cheatsheet](https://github.com/juliangaal/python-cheat-sheet/blob/master/NumPy/NumPy.md)
* [Python for Data Science cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf)
* [Another NumPy Cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)

### API reference and user guide
* [NumPy API reference](https://docs.scipy.org/doc/numpy-1.13.0/reference/)
* [NumPy user guide](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.html)

### Tutorials
You don't need to read these, but if you get stuck or want to go further in depth, the following online resources are helpful for getting familiar with scientific processing in Python:
* JR Johansson's scientific python notes: [JRJohansson](https://github.com/jrjohansson/scientific-python-lectures)
* Scientific packages are well covered in this article: [SciPy lectures](https://scipy-lectures.github.io/)
* A useful [Stanford course](http://web.stanford.edu/~arbenson/cme193.html) on scientific python

---





## Purpose of this lab
This lab should help you:
* create simple arrays
* index and slice arrays
* stack arrays together
* compute simple statistics of arrays
* understand array arithmetic and broadcasting rules
* apply conditions to arrays and use Boolean indexing


-----
# NumPy

The foundation package for numerical operations is **NumPy** which provides an array type and accelerated operations on it. 

A very important part of using numerical libraries like numpy is **vectorising operations**: avoiding explicit loops over values in the arrays and instead using library functions to do manipulations. It is *massively* faster to have numpy add to arrays together than to iterate over the elements adding them together in Python.

```{python}
a = np.zeros((5,5))
b = np.ones((5,5))
y = np.zeros((5,5))

# YES: do things like this
# a and b are NumPy arrays
x = a + b

# NO: don't do this
# this is very inefficient
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        y[i,j] = a[i,j] + b[i,j]
```


## No for loops (unless specified)
In this lab **do not** use explicit loops, like `while` or `for`, unless the question explicitly asks you to. In future labs you may need to use the occasional `for` loop, but try and avoid them where possible.


NumPy provides the **np.array** class which is a n-dimensional array of numbers **of the same type**. Arrays can be created in several ways: from a Python list (and nD arrays from nested lists), as a "blank" matrix of zeros or ones or random data, by copying existing arrays, loaded from disk or from certain special functions.

**You can always make a copy of an array using np.array() on an existing array** (e.g. `x = np.array(y)` makes a new **copy** of y). `np.array()` will also convert any iterable object (lists, tuples) into an array if it can. Note that a few operations will *change arrays in place*, and most will *return new copies*.

```{python}
x = np.array([1.0,2.0,3.0,4.0]) # create an array from a list
print(x) # print the array

# print the matrix, but nicer (note: the first string is a LaTeX expression)
print_matrix("x", x) 

print(x.dtype)  # datatype
print(x.shape)  # shape of array
```

# 1: Create some arrays
Create the following arrays, with the following specifications:
(**don't** use `np.array` to do this). If you don't know how to do this,
look at the worked example, or look at the cheatsheet or API reference.

Use `np.zeros`, `np.ones`, `np.full`, `np.arange` and `np.random.normal` to solve these questions.

* `x`: a 8 x 8 matrix of all ones
* `y`: a 2 element vector, with all elements equal to np.pi
* `z`: a 1 x 2 x 5 element array of all ones.
* `q`: an 1D array with 10 elements, from 0-18 (inclusive), stepping by 2. 
* `r`: an 400 element array, random numbers normally distributed with mean 0, std. dev. 1.0 **(don't print this one out)**

Print out your arrays using `print_matrix` to see if they look right.

Check that the tests pass. 

```{python nbgrader={'grade': False, 'grade_id': 'cell-c190eec8e4831a17', 'locked': False, 'schema_version': 1, 'solution': True}}
## BEGIN SOLUTION
x = np.zeros((8,8))
print_matrix("x", x)
y = np.full(2, np.pi )
print_matrix("y", y)
z = np.ones((1,2,5))
print_matrix("z", z)
q = np.arange(0,20,2)
print_matrix("q", q)
r = np.random.normal(0, 1, (400,))
#print_matrix("r", r)
## END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-7a8e23b3cc2d7bea', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(x,((8, 8), 0.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-3eccab087d4c343c', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(y, ((2,),  21.991148575128552)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-c3cd5694245364b2', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(z, ((1, 2, 5), 59.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-f6fd4fc0f0e04fba', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(q,((10,), 701.744562646538)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-8bc8bfb07bb942f3', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(r.shape==(400,)  and np.std(r)>0.75 and np.std(r)<1.5 and np.all(np.diff(r.ravel())!=0.0))
```

# # 2: How to keep snails alive

<img src="imgs/snails.jpg">*[[Image](https://flickr.com/photos/chodhound/6083328289 "Snail") by [ChodHound](https://flickr.com/people/chodhound) license [CC BY-SA](https://creativecommons.org/licenses/by-sa/2.0/)]*

Scientists at the Zoology Department, The University of Adelaide have studied the best conditions to keep snails alive. They have recorded a dataset of observations of snail mortality under controlled conditions. This data set is in the file `data/snails.txt`.

#### An excerpt from the data set description
>Groups of 20 snails were held for periods of 1, 2, 3 or 4 weeks in carefully
controlled conditions of temperature and relative humidity. There were two
species of snail, 0 and 1. At the end of the exposure time the snails
were tested to see if they had survived. 

>The data are unusual in that in most cases fatalities during the experiment
were fairly small. [lucky snails!]

### The task
The data is a 2D array, and has six columns, with these definitions:

     species(binary) exposure(weeks) humidity(%) temperature(deg. C) n_deaths n_snails

Each row represents one set of observations (i.e. one group of snails). You are to compute some basic properties of this data. Use NumPy operations to do the computations.


A. **Loading arrays** 
* Load this data as a NumPy array called `snails`. Note: use NumPy functions to do this! **Do not parse the file yourself** The file is space delimited.
* Print it out. Use this format to print out the results:
    
      print("snails\n", snails)

```{python nbgrader={'grade': False, 'grade_id': 'cell-6d0228523852fbcb', 'locked': False, 'schema_version': 1, 'solution': True}}
### BEGIN SOLUTION
snails = np.loadtxt("data/snails.txt")
print("snails\n", snails)
### END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-d7f4d07872174036', 'locked': True, 'points': 2, 'schema_version': 1, 'solution': False}}
with marks(2):
    assert(check_hash(snails, ((96, 6), 3082003.4024073719)))
```

B. **Basic indexing** 
Compute the following results, storing the results in the variable specified and printing them out. Use the same printing format as A.

* `hum_last` the humidity in the last entry in the table.
* `temp_first` the temperature in the first entry in the table. 
* `weeks` the whole column of "weeks exposure".
* `row_third` the third row of observations. (remember 0 indexing)


```{python nbgrader={'grade': False, 'grade_id': 'cell-17f2099d91b3038f', 'locked': False, 'schema_version': 1, 'solution': True}}
## BEGIN SOLUTION
hum_last = snails[-1, 2]
temp_first = snails[0, 3]
weeks = snails[:,1]
row_third = snails[2,:]
print("hum_last\n", hum_last)
print("temp_first\n", temp_first)
print("weeks\n", weeks)
print("row_third\n", row_third)
## END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-3bd461f1be410f75', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(hum_last, ((), 379.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-590385cd63ae790d', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(temp_first, ((), 50.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-19028943888ae090', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(weeks, ((96,), 13091.118033988751)))

```

```{python nbgrader={'grade': True, 'grade_id': 'cell-57e22f24be8159a7', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(row_third, ((6,), 490.54981015887836)))
```

C. **Aggregate functions** 
Compute the following results, storing the results in the variable specified and printing them out:

* `total_deaths` total number of snails that died
* `total_still_alive` total number of snails that survived the whole study
* `mean_temp` mean temperature in the whole study
* `max_humidity` highest humidity in the study
* `average_death_rate` mean of the ratio of snail deaths to snails in the study
* `snail_weeks` the total amount of snail effort that went into this study (number of snails times number of weeks)

Each computation should be a single line of code

```{python nbgrader={'grade': False, 'grade_id': 'cell-427021397b9a50b2', 'locked': False, 'schema_version': 1, 'solution': True}}
### BEGIN SOLUTION
total_deaths = np.sum(snails[:,4])
total_still_alive = np.sum(snails[:,5]) - total_deaths
mean_temp = np.mean(snails[:,3])
max_humidity = np.max(snails[:,2])
average_death_rate = np.mean(snails[:,4]/ snails[:,5])
snail_weeks = np.sum(snails[:,5]*snails[:,1])

print("total_deaths", total_deaths)
print("total_still_alive", total_still_alive)
print("mean_temp", mean_temp)
print("max_humidity", max_humidity)
print("average_death_rate", average_death_rate)
print("snail_weeks", snail_weeks)
### END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-12aba5ae7c698554', 'locked': True, 'points': 2, 'schema_version': 1, 'solution': False}}
with marks(2):
    assert(check_hash(total_deaths, ((), 1375.0)))

```

```{python nbgrader={'grade': True, 'grade_id': 'cell-0799d88a4e889521', 'locked': True, 'points': 2, 'schema_version': 1, 'solution': False}}
with marks(2):
    assert(check_hash(total_still_alive, ((), 8225.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-e3030fa5597a6a3c', 'locked': True, 'points': 2, 'schema_version': 1, 'solution': False}}
with marks(2):
    assert(check_hash(mean_temp, ((), 75.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-d7b5c4b686b8e185', 'locked': True, 'points': 2, 'schema_version': 1, 'solution': False}}

with marks(2):
    assert(check_hash(max_humidity, ((), 379.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-759195cd947bae30', 'locked': True, 'points': 3, 'schema_version': 1, 'solution': False}}
with marks(3):
    assert(check_hash(average_death_rate, ((), 0.71614583333333326)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-e75b606eb7b69e7a', 'locked': True, 'points': 3, 'schema_version': 1, 'solution': False}}
with marks(3):
    assert(check_hash(snail_weeks, ((), 24000.0)))
```

D. **Boolean indexing**
Compute the following results, storing the results in the variable specified and printing them out:

* `species_0` and `species_1`: split the dataset into two arrays, one with the entries for species 0 and one with the entries for species 1.

* `weakest_snail` the snail species (0 or 1) that had the highest average death rate

```{python nbgrader={'grade': False, 'grade_id': 'cell-2fb2b916ca1d1254', 'locked': False, 'schema_version': 1, 'solution': True}}
### BEGIN SOLUTION
# split into two parts, based on species column
species_0 = snails[snails[:,0]==0]
species_1 = snails[snails[:,0]==1]
print("species0\n",species_0)
print("species1\n",species_1)

death_rate = snails[:,4] / snails[:,5]


# compute mean for each subdivision of the data

print(average_death_rate_0, average_death_rate_1)
# which is bigger?
weakest_snail = 0 if np.mean(death_rate[snails[:,0]==0])>np.mean(death_rate[snails[:,0]==1]) else 1
print("weakest_snail\n",weakest_snail)
### END SOLUTION    
```

```{python}

death_rate = snails[:,4] / snails[:,5]
weakest_snail = 1 if np.mean(death_rate[snails[:,0]==0], axis=0) > np.mean(death_rate[snails[:,0]==1], axis=0) else 0
print(weakest_snail)
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-33631e32555140e0', 'locked': True, 'points': 5, 'schema_version': 1, 'solution': False}}
with marks(5):
    assert(check_hash(species_0, ((48, 6), 762041.58357993606)))
    assert(check_hash(species_1, ((48, 6), 791902.31693958596)))  
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-7405a52cf551031d', 'locked': True, 'points': 3, 'schema_version': 1, 'solution': False}}
with marks(3):
    assert(check_hash(weakest_snail, ((), 5.0)))
```

E. **Arithmetic and ordering**
Compute the following results, storing the results in the variable specified and printing them out:

* `deg_f` each temperature in the study, but in degrees Fahrenheit. Use the knowledge that `0C = 32F, 100C = 212F`
* `mean_cols` the mean of all the columns, as a 1D vector
* `death_rate` the death rates, in sorted order, smallest first
* `exposure_death_order` the exposure durations (in weeks), but in sorted in the order of death rates, smallest death rate first.

* `best_temp`, `best_hum` the best temperature and humidity to keep a snail for four weeks without it dying. *Look only at the four week exposures, ignoring snails kept for less than this time.* 

```{python nbgrader={'grade': False, 'grade_id': 'cell-965b0640474113ac', 'locked': False, 'schema_version': 1, 'solution': True}}
### BEGIN SOLUTION
# simple arithmetic on the temperature column
deg_f = snails[:,3] * (212-32)/100.0 + 32
print(deg_f)

mean_cols = np.mean(snails, axis=0)
print(mean_cols)


# compute death rate, then sort it
deaths = snails[:,4]/ snails[:,5]
death_rate = np.sort(deaths)
print(death_rate)

# compute the *ordering* that would sort the death rates
# and apply it to index the array of exposures
death_rate_order = np.argsort(deaths)
exposure_death_order = snails[:,1][death_rate_order]
print(exposure_death_order) # the leading killer of snails is old age

# select only the snails that were kept for four weeks
# select the death rate for those snails
week_4 = snails[snails[:,1]==4]
week_4_deaths = week_4[:,4]/ week_4[:,5]

best_conditions = np.argmin(week_4_deaths) # smallest death rate *index*

# now index the temp and humidity using that index
# note: this will only work indexing week_1
best_temp, best_hum = week_4[best_conditions,3], week_4[best_conditions,2]
print("best_temp", best_temp, "best_hum", best_hum)
### END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-9aca90f5ed692ce6', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(deg_f, ((96,), 275523.34846922837)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-7d0f86f908ce2a87', 'locked': True, 'points': 1, 'schema_version': 1, 'solution': False}}
with marks(1):
    assert(check_hash(mean_cols, ((6,), 522.92336963727359)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-e8a171753a823a96', 'locked': True, 'points': 2, 'schema_version': 1, 'solution': False}}
with marks(2):
    assert(check_hash(death_rate, ((96,), 1100.9803982902313)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-554ca9fb59641ac9', 'locked': True, 'points': 2, 'schema_version': 1, 'solution': False}}
with marks(2):
    assert(check_hash(exposure_death_order, ((96,), 13983.118033988751)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-22bc20db6ce8b95e', 'locked': True, 'points': 3, 'schema_version': 1, 'solution': False}}
with marks(3):
    assert(check_hash(best_temp, ((), 50.0)))
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-36ed1c66c64c5946', 'locked': True, 'points': 3, 'schema_version': 1, 'solution': False}}
with marks(3):
    assert(check_hash(best_hum, ((), 379.0)))      
```

## 3: Image operations
Images can be represented as numerical arrays. We will use images as an example to explore NumPy functionality.

* `img = ia.load_image_colour('filename.png')` will load an image as an array.
* `ia.show_image(img)` will show it in the notebook.



A)

We will:
* Load `data/parrots.png` as `img_array` 
* Print out its shape and dtype
* Show the image.

```{python}
img_array = ia.load_image_colour("data/parrots.png")
print(img_array.shape, img_array.dtype)
ia.show_image(img_array)
```

B) **Slicing arrays**
* Create an array `cropped` which has the pixels from [150,100] to [350,300]. Note that these positions are in `[row, col]` format, not `[x,y]`.
* Display the cropped array using `show_image()`. 
* Remember: the image is `WxHx3`. Think about how to slice the last dimension.
* Show the cropped image so you can see it.

```{python nbgrader={'grade': False, 'grade_id': 'cell-e55532295e444bd0', 'locked': False, 'schema_version': 1, 'solution': True}}

## BEGIN SOLUTION
cropped = img_array[150:350, 100:300, :] # note: the last colon can actually be omitted
## END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-1065728e6cf70cf4', 'locked': True, 'points': 4, 'schema_version': 1, 'solution': False}}
ia.show_image(cropped)
with marks(4):
    assert(check_hash(cropped, ((200, 200, 3), 3409234926.1084023)))
```

C)  **Modifying arrays**

Create an array "censored" which is the same as `img_array`, but has a black bar across the following regions to protect the parrot's privacy:
    * [200,100] -> [260, 310]
    * [140, 400]-> [200, 650]

Setting array elements to zero will make them black.

**Do not modify the original `img_array`**

```{python nbgrader={'grade': False, 'grade_id': 'cell-6615e422ecee6b7e', 'locked': False, 'schema_version': 1, 'solution': True}}
## BEGIN SOLUTION
censored = np.array(img_array)
# just use slicing to set all of the values to zero
censored[200:260,100:310, :] = 0
censored[140:200,400:650, :] = 0
## END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-fc66e323769d8a84', 'locked': True, 'points': 4, 'schema_version': 1, 'solution': False}}
ia.show_image(censored)
with marks(4):
    assert(check_hash(censored, ((512, 768, 3), 250654064351.74332)))
    assert(check_hash(img_array, ((512, 768, 3), 269458072078.68924)))
```

D) **Elementwise arithmetic** The image is stored as three colour planes, R,G,B. This is the `3` in the last position of the shape of the image (check the `shape` of the array where it is loaded in). The planes are often referred to as "channels". 

* Create an array `channel_diff`, which will be the **absolute** difference of the red channel of `img_array` (channel 0) from the green channel (channel 1), all scaled by a factor of **eight**. Hint: you need to use slicing here.


```{python nbgrader={'grade': False, 'grade_id': 'cell-41fe7b5935b9cdbb', 'locked': False, 'schema_version': 1, 'solution': True}}
## BEGIN SOLUTION
channel_diff = 8*np.abs(img_array[:,:,1] - img_array[:,:,0])
## END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-759063de0cbaee7a', 'locked': True, 'points': 4, 'schema_version': 1, 'solution': False}}
ia.show_image(channel_diff)
with marks(4):
    assert(check_hash(channel_diff, ((512, 768), 95264282406.86458)))
```

E) **Stacking arrays**

Using *one* `for` loop, create a *list* of arrays which represent a panning animation which is a sequence of cropped versions of `img_array`. 

* The crops should start at [190,30] and be 200 pixels wide and 100 pixels high, and move 12 pixels right and 2 pixels up in each frame (that means the column index increases by 12, and the row index decreases by 2). 
* Create 30 frames. All of them will be the same size.
* Stack the array into a single array called `panning_array`.

Your code should be less than 15 lines (excluding any comments)!

```{python nbgrader={'grade': False, 'grade_id': 'cell-3d8d7655e04e7660', 'locked': False, 'schema_version': 1, 'solution': True}}
## BEGIN SOLUTION
# starting position
row, col = 190, 30
panning = []

# loop over frames
for i in range(30):
    # append the cropped array
    panning.append(img_array[row:row+100, col:col+200, :])
    # pan
    col += 12
    row -= 2

# concatenate the array
panning_array = np.stack(panning)
# will be [30,200,200,3]

## END SOLUTION
```

```{python nbgrader={'grade': True, 'grade_id': 'cell-b0bf82f051dbe3ef', 'locked': True, 'points': 10, 'schema_version': 1, 'solution': False}}
ia.show_gif(panning_array, width="80%")
with marks(10):
    assert(check_hash(panning_array,((30, 100, 200, 3), 834973352307.7242)))
```




# SUBMIT THIS FILE **ONLY** (week_1_intro.ipynb)  ON MOODLE
