# ---
# jupyter:
#   celltoolbar: Create Assignment
#   jupytext_format_version: '1.1'
#   jupytext_formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.6
# ---

# %% [markdown]
# ## Lab 2: **Extended material**
#
# # Numerical arrays and vectorized computation
# <font color="red">
#     
# ## IMPORTANT
# This material is optional. You do not have to attempt it or look at it. There are no marks. Do not submit this file.
# </font>

# %% {"nbgrader": {"grade": false, "grade_id": "cell-7d8b7c3700ac1bc7", "locked": true, "schema_version": 1, "solution": false}}
# Standard imports
# Make sure you run this cell!
# NumPy
import numpy as np  
np.set_printoptions(suppress=True)

# Set up Matplotlib
import matplotlib as mpl   
import matplotlib.pyplot as plt
%matplotlib inline

# custom utils
import utils.float_inspector
import importlib; importlib.reload(utils.float_inspector)
from utils.checkarr import array_hash, check_hash
import utils.image_audio as ia
import utils.tick as tick

print("Everything imported OK")

# %% [markdown]
# # 3. Working with tensors [extended version]
# The file `data/font_sheet.png` contains a number of characters in different fonts. It is an image which consists of the images of each *printable* ASCII character, (characters 32-128) arranged left to right. Each character image is precisely square. 
#
# These are the characters present, in order:

# %%
chars = "".join([chr(i) for i in range(32,128)])
print(chars)

# %% [markdown]
#   
# Each font is also stacked left to right, so the image is one *very* long strip of characters. The image is grayscale.

# %%
all_fonts = ia.load_image_gray("data/font_sheet.png")
print(all_fonts.shape)

# %%
# show a portion of the image
ia.show_image(all_fonts[:,15000:18000])

# %% [markdown]
# # Tasks
# A. Rearrange the image into a tensor called `font_sheet` that is ordered like this:
#
#         (font, character, rows, cols)
#         
# * Showing the image `font_sheet[16,33,:,:]` should show the "A" character of the 17th font.
# * Showing the image `font_sheet[10, 1, :, :]` should be the "!" character of the 11th font.

# %% {"nbgrader": {"grade": false, "grade_id": "cell-66e3a2fcbe9e7e36", "locked": false, "schema_version": 1, "solution": true}}
n_chars = len(chars)
# we know characters are square, so must be 64 chars wide
# note that we *have* to transpose the sheet first before reshaping
# this is because of the "pouring rule" for reshaping
font_sheet = all_fonts.T.reshape(-1, n_chars,  64, 64) 
font_sheet = np.einsum("ijkl->ijlk", font_sheet)
# alternative: font_sheet = font_sheet.swapaxes(2,3)

# %%
# if your code worked, you should see an A below
ia.show_image_mpl(font_sheet[16,33,:,:])

# %%
# if your code worked, you should see an ! below
ia.show_image_mpl(font_sheet[10,1,:,:])

# %% {"nbgrader": {"grade": true, "grade_id": "cell-e21fcd41d335d470", "locked": true, "points": 3, "schema_version": 1, "solution": false}}
# test shape is correct
with tick.marks(3):    
    assert(check_hash(font_sheet.shape, ((4,), 944.94994725732522)))

# %% {"nbgrader": {"grade": true, "grade_id": "cell-b26945a0ff9f8ded", "locked": true, "points": 8, "schema_version": 1, "solution": false}}
# test content is ok

with tick.marks(8):
    assert(check_hash(font_sheet, ((29, 96, 64, 64), 55401395660501.18)))

# %% [markdown]
# B. Create an array `mean_letters`, which will be a 2D image containing the mean image of each character across all fonts for all of the letters in *the lowercase alphabet*.  The letters should be arranged horizontally in a strip:
#
#        abcdefghijklmnopqrstuvwxyz
#
# You should be able to *very vaguely* make out the shape of the letters. Hint: if you have trouble debugging this code, try taking the mean over just one or two fonts, to see if the letter shapes look right, before expanding to cover all fonts.
#
# Hint:
# * you will have to partially *undo* some of the swapping/reshaping you did earlier to get the data in the right format

# %% {"nbgrader": {"grade": false, "grade_id": "cell-885aa32f73499743", "locked": false, "schema_version": 1, "solution": true}}
mean_letters = np.mean(font_sheet[:,:,:,:], axis=0)
lowercase_letters = chars.find('a'), chars.find('z')+1
print(lowercase_letters)
mean_letters = mean_letters[lowercase_letters[0]:lowercase_letters[1], :, :]

# have to swap the middle axes then transpose to get right shape
mean_letters = mean_letters.swapaxes(1,2).reshape(-1, 64).T

# %%
# the results will be blurry, but you should be able to make out the letters (just)
ia.show_image(mean_letters)

# %% {"nbgrader": {"grade": true, "grade_id": "cell-70f65b6bc9676a93", "locked": true, "points": 8, "schema_version": 1, "solution": false}}

with tick.marks(8):
    assert(check_hash(mean_letters,((64, 1664), 4861094994.1019411)))

# %% [markdown]
# C. Increase the contrast of the mean letters by applying this formula:
#
#         x_contrast = ((x - 0.5) * contrast_factor) + 0.5
#
#  Store the result in `mean_letters_contrast`. Use a contrast factor of 3.0. Show the result.
#  Hint: this is easy.

# %% {"nbgrader": {"grade": false, "grade_id": "cell-e9b55239e54bd3d9", "locked": false, "schema_version": 1, "solution": true}}
mean_letters_contrast = ((mean_letters-0.5) * 3) + 0.5
ia.show_image(mean_letters_contrast)

# %% {"nbgrader": {"grade": true, "grade_id": "cell-23945e7b1798eac2", "locked": true, "points": 3, "schema_version": 1, "solution": false}}
with tick.marks(3):
    assert(check_hash(mean_letters_contrast, ((64, 1664), 8912532722.3058205)))

# %% [markdown]
# D. Create a **binarized** version of the image, which is 0 if the contrast-adjusted mean is <0.75 and 1 otherwise. Store this in `binary_mean_letters_contrast`.

# %% {"nbgrader": {"grade": false, "grade_id": "cell-c46348c52d6b66c8", "locked": false, "schema_version": 1, "solution": true}}
binary_mean_letters_contrast = np.where(mean_letters_contrast<0.75, 0, 1)
ia.show_image(binary_mean_letters_contrast)
print(np.sum(binary_mean_letters_contrast))

# %% {"nbgrader": {"grade": true, "grade_id": "cell-6d4e26869e3c724b", "locked": true, "points": 3, "schema_version": 1, "solution": false}}
with tick.marks(3):
    assert(check_hash(binary_mean_letters_contrast, ((64, 1664), 704604624.20427608)))

# %% [markdown]
# E. Complete the function below. It should render text using the provided font index, and *return* a single array with the text rendered in a horizontal strip. It should use `font_sheet` that you defined earlier. You can assume equal spacing of letters. 
#
# * You can compute the index of the character in the same units as the font sheet using the formula:
#
#       ix = ord(char) - 32
#     
# Every ASCII character (32-127) should be rendered. Any character that could not be rendered should be rendered as a **blank white** square.    
#
# * It is fine to use a `for` loop to solve this problem

# %% {"nbgrader": {"grade": false, "grade_id": "cell-ca02abfc30457680", "locked": false, "schema_version": 1, "solution": true}}
def render_text(string, font_index):
    """Returns an image with the given string rendered, using the font_index selected.
    Reads characters from font_sheet.
    string: String to be rendered.
    font_index: index of the font to use"""
    pass # you can delete this line
    ### BEGIN SOLUTION
    glyphs = []
    for char in string:
        index = ord(char) - 32
        if index>=0 and index<96:
            glyph = font_sheet[font_index, index, :, :]
        else:
            glyph = np.ones((64,64)) # blank character
        glyphs.append(glyph)
    return np.concatenate(glyphs, axis=1)
    ### END SOLUTION
    

# %%
# you should be able to read this
ia.show_image(render_text("Can you see this clearly?", 23))

# %%
# this should look the same
ia.show_image(render_text("Can\tyou\nsee\xf5this\x00clearly?", 23))

# %%
ia.show_image(render_text("Data Fundamentals (H)", 1))

# %% {"nbgrader": {"grade": true, "grade_id": "cell-dd638baea99d897d", "locked": true, "points": 2, "schema_version": 1, "solution": false}}
with tick.marks(2):
    assert(check_hash(render_text("Test 1", 1), ((64, 384), 269160963.20571893)))

# %% {"nbgrader": {"grade": true, "grade_id": "cell-0af36cc95e9bf868", "locked": true, "points": 2, "schema_version": 1, "solution": false}}
with tick.marks(2):
    assert(check_hash(render_text("Test 2", 2),((64, 384), 282670129.18082076)))

# %% {"nbgrader": {"grade": true, "grade_id": "cell-ba4876872eacdce0", "locked": true, "points": 2, "schema_version": 1, "solution": false}}
with tick.marks(2):
    assert(check_hash(render_text("Test\n3", 3), ((64, 384), 283057779.18977338)))

# %% {"nbgrader": {"grade": true, "grade_id": "cell-62a4372779989f3c", "locked": true, "points": 2, "schema_version": 1, "solution": false}}
with tick.marks(2):
    assert(check_hash(render_text("\n\tTest\x00\xff4", 4), ((64, 576), 657469474.43368447)))
