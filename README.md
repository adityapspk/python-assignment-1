1. What makes NumPy.shape() different from NumPy.size()?
Python NumPy shape()
1.NumPy arrays have a function called shape that always returns a tuple with each index having the number of adjacent elements.
2.The Numpy array shape property is to find the shape of an array.
3.In this method we can easily use the numpy.shape() function
Syntex:

numpy.shape
           (
            arr
           )

It consists of few parameters:
--arr: input array
--Returns: The values of the shape function always give the length of the adjacent np. array.
Example:
Let’s take an example to check how to implement Python NumPy shape.
import numpy as np

arr2= np.array([[4, 2, 3, 2, 1, 8],
                [5, 4,6,7,8,9]])
res = np.shape(arr2)
print(res)

numpy.size() 
count the number of elements along a given axis.
Syntex:
 numpy.size(arr, axis=None)
Parameters: 
arr: [array_like] Input data. 
axis:
 [int, optional] Axis(x,y,z) along which the elements(rows or columns) are counted. By default, give the total number of elements in a array
Returns:
 [int] Return the number of elements along a given axis. 

2. In NumPy, describe the idea of broadcasting?

 The term broadcasting refers to the ability of NumPy to treat arrays of different shapes during arithmetic operations. Arithmetic operations on arrays are usually done on corresponding elements. If two arrays are of exactly the same shape, then these operations are smoothly performed.
Example 1
import numpy as np 
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print c
Its output is as follows −
[10   40   90   160]  

If the dimensions of two arrays are dissimilar, element-to-element operations are not possible. However, operations on arrays of non-similar shapes is still possible in NumPy, because of the broadcasting capability. The smaller array is broadcast to the size of the larger array so that they have compatible shapes.
Broadcasting is possible if the following rules are satisfied −

Array with smaller ndim than the other is prepended with '1' in its shape.
Size in each dimension of the output shape is maximum of the input sizes in that dimension.
An input can be used in calculation, if its size in a particular dimension matches the output size or its value is exactly 1.
If an input has a dimension size of 1, the first data entry in that dimension is used for all calculations along that dimension.
A set of arrays is said to be broadcastable if the above rules produce a valid result and one of the following is true −

Arrays have exactly the same shape.
Arrays have the same number of dimensions and the length of each dimension is either a common length or 1.
Array having too few dimensions can have its shape prepended with a dimension of length 1, so that the above stated property is true.

The following program shows an example of broadcasting.
Example 2

import numpy as np 
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])  
print 'First array:' 
print a 
print '\n'  
print 'Second array:' 
print b 
print '\n'  
print 'First Array + Second Array' 
print a + b

The output of this program would be as follows −
First array:
[[ 0. 0. 0.]
 [ 10. 10. 10.]
 [ 20. 20. 20.]
 [ 30. 30. 30.]]

Second array:
[ 1. 2. 3.]

First Array + Second Array
[[ 1. 2. 3.]
 [ 11. 12. 13.]
 [ 21. 22. 23.]
 [ 31. 32. 33.]]

3. What makes Python better than other libraries for numerical computation?
    1| SciPy (Scientific Numeric Library):
Officially released in 2000-01, SciPy is free and open source library used for scientific computing and technical computing. The library consists of modules for optimisation,

image processing, FFT, special functions and signal processing.

The SciPy package includes algorithms and functions which are the crux of Python scientific computing capabilities. The sub-package includes: 

-io: used for the standard input and output
-lib: this function is used to wrap python external libraries
-signal: used for processing signal tools
-sparse: used for algorithms related to sparse matrix
-spatial: widely used to determine paths in KD-trees, nearest neighbor and distance functions.
-optimise: used to optimise algorithms which include linear programming.
-linals: used for the regular linear algebra applications.
-interpolate: used for the integration of tools
-intergate: applied for integration of numerical tools
-fftpack: this subpackage helps for the discretion Fourier to transform algorithms
-cluster: the package consists of hierarchical clustering, vector quantisation, and K-means.
-misc: used for the miscellaneous utility applications.
-special: used to switch in special functions.
weave: a tool to convert C/C++ codes to python programming.
-ndimage: used for wide range of functions in multi-dimensional image processing.
-stats: used for better understanding and analysing of statistical functions.
-constants: this algorithm includes physical specification and conversion components.

     2| Pandas (Data Analytics Library):
Pandas is the most important data analysis library of Python. Being open source, it is used for analysing data with Python. It can take data formats of CSV or TSV files, or a SQL database and convert it into Python data frames with rows and columns which is similar to tables in statistical formats. The package makes comparisons with dictionaries with the aid of ‘for’ loops which are very easy to understand and operate.

Python 2.7 and above versions are required to install Pandas package. We need to import the Panda’s library into the memory to work with it. The following codes can be run to implement different operations on pandas.

 - Import pandas as pd (importing pandas library to memory), it is highly suggested to import the library as pd because next time when we want to use the application we need not mention the package full name instead we can name as pd, this avoids confusion.
-pd.read_filetype() (to open the desired file)
-pd.DataFrame() (to convert a specified python object)
-df.to_filetype (filename) (to save a data frame you are currently working with)
The advantage of using Pandas is that it can perform a bunch of functions on the tables we have created. The following are some functions that can be performed on selected data frames.

 -df.median()-to get the median of each column
-df.mean()-to get the mean of all columns
-df.max()-to get the highest value of a column
-df.min()-to get the minimum value of a column
-df.std()-to get the standard deviation of each column.
-df.corr()-to specify the relationship between columns of a data frame.
-df.count()-to get the number of non-null values in each column of the data frame.

      3| IPython (Command Shell)
Developed by Fernando Perez in the year 2001, IPython is a command shell which is designed for interactive calculation in various programming languages. It offers self-examination, rich media, shell syntax, tab completion, and history.

IPython is a browser-based notebook interface which supports code, text, mathematical expressions, inline plots and various media for interactive data visualisation with the use of GUI (Graphic User Interface) toolkits for flexible and rectifiable interpreters to load into one’s own projects.

IPython architecture contributes to parallel and distributed computing. It facilitates for the enhanced parallel applications of various styles of parallelism such as:

-Customer user defined prototypes
-Task Parallelism
-Data Parallelism
-Message cursory using M.P.I (Message Passing Interface)
-Multiple programs, multiple data (MIMD) parallelism
-A single program, multiple data (SPMD) parallelism

      4| Numeric Python (Fundamental Numeric Package):
Better known as Numpy, numeric Python has developed a module for Python, mostly written in C.  Numpy guarantees swift execution as it is accumulated with mathematical and numerical functions.

Robust Python with its dynamic data structures, efficient implementation of multi-dimensional arrays and matrices, Numpy assures accurate calculations with matrices and arrays.

We need to import Numpy into memory to perform numerical operations.

-Import numpy as np (to import Numpy into memory)
-A_values=[20,30,40,50] (defining a list)
-A=np.array(A_values) (to convert list into one dimensional numpy array)
-print(A) (to get one dimensional array displayed)
-print(A*9/5 +32) (to turn values in the list into degrees fahrenheit)

      5| Natural Language Toolkit (Library For Mathematical And Text Analysis):
Simply known as NLP, Natural Language Processing library is used to build applications and services that can understand and analyse human languages and data. One of the sub-libraries which are widely used in NLP is NLTK (Natural Language Toolkit). It has an active discussion forum through which they give hands-on guidance on programming basic topics such as computational linguistics, comprehensive API documentation, linguistics to engineers, students, industries and researchers. NLTK is an open source free community-driven project which is accessible for operating systems such as Windows, MAC OS X, and Linux. The implementations of NLP are:

 -Search engines (eg: Yahoo, Google, firefox etc) they use NLP to optimise the search results for users.
-Social websites like Facebook, Twitter use NLP for the news feed. The NLP algorithms understand the interests of the users and show related posts.
-Spam filters: unlike the traditional spam filters, the NLP has driven spam filters to understand what the mail is about and decides whether it is a spam or not.
NLP includes well known and advanced sub-libraries which are very effective in mathematical calculations.

  -NLTK, which handles text analysis and related problems. Having over 50 corpora and lexicons, 9 stemmers and handful of algorithms NLTK is very popular for education and research. The    application involves a deep learning and analysing process which makes it one of the tough libraries in NLP
-TextBlob, which is a simple library for text analysis
-Stanford core NLP, a library that includes entity recognition, pattern understanding, parsing, tagging etc.
-SpaCy, which presents the best algorithm for the purpose
-Gensim, which is used for topic prototypes and document similarity analysis

4. How does NumPy deal with files?
  NumPy introduces a simple file format for ndarray objects. This .npy file stores data, shape, dtype and other information required to reconstruct the ndarray in a disk file such that the array is correctly retrieved even if the file is on another machine with different architecture.
   -- load() and save() functions handle /numPy     binary files (with npy extension).
   -- loadtxt() and savetxt() functions handle normal          text files.
numpy.save()
The numpy.save() file stores the input array in a disk file with npy extension.

import numpy as np 
a = np.array([1,2,3,4,5]) 
np.save('outfile',a)
To reconstruct array from outfile.npy, use load()
function.
import numpy as np 
b = np.load('outfile.npy') 
print b

It will produce the following output −
array([1, 2, 3, 4, 5])

The save() and load() functions accept an additional Boolean parameter allow_pickles. A pickle in Python is used to serialize and de-serialize objects before saving to or reading from a disk file.

savetxt()
The storage and retrieval of array data in simple text file format is done with savetxt() and loadtxt() functions.

Example
import numpy as np 

a = np.array([1,2,3,4,5]) 
np.savetxt('out.txt',a) 
b = np.loadtxt('out.txt') 
print b 

It will produce the following output −
[ 1.  2.  3.  4.  5.] 

The savetxt() and loadtxt() functions accept additional optional parameters such as header, footer, and delimiter.

5. Mention the importance of NumPy.empty() ?
numpy.empty()
   The numpy module of Python provides a function called numpy.empty(). 
This function is used to create an array without initializing the entries of given shape and type.
Just like numpy.zeros(), the numpy.empty() function doesn't set the array values to zero, and it is quite faster than the numpy.zeros(). This function requires the user to set all the values in the array manually and should be used with caution.
Syntax:
numpy.empty(shape, dtype=float, order='C')  
Parameters:
shape: int or tuple of ints
This parameter defines the shape of the empty array, such as (3, 2) or (3, 3).
dtype: data-type(optional)
This parameter defines the data type, which is desired for the output array.
order: {'C', 'F'}(optional)
This parameter defines the order in which the multi-dimensional array is going to be stored either in row-major or column-major. By default, the order parameter is set to 'C'.
Returns:
This function returns the array of uninitialized data that have the shape, dtype, and order defined in the function.
Example 1:
import numpy as np  
x = np.empty([3, 2])  
x  
Output:

array([[7.56544226e-316, 2.07617768e-316],
           [2.02322570e-316, 1.93432036e-316],
           [1.93431918e-316, 1.93431799e-316]])

In the above code:
 --We have imported numpy with alias name np.
 --We have declared the variable 'x' and assigned the returned value of the np.empty() function.
 --We have passed the shape in the function.
 --Lastly, we tried to print the value of 'x' and the difference between elements.

Example 2:
import numpy as np  
x = np.empty([3, 3], dtype=float, order='C')  
x  
Output:

array([[ 2.94197848e+120, -2.70534020e+252, -4.25371363e+003],
           [ 1.44429964e-088,  3.12897830e-053,  1.11313317e+253],
           [-2.28920735e+294, -5.11507284e+039,  0.00000000e+000]]) 

In the above code:
 --We have imported numpy with alias name np.
 --We have declared the variable 'x' and assigned the returned value of the np.empty() function.
 --We have passed the shape, data-type, and order in the function.
 --Lastly, we tried to print the value of 'x' and the difference between elements.
In the output, it shows an array of uninitialized values of defined shape, data type, and order.
