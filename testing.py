import pandas as pd
import numpy as np

d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)

print(df)

print(df['two']['b'])


#BGR
def color_func(x):
    ind = np.argmax(x, axis=0)
    if ind == 0:
        return [25, 0, 0]
    # elif ind == 1:
    #     #red
    #     return [0, 0, 255]
    # elif ind == 2:
    #     #yellow
    #     return [0, 255, 255]
    # elif ind == 3:
    #     #green
    #     return [0, 255, 0]
    else:
        return [255, 0, 0]

a = np.array([[1, 0], [0, 1]])
b = np.apply_along_axis(color_func, axis=1, arr=a)
print(b)



