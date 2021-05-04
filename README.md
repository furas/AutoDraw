# AutoDraw

It is code from [How I Used Machine Learning to Automatically Hand-Draw Any Picture](https://towardsdatascience.com/how-i-used-machine-learning-to-automatically-hand-draw-any-picture-7d024d0de997)

Because code was uncomplete and splited in few parts so I put all in one file, found needed modules, add some debug messages and it seems it works.

I added two modifications:

- when `PyAutoGUI` controls mouse to draw image then it is almost impossible to stop script - so I add `pynput.keyboard.Listener` to listen keyboard and stop code on `ESC`
- because I have two monitors so it was drawing in wrong place - between monitors - so I added option to set scree size manually `AutoDraw(...., screen_size=(width, height))`

I tested it with `GIMP` on Linux. It works with any program because it only control mouse to press/release button and move mouse in new place.
You have to manually open painting program. From time to time it asks to change color in painting program because it doesn't know how to do this. 
It shows what color to use. 
This color is in `BGR` (Blue, Green, Red), not `RGB` (Red, Green, Blue), (because `cv2` uses `BRG`) so you have to reember to put values in painting program in different order.


# https://python-kdtree.readthedocs.io/en/latest/
# https://github.com/stefankoegl/kdtree
# https://pypi.org/project/kdtree/
# pip install kdtree


"""
AutoDraw - inital version 

original (incomplete) code: 

    Austin Nguyen, Jun 1, 2020

    How I Used Machine Learning to Automatically Hand-Draw Any Picture
    Supervised and unsupervised learning made easy!

    https://towardsdatascience.com/how-i-used-machine-learning-to-automatically-hand-draw-any-picture-7d024d0de997

code completion:

    Bartlomiej "furas" Burek (https://blog.furas.pl)

    date: 2021.05.04
"""


