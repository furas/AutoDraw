# AutoDraw

It is code from 

> [How I Used Machine Learning to Automatically Hand-Draw Any Picture](https://towardsdatascience.com/how-i-used-machine-learning-to-automatically-hand-draw-any-picture-7d024d0de997)  
> Supervised and unsupervised learning made easy!   
> Austin Nguyen (Jun 1, 2020)  


Because code was uncomplete and splited in few parts so I put all code in one file (`main-original.py`), 
and add needed imports and missing functions (to load and display image) and it seems it works.

Next I created version `main.py` with some modifications.

I added two modifications:

- when `PyAutoGUI` controls mouse to draw image then it is almost impossible to stop script - so I add `pynput.keyboard.Listener` to listen keyboard and stop code on `ESC`
- because I have two monitors so it was drawing in wrong place - between monitors - so I added option to set screen size manually `AutoDraw(...., screen_size=(width, height))`

See: [CHANGELOG](CHANGELOG.md)

---

I tested it with `GIMP` on Linux. It can work with any program because it only control mouse to press/release mouse button and move mouse in new place.  
You have to manually open painting program, select colors (when script ask for it and shows color values).  
This color is in `BGR` (Blue, Green, Red), not `RGB` (Red, Green, Blue), (because `cv2` uses `BRG`) so you have to remember to copy values in correct order.

---

Docstring from scripts:

```
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
```
---

Modules 

```
pip install opencv-python
pip install numpy
pip install PyAutoGUI
pip install sklearn
pip install kdtree
```

---

Original Image


![](https://github.com/furas/AutoDraw/raw/main/original-image-1a.png)

AutoDraw + GIMP using small brush (3px) (Linux Mint, original size 1920x1200 - just click image)

![](https://github.com/furas/AutoDraw/raw/main/autodraw-image-1a.png)

---

See: [CHANGELOG](CHANGELOG.md)
