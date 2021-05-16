### date: 2021.05.16
    
- add other options in __init__ (with default values): 
    start_x=None, start_y=None, detail=1, scale=7/12, 
    sketch_before=False, with_color=True, num_colors=10, outline_again
- use `colorama` for colors (to make sure it will work on all systems)


### 2021.05.05

- add `\r` in when display messages with number of paths - now displays it in one line like in progress bars.
- add colors in messages (It uses `bash` codes so may work only on Linux). 
     Change `PRINT_COLORS = False` at the beginning of code to turn off colors.
- display color values in RGB instead of BGR - and also as hex code (ie. `#3F08C2`) - to simpler copy it to painting program. It uses also colors for display it.
- display at start number of colors (it shows how many times we will have to change colors in painting program)
- display number of points in path

### 2021.05.04

- add debug messages
- add option to set screen size manually
- set screen size (1920,1200) because I have two monitors and it was trying to draw in wrong place,
- add `ESC` (using `pynpyt.keyboard.Listener`) to stop code when it uses mouse to draw (and when it is impossible to use mouse/keyboard for anything and PyAutoGUI moves mouse in wrong place), 

### 2021.05.04

- start it (put all code in one file, add missing imports and two missing functions)
