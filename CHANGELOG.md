### 2021.05.05

- add `\r` in when display messages with number of paths - now displays it in one line like in progress bars.
- add colors in messages (uses bash codes so may works only on Linux). Change `PRINT_COLORS = False` at the beginning of code to change it.
- display color in RGB instead of BGR - and also as hex code (ie. `#3F08C2`). It uses alos colors for this.
- display at start number of colors (it shows how many times we will have to change colors in painting program)
- display number of points in path

### 2021.05.04

- add debug messages
- set manually screen size (1920,1200) because I have two monitors and it was trying to drawing in wrong place,
- add `ESC` (using `pynpyt.keyboard.Listener`) to stop code when it uses mouse to draw (and it is impossible to use mouse/keyboard for anything), 

### 2021.05.04

- start it (put all code in one file, add missing imports and two missing functions)
