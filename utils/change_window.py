import pygetwindow as gw
import ctypes
import win32gui
from log import log

EURO_TITLE = "Euro Truck Simulator 2"  # Windows title for Euro Truck Simulator 2


def get_window_position(title):
    try:
        window = gw.getWindowsWithTitle(title)[0]  # get the first matching window
        return window.topleft, window.bottomright
    except IndexError:
        print(f"No window with title '{title}' found.")
        return None, None


def move_window(title, x, y):
    try:
        window = gw.getWindowsWithTitle(title)[0]  # get the first matching window
        window.moveTo(x, y)
    except IndexError:
        print(f"No window with title '{title}' found.")


# move window to the top left corner
def set_window_topleft():
    move_window(EURO_TITLE, -8, 0)


def is_window_visible(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        return window.visible  # check if the window is visible
    except IndexError:
        return False  # no window found with that title


def is_window_active(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        return window.isActive  # check if the window is active
    except IndexError:
        return False  # no window found with that title


def restore_window(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        if window.isMinimized:  # minimized
            window.restore()  # restore the window
    except IndexError:
        print(f"Window titled '{window_title}' not found.")


# check and correct the window state
def correction_window():
    if not is_window_visible(EURO_TITLE):
        print(f"{EURO_TITLE} is not visible.")
        restore_window(EURO_TITLE)  # restore the window if it is minimized
        gw.getWindowsWithTitle(EURO_TITLE)[0].activate()  # activate the window
        set_window_topleft()

    elif not is_window_active(EURO_TITLE):
        print(f"{EURO_TITLE} is in the background.")
        restore_window(EURO_TITLE)  # restore the window if it is minimized
        gw.getWindowsWithTitle(EURO_TITLE)[0].activate()  # activate the window
        set_window_topleft()

    else:
        print(f"{EURO_TITLE} is visible and active.")


def get_window_resolution(window_title):
    # get the handle of the window by its title
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd:
        # use ctypes to get the client rectangle of the window
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        return width, height
    else:
        print(f"Window with title '{window_title}' not found.")
        return None


# check if the window resolution matches the expected values
def check_window_resolution_same(weight, height):
    resolution = get_window_resolution(EURO_TITLE)
    log.debug(f"Actual resolution：{resolution}")
    return resolution[0] == weight and resolution[1] == height
