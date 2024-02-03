import pyautogui as pygui

# если контроллер не активен, то ничего не делать
# if not MouseController.active:
#     return
# OR
# if MouseController.active:
#     pass


class MouseController:
    def __init__(self):
        self.active = False

    @staticmethod
    def click(coords: tuple, button: str = "left"):
        x, y = coords
        pygui.click(button=button, x=x, y=y)

    @staticmethod
    def push(coords: tuple, button: str = "left", down: bool = True):
        x, y = coords
        if down:
            pygui.mouseDown(button=button, x=x, y=y)
        else:
            pygui.mouseUp(button=button, x=x, y=y)

    @staticmethod
    def scroll(direction: int = 10, vertical: bool = True):
        if vertical:
            pygui.vscroll(direction)
        else:
            pygui.hscroll(direction)

    @staticmethod
    def move(coords: tuple):
        x, y = coords
        pygui.moveTo(x, y)

    @staticmethod
    def get_screen_size() -> tuple:
        return pygui.size()  # 16:10 screen proportions

    @staticmethod
    def get_screen_aspect_ratio() -> float:
        width, height = pygui.size()
        return height / width  # 16:10 screen proportions

    @staticmethod
    def get_position() -> tuple:
        return pygui.position()
