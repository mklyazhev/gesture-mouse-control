from typing import Optional, Tuple
from controller.mouse_controller import MouseController
from classifier.constants import gestures, hands

NO_GESTURE = gestures.get(11)
LIKE = gestures.get(1)

class GestureHandler:
    def __init__(self):
        self.queue = []
        self.controller = MouseController()
        self.commands = {
            "like": self.__activate_controller,
            "dislike": self.__deactivate_controller,
            "ok": self.__move,
            "one": self.__left_click,
            "two up": self.__right_click,
            "three": self.__middle_click,
            "fist": self.__press_left_button,
            "palm": self.__release_left_button,
            "fist-stop": self.__scroll_down,
            "stop-fist": self.__scroll_up,
            "stop-stop inverted": self.__scroll_right,
            "stop inverted-stop": self.__scroll_left,
        }
        # для более точной классификации, можно использовать усреднение предсказаний последних 3-5 кадров
        # нужно сделать систему запоминания последних двух жестов чтобы сделать динамические
        # сделать словарь, в котором к каждому жесту сопоставляется функция
        # для сопоставления передается очередь последних жестов, если в словаре не нашлось команды, то ничего не происходит
        # если предсказывать раз в несколько кадров и показывать 1 жест не меняя его, то можно многократно нажимать лкм
        # например если показывать палец вверх в течение секунды, то несколько раз нажмется лкм

    def process(self, gesture: str, args: dict):
        if gesture == NO_GESTURE:
            return

        if self.controller.active or gesture == LIKE:
            gesture = self.__process_queue()
            command = self.commands.get(gesture)

            if command:
                command(args)

            if len(self.queue) >= 5:  # Save last 5 gestures only
                self.queue = self.queue[-5:]

    def __process_queue(self):
        if self.queue[0] == self.queue[-1]:
            return self.queue[-1]
        else:
            return "-".join((self.queue[0], self.queue[-1]))

    def __activate_controller(self, *args):
        self.controller.active = True

    def __deactivate_controller(self, *args):
        self.controller.active = False

    def __move(self, args):
        coords = args.get("coords")
        self.controller.move(coords)

    def __left_click(self, args):
        coords = args.get("coords")
        self.controller.click(coords, button="left")

    def __right_click(self, args):
        coords = args.get("coords")
        self.controller.click(coords, button="right")

    def __middle_click(self, args):
        coords = args.get("coords")
        self.controller.click(coords, button="middle")

    def __press_left_button(self, args):
        coords = args.get("coords")
        self.controller.push(coords, button="left", down=True)

    def __release_left_button(self, args):
        coords = args.get("coords")
        self.controller.push(coords, button="left", down=False)

    def __scroll_down(self, *args):
        self.controller.scroll(-30)

    def __scroll_up(self, *args):
        self.controller.scroll(30)

    def __scroll_right(self, *args):
        self.controller.scroll(30, vertical=False)

    def __scroll_left(self, *args):
        self.controller.scroll(-30, vertical=False)
