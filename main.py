from view.window import Window
from deep_learning.models.model_one.model_one import ModelOne
from deep_learning.models.model_two.model_two import ModelTwo
from deep_learning.models.model_three.model_three import ModelThree


class Main:
    def __init__(self) -> None:
        self.window = Window()
    
    def start_app(self):
        mode = int(input("1: Probar Modelo , 2: Entrenar modelo \n"))
        if mode == 1:
            self.window.run_window()
        if mode == 2:
            model = int(input("1: Primer modelo , 2: Segundo modelo, 3: Tercer modelo \n"))
            if model == 1:
                model = ModelOne()
                model.run()
            if model == 2:
                model = ModelTwo()
                model.run()
            if model == 3:
                model = ModelThree()
                model.run()

if __name__ == "__main__":
    main = Main()
    main.start_app()