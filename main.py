from view.window import Window
from deep_learning.models.model_one.model_one import ModelOne

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
                pass
            if model == 3:
                pass

if __name__ == "__main__":
    main = Main()
    main.start_app()