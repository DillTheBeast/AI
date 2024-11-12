class animal:
    def __init__(self, name):
        self.name = name
    def PrintVal(self):
        print(self.name)
        
class dog(animal):
    def __init__(self, name, breed):
        self.breed = breed
        animal.__init__(self, name)
        
a = animal("dog")
a.PrintVal()
d = dog("name","thing")
d.PrintVal()