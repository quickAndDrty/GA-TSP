
class City:

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def getId(self):
        return self.id

    def getX(self):
        return int(self.x)

    def getY(self):
        return int(self.y)

    def getCity(self):
        return (self.id + " " + self.x + " " + self.y)