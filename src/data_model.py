class ResultRow:
    def __self__(self, name, x, y, height, width, label):
        self.name = name,
        self.x = x,
        self.y = y,
        self.height = height,
        self.width = width
        self.label = label

    def __init__(self, name, x, y, height, width, label):
        self.name = name,
        self.x = x,
        self.y = y,
        self.height = height,
        self.width = width
        self.label = label

    def to_dict(self):
        return {
            'path': self.name[0],
            'x': self.x[0],
            'y': self.y[0],
            'height': self.height[0],
            'width': self.width,
            'label': self.label
        }
