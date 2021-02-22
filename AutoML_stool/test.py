class B_VGGNet(object):

    def __init__(self):

        self.a = None
        self.b = []
        m = [self.a, self.b]

    def model(self):
    
        self.b = 2
        print(type(self.a))

x = B_VGGNet()
x.model()
