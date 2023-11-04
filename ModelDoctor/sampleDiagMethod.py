from matplotlib import pyplot as plt

class DiagMethod():
    def __init__(self):
        pass

    '''
        datasetID:
            "cifar10"
            "cifar100"
            "mnist"
        model:
            the return of PTLoader.getModel()
    '''
    def loadModel(self,model,datasetID):
        patternImg=None

        #you need to make sure the format of patternImg is np.uint8 nparray and can be shown by DiagMethod(patternImg)
        return(patternImg)
    

    def showImg(self,patternImg):
        plt.imshow(patternImg)
        plt.title("original")
        plt.show()
