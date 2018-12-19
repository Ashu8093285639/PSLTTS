from import Individu


class BackpropagationPSO(BackpropagationNN):
            """docstring for BackpropagationPSO"""
            def __init__(self, arg):
                super(BackpropagationPSO, self).__init__()
                self.arg = arg

            def initWeight(self, partikel):
                layer = list()
                partikel_dimens_idx = 0
                input_to_hidden = list()

                for i in range(self.HIDDEN_LAYER):
                    w = list()
                    for j in range(self.INPUT_LAYER + 1):
                        w.append(partikel[partikel_dimens_idx])
                        partikel_dimens_idx += 1
                    input_to_hidden.append(Neuron(w))
                layer.append(input_to_hidden)

                hidden_to_output = list()
                for i in range(self.OUTPUT_LAYER):
                    w = list()
                    for j in range(self.HIDDEN_LAYER + 1):
                        w.append(partikel[partikel_dimens_idx])
                        partikel_dimens_idx += 1
                    hidden_to_output.append(Neuron(w))
                layer.append(hidden_to_output)
                self.net = layer

class BackpropagationParticle(Individu):

    def __init__(self, arg):
        super(BackpropagationParticle, self).__init__()

    def get_fitness():
        backPro = BackpropagationPSO(5, 3, 1, 0.4)
        backPro.initWeight(self.position)
        backPro.training()
        fitness = backPro.data_testing()

