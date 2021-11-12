class Instance:
    def __init__(self, text, round):
        self.text = text
        self.uncertainties = []
        self.start_round = round

    def append_uncertainty(self, uncertainty):
        self.uncertainties.append(uncertainty)


