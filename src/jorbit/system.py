class System:
    def __init__(self):
        pass

    # break up the combinations of particles into "likelihood blocks", where each block
    # is the minimum number of particles needed to compute the likelihood of *1* set
    # of observations. This is useful for parallelizing the likelihood computation.
