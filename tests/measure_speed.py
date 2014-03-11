from timeit import Timer
from gala import morpho
from gala import cmorpho
from scipy import misc
import numpy as np

def measure_flood_fill_speed():
    TRIALS = 2
    ATTEMPTS_PER_TRIAL = 100
    MATRIX_SHAPE = (512,512,10)
    VALUES = 3
    test_matrix = np.random.randint(0, VALUES, MATRIX_SHAPE)

    points = []
    for ii in range(0, ATTEMPTS_PER_TRIAL):
        points.append((random_point_in_matrix(MATRIX_SHAPE), [np.random.choice(VALUES)]))

    def time_with(testee):
        def timing_func():
            for starting_point, acceptable in points:
                result = testee.flood_fill(test_matrix, starting_point, acceptable, True)
        return timing_func

    for testee in [morpho, cmorpho, morpho, cmorpho]:
        print "Timing with %s" % (str(testee))
        action = time_with(testee)
        timer = Timer(action)
        time = timer.timeit(TRIALS)
        print "-- Time across %i attempts: %f seconds" % (TRIALS*ATTEMPTS_PER_TRIAL, time)

def random_point_in_matrix(shape):
    selected = []
    for dim_size in shape:
        selected.append(np.random.choice(dim_size))
    return tuple(selected)

def measure_nchoosek_speed():
    TRIALS = 100000
    for imp in ["from gala.cagglo import nchoosek",
                 "from scipy.misc import comb as nchoosek"]:
        timer = Timer("nchoosek(10000,20)", setup=imp)
        time = timer.timeit(TRIALS)
        print "-- %s time across %i trials: %f seconds" % (imp, TRIALS, time)

if __name__ == '__main__':
    measure_nchoosek_speed()
