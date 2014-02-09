import numpy as np
from gala import morpho
from timeit import Timer

def measure_flood_fill_speed():
    TRIALS = 100
    setup="""import numpy as np
from gala import morpho
MATRIX_SHAPE = (512,512,10)
VALUES = 3
test_matrix = np.random.randint(0, VALUES, MATRIX_SHAPE)

def random_point_in_matrix(shape):
    selected = []
    for dim_size in shape:
        selected.append(np.random.choice(dim_size))
    return tuple(selected)
    """
    action="""starting_point = random_point_in_matrix(MATRIX_SHAPE)
acceptable = [np.random.choice(VALUES)]
result = morpho.flood_fill(test_matrix, starting_point, acceptable, True)
"""

    timer = Timer(action, setup=setup)
    time = timer.timeit(TRIALS)
    print "Time across %i trials: %f seconds" % (TRIALS, time)


if __name__ == '__main__':
    measure_flood_fill_speed()
