#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import random
import array

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import shutil
import pickle

from itertools import chain


def mutDE(y, a, b, c, f, lb, ub):
    """
    Perform bounded differential evolution mutation.

    This function implements the mutation step in differential evolution (DE) algorithms
    with boundary constraints. It updates the vector `y` by adding a scaled difference of
    vectors `b` and `c` to the vector `a`. The result is clipped to remain within the bounds
    specified by `lb` (lower bounds) and `ub` (upper bounds).

    Parameters
    ----------
    y : list or np.ndarray
        The target vector to be mutated. This vector will be modified in place.
    a : list or np.ndarray
        The base vector for the mutation.
    b : list or np.ndarray
        The first difference vector.
    c : list or np.ndarray
        The second difference vector.
    f : float
        Scaling factor for the difference between `b` and `c`.
    lb : list or np.ndarray
        Lower bounds for each dimension, ensuring that the mutated values do not go below these bounds.
    ub : list or np.ndarray
        Upper bounds for each dimension, ensuring that the mutated values do not exceed these bounds.

    Returns
    -------
    list or np.ndarray
        The mutated vector `y` after applying the bounded differential evolution operation.

    Notes
    -----
    The function modifies `y` in place, and the resulting `y` values are clipped between `lb` and `ub`.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.5, 0.2, 0.8])
    >>> a = np.array([0.4, 0.1, 0.9])
    >>> b = np.array([0.6, 0.3, 0.7])
    >>> c = np.array([0.2, 0.4, 0.6])
    >>> f = 0.5
    >>> lb = np.array([0.0, 0.0, 0.0])
    >>> ub = np.array([1.0, 1.0, 1.0])
    >>> mutDE(y, a, b, c, f, lb, ub)
    array([0.5 , 0.2 , 0.75])
    """
    size = len(y)
    for i in range(size):
        y[i] = np.clip(a[i] + f * (b[i] - c[i]), lb[i], ub[i])
    return y


def cxBinomial(x, y, cr, lb, ub):
    """
    Perform a bounded differential evolution binomial mating operation.

    This function implements the binomial crossover operator used in
    differential evolution algorithms. The crossover is performed between
    two parents (`x` and `y`) based on the crossover rate `cr`. The resulting
    offspring `x` is clipped within the bounds specified by `lb` and `ub`.

    Parameters
    ----------
    x : list or np.ndarray
        The first parent, which will also be modified to hold the offspring.
    y : list or np.ndarray
        The second parent, used as the donor vector.
    cr : float
        The crossover rate, a probability value (0 <= cr <= 1).
    lb : list or np.ndarray
        The lower bounds for each dimension.
    ub : list or np.ndarray
        The upper bounds for each dimension.

    Returns
    -------
    x : list or np.ndarray
        The offspring produced by the crossover, with values clipped
        to the specified bounds.

    Notes
    -----
    A random index is selected, ensuring at least one crossover point. For
    each element in `x`, the corresponding element in `y` is used if a
    random number is less than the crossover rate `cr` or if the current
    index matches the randomly selected index. The result is then clipped
    to the range [lb, ub].

    Examples
    --------
    >>> x = [0.1, 0.2, 0.3]
    >>> y = [0.5, 0.6, 0.7]
    >>> lb = [0.0, 0.0, 0.0]
    >>> ub = [1.0, 1.0, 1.0]
    >>> cr = 0.9
    >>> cxBinomial(x, y, cr, lb, ub)
    [0.5, 0.6, 0.3]  # Result may vary due to randomness

    """
    size = len(x)
    index = random.randrange(size)
    for i in range(size):
        if i == index or random.random() < cr:
            x[i] = np.clip(y[i], lb[i], ub[i])
    return x


def cxExponential(x, y, cr, lb, ub):
    """
    Bounded differential evolution exponential selection operator.

    This function applies an exponential crossover operator commonly used in
    differential evolution algorithms. It takes two parent vectors `x` and `y`,
    and produces a modified offspring `x` within the given lower (`lb`) and upper
    (`ub`) bounds. The crossover probability `cr` controls the likelihood of
    crossover continuation from a randomly chosen starting index.

    The crossover works by selecting a random starting index and sequentially
    copying elements from the second parent (`y`) into the first parent (`x`)
    within the bounds provided (`lb`, `ub`). The copying continues until a
    random number exceeds the crossover probability `cr`.

    Parameters
    ----------
    x : list or numpy.ndarray
        The first parent vector, which will be modified in-place.
    y : list or numpy.ndarray
        The second parent vector used for the crossover operation.
    cr : float
        Crossover probability, controlling how far the crossover operation will
        continue before stopping.
    lb : list or numpy.ndarray
        Lower bounds for the values in the resulting vector `x`.
    ub : list or numpy.ndarray
        Upper bounds for the values in the resulting vector `x`.

    Returns
    -------
    list or numpy.ndarray
        The modified vector `x` after applying the exponential crossover operator.

    Notes
    -----
    - This operator ensures that the resulting values in `x` stay within the
      provided lower (`lb`) and upper (`ub`) bounds.
    - The loop performs a wrapping traversal of the indices, starting from a
      randomly chosen point and wrapping around to the beginning, ensuring
      a complete crossover when necessary.

    Examples
    --------
    >>> import numpy as np
    >>> import random
    >>> from itertools import chain
    >>> x = np.array([0.5, 0.6, 0.7])
    >>> y = np.array([1.0, 1.1, 1.2])
    >>> lb = np.array([0.0, 0.0, 0.0])
    >>> ub = np.array([1.0, 1.0, 1.0])
    >>> cr = 0.5
    >>> random.seed(42)  # For reproducibility
    >>> cxExponential(x, y, cr, lb, ub)
    array([1. , 0.6, 0.7])
    """
    size = len(x)
    index = random.randrange(size)
    # Loop on the indices index -> end, then on 0 -> index
    for i in chain(range(index, size), range(0, index)):
        x[i] = np.clip(y[i], lb[i], ub[i])
        if random.random() < cr:
            break
    return x


def uniform(low, up, size=None):
    """
    Generate a list of uniformly distributed random numbers within a specified range.

    This function initializes a list of random numbers where each number is drawn from a uniform distribution
    between corresponding `low` and `up` bounds. If `low` and `up` are scalars, the list is initialized
    with `size` random numbers between `low` and `up`. If they are sequences, the function will generate
    random numbers for each pair of corresponding elements in `low` and `up`.

    Parameters
    ----------
    low : float or sequence of floats
        Lower bound(s) of the uniform distribution.
        If a single float is provided, it will be used as the lower bound for all random numbers.
        If a sequence is provided, each corresponding element will serve as the lower bound for
        the respective random number.

    up : float or sequence of floats
        Upper bound(s) of the uniform distribution.
        If a single float is provided, it will be used as the upper bound for all random numbers.
        If a sequence is provided, each corresponding element will serve as the upper bound for
        the respective random number.

    size : int, optional
        The number of random numbers to generate, if `low` and `up` are scalar values.
        If not provided, the function will infer the size from the length of `low` and `up` when
        they are sequences.

    Returns
    -------
    list of float
        A list of random numbers generated from the uniform distribution within the specified bounds.

    Raises
    ------
    TypeError
        If `low` or `up` are sequences and `size` is not provided, or if the dimensions of `low` and `up`
        do not match.

    Examples
    --------
    Generate 3 random numbers between 0 and 1:

    >>> uniform(0, 1, size=3)
    [0.450598, 0.325857, 0.846145]

    Generate 2 random numbers where the bounds are specified by sequences:

    >>> uniform([0, 10], [5, 20])
    [3.175839, 15.732489]
    """
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


class DifferentialEvolution:
    """Differential Evolution class for global optimization.

    Implements the Differential Evolution (DE) algorithm for minimizing a
    nonlinear function.

    Parameters
    ----------
    obj : callable
        The objective function to be minimized. It takes a single argument
        which is a 1D array representing the solution vector, and returns a
        scalar representing the fitness value.
    ndim : int
        The number of dimensions of the problem.
    lb : array_like
        The lower bounds of the search space. Array must have shape (ndim,).
    ub : array_like
        The upper bounds of the search space. Array must have shape (ndim,).
    cr : float, optional
        The crossover probability (default is 0.25).
    f : float, optional
        The scaling factor for the mutation step (default is 1.0).
    mu : int, optional
        The population size (default is 300).
    ngen : int, optional
        The number of generations (default is 200).
    weight : tuple, optional
        The weights for the fitness function. Defaults to (-1.0,).
    pool : None or Pool-like object, optional
        The pool of workers for parallel evaluation. If None, uses a
        serial implementation.

    Attributes
    ----------
    toolbox : base.Toolbox
        A DEAP toolbox containing registered operators.
    pop : list
        The current population of individuals.
    hof : tools.HallOfFame
        The Hall of Fame storing the best individual.
    fitnesses : list
        The fitness values of the current population.
    stats : tools.Statistics
        Statistics object for tracking population fitness.
    logbook : tools.Logbook
        A logbook object for recording statistics during evolution.

    Methods
    -------
    optimize()
        Performs the DE optimization and returns the logbook object.
    """

    def __init__(
        self,
        obj,
        ndim,
        lb,
        ub,
        cr=0.25,
        f=1,
        mu=300,
        ngen=200,
        weight=(-1.0,),
        pool=None,
    ):

        # populate attributes
        self.obj = obj
        self.ndim = ndim
        self.lb = lb
        self.ub = ub
        self.cr = cr
        self.f = f
        self.mu = mu
        self.ngen = ngen
        self.weight = weight
        self.pool = pool
        self.creator = creator

        # create fitness function and individual prototype
        self.creator.create("FitnessMin", base.Fitness, weights=self.weight)
        self.creator.create(
            "Individual", array.array, typecode="d", fitness=self.creator.FitnessMin
        )

        # initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", uniform, self.lb, self.ub, self.ndim)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            self.creator.Individual,
            self.toolbox.attr_float,
            self.ndim,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("mutate", mutDE, f=0.8, lb=self.lb, ub=self.ub)
        self.toolbox.register("mate", cxExponential, cr=0.8, lb=self.lb, ub=self.ub)
        self.toolbox.register("select", tools.selRandom, k=3)
        self.toolbox.register("evaluate", self.obj)

        # initialize parallel map if necessary
        if self.pool is not None:
            self.toolbox.register("map", self.pool.map)

        # initialize stats and logbooks
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max", "hof"
        self.record = 0.0

        # Start a new evolution
        self.pop = self.toolbox.population(n=self.mu)
        self.hof = tools.HallOfFame(1)
        self.fitnesses = [0.0] * self.mu

    def optimize(self):

        # Evaluate the individuals
        self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = fit

        self.record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(self.pop), **self.record)
        print(self.logbook.stream)

        for g in range(1, self.ngen):
            children = []
            for agent in self.pop:
                # We must clone everything to ensure independance
                a, b, c = [
                    self.toolbox.clone(ind) for ind in self.toolbox.select(self.pop)
                ]
                x = self.toolbox.clone(agent)
                y = self.toolbox.clone(agent)
                y = self.toolbox.mutate(y, a, b, c)
                z = self.toolbox.mate(x, y)
                del z.fitness.values
                children.append(z)

            self.fitnesses = self.toolbox.map(self.toolbox.evaluate, children)
            for (i, ind), fit in zip(enumerate(children), self.fitnesses):
                ind.fitness.values = fit
                if ind.fitness > self.pop[i].fitness:
                    self.pop[i] = ind

            self.hof.update(self.pop)
            self.record = self.stats.compile(self.pop)
            self.logbook.record(
                gen=g, evals=len(self.pop), **self.record, hof=list(self.hof[0])
            )

            # print stats
            print(self.logbook.stream)

        print("Best individual is ", self.hof[0])
        print("with fitness", self.hof[0].fitness.values[0])

        return self.logbook


class NSGAII:
    """NSGA-II (Non-dominated Sorting Genetic Algorithm II) class

    This class implements the NSGA-II algorithm for multi-objective optimization.

    Parameters
    ----------
    obj : function
        The objective function to be minimized. This function should take a single
        argument, which is an individual (array-like) and return a list of fitness
        values (one for each objective).
    ndim : int
        The number of dimensions (variables) in the problem.
    lb : array-like
        The lower bounds of the search space (one for each dimension).
    ub : array-like
        The upper bounds of the search space (one for each dimension).
    eta : float, optional (default: 20.0)
        The crowding distance parameter for diversity preservation.
    cxpb : float, optional (default: 0.9)
        The crossover probability.
    mu : int, optional (default: 300)
        The population size.
    ngen : int, optional (default: 200)
        The number of generations.
    weight : tuple, optional (default: (-1.0, -1.0))
        The weights for each objective function. A negative weight indicates
        minimization, while a positive weight indicates maximization.
    pool : None or multiprocessing.Pool, optional
        A multiprocessing pool to be used for parallel evaluation.

    Attributes
    ----------
    creator : deap.creator
        A DEAP creator object used to create fitness and individual classes.
    toolbox : deap.base.Toolbox
        A DEAP toolbox object holding various genetic operators.
    stats : deap.tools.Statistics
        A DEAP statistics object to track population statistics.
    logbook : deap.tools.Logbook
        A DEAP logbook object to record population information during evolution.
    record : float
        The record of the best individual's fitness values.
    fitnesses : list
        A list to store fitness values of individuals during evaluation.
    pop : list of deap.creator.Individual
        The current population of individuals.
    pop_list : list of list of deap.creator.Individual
        A list to store population history across generations.

    Methods
    -------
    optimize()
        Optimizes the objective function using the NSGA-II algorithm.

    Returns
    -------
    tuple (pop, logbook, stats, pop_list)
        A tuple containing:
            - pop: The final population of individuals.
            - logbook: The logbook object with population information.
            - stats: The statistics object with population statistics.
            - pop_list: A list of populations across generations.
    """

    def __init__(
        self,
        obj,
        ndim,
        lb,
        ub,
        eta=20.0,
        cxpb=0.9,
        mu=300,
        ngen=200,
        weight=(-1.0, -1.0),
        pool=None,
    ):

        # populate attributes
        self.obj = obj
        self.ndim = ndim
        self.lb = lb
        self.ub = ub
        self.eta = eta
        self.cxpb = cxpb
        self.mu = mu
        self.ngen = ngen
        self.weight = weight
        self.pool = pool
        self.creator = creator
        self.toolbox = base.Toolbox()

        # create multiple obj fitness and individual
        self.creator.create("FitnessMin", base.Fitness, weights=self.weight)
        self.creator.create(
            "Individual", array.array, typecode="d", fitness=self.creator.FitnessMin
        )

        # initialize toolbox
        self.toolbox.register("attr_float", uniform, self.lb, self.ub, self.ndim)
        self.toolbox.register(
            "individual",
            tools.initIterate,
            self.creator.Individual,
            self.toolbox.attr_float,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.obj)
        self.toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=self.lb,
            up=self.ub,
            eta=self.eta,
        )
        self.toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=self.lb,
            up=self.ub,
            eta=self.eta,
            indpb=1.0 / self.ndim,
        )
        self.toolbox.register("select", tools.selNSGA2)

        # initialize parallel map if necessary
        if self.pool is not None:
            self.toolbox.register("map", self.pool.map)

        # initialize statistics and bookeeping
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        self.record = 0.0
        self.fitnesses = [0.0] * self.mu

        # initialize population
        self.pop = self.toolbox.population(n=self.mu)
        self.pop_list = []

    def optimize(self):
        """
        Optimize the defined objective function
        """

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self.fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        self.pop = self.toolbox.select(self.pop, len(self.pop))

        self.record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **self.record)
        print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, self.ngen):

            # Vary the population
            offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.cxpb:
                    self.toolbox.mate(ind1, ind2)

                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, self.fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            self.pop = self.toolbox.select(self.pop + offspring, self.mu)
            self.pop_list.append(self.pop)
            self.record = self.stats.compile(self.pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **self.record)
            print(self.logbook.stream)

        return self.pop, self.logbook, self.stats, self.pop_list


class NSGAIII:
    """
    NSGA-III (Non-dominated Sorting Genetic Algorithm III)

    This class implements the NSGA-III algorithm for multi-objective optimization.

    Parameters
    ----------
    obj : function
        The objective function to be minimized. It should take a single
        argument (individual) and return a list of fitness values.
    ndim : int
        The number of dimensions of the problem.
    lb : list
        A list of lower bounds for each dimension.
    ub : list
        A list of upper bounds for each dimension.
    ref_points : list
        A list of reference points for the objective space.
    eta : float, optional
        The crowding distance parameter for the selection operator (default: 20.0).
    cxpb : float, optional
        The probability of crossover (default: 0.9).
    mutpb : float, optional
        The probability of mutation (default: 1.0).
    mu : int, optional
        The population size (default: 300).
    ngen : int, optional
        The number of generations (default: 200).
    weight : tuple, optional
        The weights for the objective function (default: (-1.0, -1.0)).
    pool : None or multiprocessing.Pool, optional
        A multiprocessing pool to use for parallel evaluation (default: None).

    Attributes
    ----------
    creator : deap.creator
        The creator object from DEAP.
    toolbox : deap.base.Toolbox
        The toolbox object from DEAP.

    Methods
    -------
    optimize
        Optimizes the defined objective function and returns the final population,
        logbook, statistics, and population history.

    """

    def __init__(
        self,
        obj,
        ndim,
        lb,
        ub,
        ref_points,
        eta=20.0,
        cxpb=0.9,
        mutpb=1.0,
        mu=300,
        ngen=200,
        weight=(-1.0, -1.0),
        pool=None,
    ):

        # populate attributes
        self.obj = obj
        self.ndim = ndim
        self.lb = lb
        self.ub = ub
        self.ref_points = ref_points
        self.eta = eta
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.mu = mu
        self.ngen = ngen
        self.weight = weight
        self.pool = pool
        self.creator = creator
        self.toolbox = base.Toolbox()

        # create multiple obj fitness and individual
        self.creator.create("FitnessMin", base.Fitness, weights=self.weight)
        self.creator.create(
            "Individual", array.array, typecode="d", fitness=self.creator.FitnessMin
        )

        # initialize toolbox
        self.toolbox.register("attr_float", uniform, self.lb, self.ub, self.ndim)
        self.toolbox.register(
            "individual",
            tools.initIterate,
            self.creator.Individual,
            self.toolbox.attr_float,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.obj)
        self.toolbox.register(
            "mate",
            tools.cxSimulatedBinaryBounded,
            low=self.lb,
            up=self.ub,
            eta=self.eta,
        )
        self.toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=self.lb,
            up=self.ub,
            eta=self.eta,
            indpb=1.0 / self.ndim,
        )
        self.toolbox.register("select", tools.selNSGA3, refpoints=self.ref_points)

        # initialize parallel map if necessary
        if self.pool is not None:
            self.toolbox.register("map", self.pool.map)

        # initialize statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        # initialize bookkeeping
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        self.record = 0.0
        self.fitnesses = [0.0] * self.mu

        # initialize population and population history
        self.pop = self.toolbox.population(n=self.mu)
        self.pop_history = []

    def optimize(self):
        """
        Optimize the defined objective function
        """

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self.fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        self.record = self.stats.compile(self.pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **self.record)
        print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, self.ngen):

            # generate next generation
            offspring = algorithms.varAnd(self.pop, self.toolbox, self.cxpb, self.mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, self.fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            self.pop = self.toolbox.select(self.pop + offspring, self.mu)

            # Compile statistics about the new population
            self.record = self.stats.compile(self.pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **self.record)
            print(self.logbook.stream)

        return self.pop, self.logbook, self.stats, self.pop_history
