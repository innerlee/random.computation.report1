# project

### dataset

[MovieLens 20M](http://grouplens.org/datasets/movielens/20m/)

### report
* [project.pdf](project.pdf)

### codes
* [``src/project.jl``](src/project.jl)

----

# hw2

### report
* [hw2.pdf](hw2.pdf)

### codes
* [``src/hw2.jl``](src/hw2.jl)
* [``src/l1.solver.jl``](src/l1.solver.jl)

### raw data
* [``result/l1.100.txt``](result/l1.100.txt)
* [``result/l1.500.txt``](result/l1.500.txt)

----

# hw1

* details see [report.pdf](report.pdf).
* codes are in [src/](src/) and written in [Julia](http://julialang.org/).
* raw outputs from codes are availble at [result/](result/) folder.

## data preparation

download dataset and extract ``train.csv`` into ``data/``.
make an ``output/`` dir for save results of data preprocessing.

## run codes

* first run the functions in ``preprocess.jl`` to do preprocessing.
* run ``process.jl`` to see baseline results.
* ``julia gaussian.jl 100 1000`` to run Gaussian skeching 100 times with parameter ``k=1000``.
* similar for other methods.
