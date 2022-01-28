from src.experiment_setups.experiment_setups import create_logspace, perform_experiments
import numpy as np
from global_ import gpu_memory_
from src.gpu.memory_allocation import set_gpu_memory

set_gpu_memory(gpu_memory_)

Cs = create_logspace(0.5, 100, 4)
eps_factors = [0.5]
taus = [50]
tol = 0.0008

# #####################################################################################################################
# L1 CGAVI & L2 CGAVI & AGDAVI
data_sets = ['banknote', 'cancer', 'htru2', 'iris', 'seeds', 'wine']
psis_oavi = list(create_logspace(0.04, 0.9, 14))
lmbdas_oavi = list(np.linspace(0, 5, 8))
hyperparameters_l1_cgavi = {"algorithm": "l1_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_l2_cgavi = {"algorithm": "l2_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_agdavi = {"algorithm": "agdavi",
                          "psi": psis_oavi,
                          "lmbda": lmbdas_oavi,
                          "tau": taus,
                          "C": Cs,
                          'tol': tol}
perform_experiments(data_sets, hyperparameters_l1_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_l2_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_agdavi, saving=True)


data_sets = ['sonar']
psis_oavi = list(create_logspace(0.1, 0.9, 14))
hyperparameters_l1_cgavi = {"algorithm": "l1_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_l2_cgavi = {"algorithm": "l2_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_agdavi = {"algorithm": "agdavi",
                          "psi": psis_oavi,
                          "lmbda": lmbdas_oavi,
                          "tau": taus,
                          "C": Cs,
                          'tol': tol}
perform_experiments(data_sets, hyperparameters_l1_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_l2_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_agdavi, saving=True)


data_sets = ['spam']
psis_oavi = list(create_logspace(0.01, 0.9, 14))
hyperparameters_l1_cgavi = {"algorithm": "l1_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_l2_cgavi = {"algorithm": "l2_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_agdavi = {"algorithm": "agdavi",
                          "psi": psis_oavi,
                          "lmbda": lmbdas_oavi,
                          "tau": taus,
                          "C": Cs,
                          'tol': tol}
perform_experiments(data_sets, hyperparameters_l1_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_l2_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_agdavi, saving=True)


data_sets = ['voice']
psis_oavi = list(create_logspace(0.12, 0.9, 14))
lmbdas_oavi = list(np.linspace(0, 3, 7))
hyperparameters_l1_cgavi = {"algorithm": "l1_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_l2_cgavi = {"algorithm": "l2_cgavi",
                            "psi": psis_oavi,
                            "eps_factor": eps_factors,
                            "lmbda": lmbdas_oavi,
                            "tau": taus,
                            "C": Cs,
                            'tol': tol}
hyperparameters_agdavi = {"algorithm": "agdavi",
                          "psi": psis_oavi,
                          "lmbda": lmbdas_oavi,
                          "tau": taus,
                          "C": Cs,
                          'tol': tol}
perform_experiments(data_sets, hyperparameters_l1_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_l2_cgavi, saving=True)
perform_experiments(data_sets, hyperparameters_agdavi, saving=True)
# #####################################################################################################################
# VCA
psis_vca = create_logspace(0.05, 10, 40)
Cs_vca = create_logspace(0.5, 100, 10)
hyperparameters_vca = {"algorithm": "vca", "psi": psis_vca, "C": Cs_vca}
data_sets = ['banknote', 'cancer', 'htru2', 'iris', 'seeds', 'sonar', 'voice', 'wine']
perform_experiments(data_sets, hyperparameters_vca, saving=True)

psis_vca = create_logspace(0.15, 10, 35)
hyperparameters_vca = {"algorithm": "vca", "psi": psis_vca, "C": Cs_vca}
data_sets = ['spam']
perform_experiments(data_sets, hyperparameters_vca, saving=True)

# #####################################################################################################################
# AVI
psis_avi = create_logspace(0.05, 10, 20)
taus = list(np.linspace(0, 0.3, 4))
hyperparameters_avi = {"algorithm": "avi", "psi": psis_avi, "tau": taus, "C": Cs}
data_sets = ['banknote', 'cancer', 'htru2', 'iris', 'seeds', 'wine']
perform_experiments(data_sets, hyperparameters_avi, saving=True)

psis_avi = create_logspace(0.15, 10, 20)
hyperparameters_avi = {"algorithm": "avi", "psi": psis_avi, "tau": taus, "C": Cs}
data_sets = ['sonar', 'voice']
perform_experiments(data_sets, hyperparameters_avi, saving=True)

psis_avi = create_logspace(0.4, 10, 20)
hyperparameters_avi = {"algorithm": "avi", "psi": psis_avi, "tau": taus, "C": Cs}
data_sets = ['spam']
perform_experiments(data_sets, hyperparameters_avi, saving=True)


# #####################################################################################################################
# SVM

data_sets = ['banknote', 'cancer', 'iris', 'seeds', 'sonar', 'spam', 'voice', 'wine']


Cs = create_logspace(0.5, 100, 4)
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hyperparameters_svm = {"algorithm": "svm",
                       "C": Cs,
                       'degree': degrees}

perform_experiments(data_sets, hyperparameters_svm, saving=True)


data_sets = ['htru2']

Cs = create_logspace(0.5, 100, 4)
degrees = [1, 2, 3, 4, 5, 6]
hyperparameters_svm = {"algorithm": "svm",
                       "C": Cs,
                       'degree': degrees}

perform_experiments(data_sets, hyperparameters_svm, saving=True)

